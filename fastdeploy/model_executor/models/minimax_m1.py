from __future__ import annotations
import math
import re
from typing import Optional
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
    ReplicatedLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.utils import default_weight_loader, process_weights_after_loading
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import lightning_attention, linear_decode_forward_triton
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce

class RMSNormTP(nn.Layer):
    def __init__(self, fd_config: FDConfig, hidden_size: int, prefix: str, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.prefix = prefix
        self.weight_key = f"{prefix}.weight"
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank
        shard_size = hidden_size // self.tp_size
        if hidden_size % self.tp_size != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by tp_size ({self.tp_size}) for RMSNormTP")
        self.weight = self.create_parameter(
            shape=[shard_size],
            default_initializer=nn.initializer.Constant(1.0),
            dtype="float32"
        )
        # ----------------- 【核心修改 1】 -----------------
        # 将实例方法 shard_weight_loader 附加到 weight 参数上
        self.weight.weight_loader = self.shard_weight_loader
        # ----------------------------------------------

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.cast("float32")
        variance = x_float.pow(2).mean(axis=-1, keepdim=True)
        if self.tp_size > 1:
            tensor_model_parallel_all_reduce(variance)
            variance = variance / self.tp_size
        inv_std = paddle.rsqrt(variance + self.eps)
        norm_out = (x_float * inv_std).cast(orig_dtype) * self.weight
        return norm_out

    # ----------------- 【核心修改 2】 -----------------
    # 移除 @staticmethod，并将其变为一个标准的实例方法
    def shard_weight_loader(self, param, loaded_weight):
        """Custom loader to shard the full weight."""
        full_weight = get_tensor(loaded_weight)
        shard_size = full_weight.shape[0] // self.tp_size
        my_shard = full_weight[self.tp_rank * shard_size : (self.tp_rank + 1) * shard_size]
        param.set_value(my_shard.cast(param.dtype))
    # ----------------------------------------------

# +++++++++++++++ 新增: 线性注意力模块 +++++++++++++++
class MiniMaxM1LinearAttention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id
        
        tp_size = fd_config.parallel_config.tensor_parallel_size
        tp_rank = fd_config.parallel_config.tensor_parallel_rank
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.tp_heads = self.num_heads // tp_size
        
        hidden_inner_size = self.head_dim * self.num_heads

        self.qkv_proj = ColumnParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", input_size=config.hidden_size, output_size=hidden_inner_size * 3, with_bias=False)
        self.output_gate = ColumnParallelLinear(fd_config, prefix=f"{prefix}.output_gate", input_size=config.hidden_size, output_size=hidden_inner_size, with_bias=False)
        self.out_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.out_proj", input_size=hidden_inner_size, output_size=config.hidden_size, with_bias=False)
        
        # 使用我们新定义的 RMSNormTP
        self.norm = RMSNormTP(fd_config, hidden_size=hidden_inner_size, prefix=f"{prefix}.norm", eps=1e-5)

        slope_rate = self._build_slope_tensor(self.num_heads)
        if config.num_hidden_layers > 1:
            self.slope_rate = slope_rate * (1 - layer_id / (config.num_hidden_layers - 1) + 1e-5)
        else:
            self.slope_rate = slope_rate * (1 + 1e-5)
        
        self.tp_slope = self.slope_rate[tp_rank * self.tp_heads : (tp_rank + 1) * self.tp_heads].contiguous()

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n_attention_heads).is_integer():
            slopes = get_slopes_power_of_2(n_attention_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_attention_heads))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2) + 
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_attention_heads - closest_power_of_2]
            )
        
        return paddle.to_tensor(slopes, dtype='float32').reshape([n_attention_heads, 1, 1])

    
    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        model_dtype = self.out_proj.weight.dtype
        # hidden_states: (total_tokens, hidden_size)
        total_tokens = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states) # (total_tokens, 3 * inner_hidden_size_tp)
        qkv_act = F.silu(qkv)
        
        q, k, v = qkv_act.split(3, axis=-1) # Each is (total_tokens, inner_hidden_size_tp)

        # Reshape for attention computation
        # (total_tokens, tp_heads, head_dim)
        q = q.reshape((total_tokens, self.tp_heads, self.head_dim))
        k = k.reshape((total_tokens, self.tp_heads, self.head_dim))
        v = v.reshape((total_tokens, self.tp_heads, self.head_dim))
        
        if forward_meta.forward_mode.is_prefill():
            # Prefill 逻辑暂时保持不变，因为 dummy_run 的 decode 阶段才会触发错误
            # Prefill expects (B, H, N, D), assuming B=1 for now.
            q = q.transpose((1, 0, 2)).unsqueeze(0)
            k = k.transpose((1, 0, 2)).unsqueeze(0)
            v = v.transpose((1, 0, 2)).unsqueeze(0)

            state_cache = forward_meta.linear_attn_caches[:, self.layer_id, :, :, :]
            output, updated_state_cache = lightning_attention(q, k, v, self.tp_slope, kv_history=state_cache)
            forward_meta.linear_attn_caches[:, self.layer_id, :, :, :] = updated_state_cache
            # (1, H, N, D) -> (N, H*D)
            output = output.squeeze(0).transpose((1, 0, 2)).reshape((total_tokens, -1))
        
        else: # decode
            q = q.unsqueeze(2) # (B, H, 1, D)
            k = k.unsqueeze(2) # (B, H, 1, D)
            v = v.unsqueeze(2) # (B, H, 1, D)
            # ----------------------------------------------

            state_cache = forward_meta.linear_attn_caches[:, self.layer_id, :, :, :]
            slot_mapping = forward_meta.slot_mapping
            # output of triton kernel is (B, H*D)
            output = linear_decode_forward_triton(q, k, v, state_cache, self.tp_slope, slot_mapping)
        
        # (total_tokens, inner_hidden_size_tp)
        output = self.norm(output)
        
        gate = self.output_gate(hidden_states) # (total_tokens, inner_hidden_size_tp)
        output = F.sigmoid(gate) * output.cast(model_dtype)
        
        # final_output shape should be (total_tokens, hidden_size)
        final_output = self.out_proj(output)
        
        return final_output
    
    
    
class MiniMaxM1MLP(nn.Layer):
    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = "", reduce_results: bool = True):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear(fd_config, prefix=f"{prefix}.up_gate_proj", input_size=config.hidden_size, output_size=intermediate_size * 2, with_bias=False)
        self.down_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.down_proj", input_size=intermediate_size, output_size=config.hidden_size, with_bias=False, reduce_results=reduce_results)
        self.act_fn = SiluAndMul()
    def forward(self, x):
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out

class MiniMaxM1MoEBlock(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.gate = ReplicatedLinear(fd_config, prefix=f"{prefix}.gate", input_size=config.hidden_size, output_size=config.num_local_experts, with_bias=False, weight_dtype="float32")
        self.experts = FusedMoE(fd_config, moe_intermediate_size=config.intermediate_size, num_experts=config.num_local_experts, top_k=config.num_experts_per_tok, layer_idx=layer_id)
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return self.experts(hidden_states, self.gate)

# +++++++++++++++ 修改: DecoderLayer 支持两种 Attention +++++++++++++++
class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, original_layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.original_layer_id = original_layer_id
        
        self.attn_type = config.attn_type_list[original_layer_id]
        
        attn_prefix = f"{prefix}.self_attn"

        if self.attn_type == 1:  # GQA
            logger.info(f"Initializing DecoderLayer with prefix '{prefix}' (original layer {original_layer_id}) as GQA type.")
            # 【关键修改 1】: prefix 直接使用层的 prefix，而不是 attn_prefix
            self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
            self.o_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.o_proj", input_size=config.num_attention_heads * config.head_dim, output_size=config.hidden_size, with_bias=False)
            self.self_attn = Attention(fd_config, layer_id=original_layer_id, prefix=attn_prefix, use_neox_rotary_style=True)
        elif self.attn_type == 0: # 线性注意力
            logger.info(f"Initializing DecoderLayer with prefix '{prefix}' (original layer {original_layer_id}) as Linear Attention type.")
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id=original_layer_id, prefix=attn_prefix)
            self.qkv_proj = None 
            self.o_proj = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type} for layer {original_layer_id}")

        self.mlp = MiniMaxM1MoEBlock(fd_config, original_layer_id, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.post_attention_layernorm")
        
        self.shared_moe = config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(fd_config, config.shared_intermediate_size, prefix=f"{prefix}.shared_mlp", reduce_results=False)
            self.coefficient = ReplicatedLinear(fd_config, prefix=f"{prefix}.coefficient", input_size=config.hidden_size, output_size=1, with_bias=False, weight_dtype="float32")

    # forward 方法保持不变...
    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: Optional[paddle.Tensor]):
        if residual is None:
            residual = hidden_states
            normed_hidden_states = self.input_layernorm(hidden_states)
        else:
            normed_hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.attn_type == 1: # GQA forward path
            qkv_out = self.qkv_proj(normed_hidden_states)
            attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
            attn_output = self.o_proj(attn_output)
        else: # 线性注意力 forward path
            attn_output = self.self_attn(normed_hidden_states, forward_meta)
        
        normed_attn_output, residual = self.post_attention_layernorm(attn_output, residual)
        
        mlp_output = self.mlp(normed_attn_output)
        
        if self.shared_moe:
            shared_output = self.shared_mlp(normed_attn_output)
            coef_logits = self.coefficient(normed_attn_output.cast("float32")) 
            coef = F.sigmoid(coef_logits)
            mlp_output = mlp_output.cast(coef.dtype) * (1 - coef) + shared_output.cast(coef.dtype) * coef

        return mlp_output, residual

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.config = fd_config.model_config
        prefix = "model"
        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens"
        )

        # ----------------- 【核心修改】 -----------------
        # 使用 nn.LayerDict 来构建层，key 就是原始的层号
        layers_to_build = {}
        # 假设我们总是从 0 构建到 num_hidden_layers - 1
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{i}"
            # original_layer_id 就是 i
            layers_to_build[str(i)] = MiniMaxM1DecoderLayer(fd_config, original_layer_id=i, prefix=layer_prefix)
        
        self.layers = nn.LayerDict(layers_to_build)
        # ----------------------------------------------
        self.norm = RMSNorm(
            fd_config,
            self.config.hidden_size, # <-- 位置参数
            eps=self.config.rms_norm_eps,
            prefix=f"{prefix}.norm"
        )

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
        residual = None
        # 遍历 LayerDict 时，需要按 key 的数字顺序
        for i in range(len(self.layers)):
            layer = self.layers[str(i)]
            hidden_states, residual = layer(forward_meta=forward_meta, hidden_states=hidden_states, residual=residual)
        if residual is not None:
             hidden_states = hidden_states + residual
        out = self.norm(hidden_states)
        return out


class MiniMaxM1ForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        config = self.fd_config.model_config
        config.pretrained_config.prefix_name = "model"
        if hasattr(config, "num_local_experts") and not hasattr(config, "moe_num_experts"):
            config.moe_num_experts = config.num_local_experts
        if hasattr(config, "rotary_dim") and hasattr(config, "head_dim") and config.rotary_dim < config.head_dim:
            config.partial_rotary_factor = config.rotary_dim / config.head_dim
        if not hasattr(config, "first_k_dense_replace"):
            config.first_k_dense_replace = 0
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, embedding_dim=config.hidden_size, num_embeddings=config.vocab_size, prefix="lm_head")
    
    @classmethod
    def name(cls): return "MiniMaxM1ForCausalLM"
    
    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        return self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
    
    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs):
        logits = self.lm_head(hidden_states)
        # 将 logits 转换为 float32，以确保与采样算子的兼容性
        return logits.cast("float32")

    # +++++++++++++++ 修改: 最终版 load_weights +++++++++++++++
    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        logger.info("Initializing robust multi-GPU weight loader for MiniMax-M1...")
        from fastdeploy.model_executor.utils import (default_weight_loader,
                                                     process_weights_after_loading, get_tensor)

        params_dict = dict(self.named_parameters())
        sublayers_dict = dict(self.named_sublayers())
        
        # manual_moe_loader 保持不变
        def manual_moe_loader(param, loaded_weight, expert_id, shard_id):
            # ... (这部分代码是正确的，保持不变) ...
            config = self.fd_config.model_config
            tp_rank = self.fd_config.parallel_config.tensor_parallel_rank
            tp_size = self.fd_config.parallel_config.tensor_parallel_size
            is_torch_format = config.model_format == "torch"
            if not param._is_initialized(): param.set_value(paddle.zeros(param.shape, dtype=param.dtype))
            loaded_weight_tensor = get_tensor(loaded_weight)
            if is_torch_format and len(loaded_weight_tensor.shape) == 2:
                loaded_weight_tensor = loaded_weight_tensor.transpose([1, 0])
            if shard_id in ["gate", "up"]:
                output_size_per_shard = loaded_weight_tensor.shape[1] // tp_size
                start, end = tp_rank * output_size_per_shard, (tp_rank + 1) * output_size_per_shard
                loaded_weight_shard = loaded_weight_tensor[:, start:end]
                target_expert_slice = param[expert_id]
                intermediate_size_sharded = target_expert_slice.shape[1] // 2
                if shard_id == "gate": target_sub_slice = target_expert_slice[:, :intermediate_size_sharded]
                else: target_sub_slice = target_expert_slice[:, intermediate_size_sharded:]
                if target_sub_slice.shape != loaded_weight_shard.shape: raise ValueError(f"[MoE Shape Mismatch] gate/up exp {expert_id}: Param={target_sub_slice.shape}, Loaded={loaded_weight_shard.shape}")
                target_sub_slice.set_value(loaded_weight_shard)
            else: # down
                input_size_per_shard = loaded_weight_tensor.shape[0] // tp_size
                start, end = tp_rank * input_size_per_shard, (tp_rank + 1) * input_size_per_shard
                loaded_weight_shard = loaded_weight_tensor[start:end, :]
                target_expert_slice = param[expert_id]
                if target_expert_slice.shape != loaded_weight_shard.shape: raise ValueError(f"[MoE Shape Mismatch] down exp {expert_id}: Param={target_expert_slice.shape}, Loaded={loaded_weight_shard.shape}")
                target_expert_slice.set_value(loaded_weight_shard)

        # 记录已处理的权重，避免重复警告
        loaded_checkpoint_keys = set()
        
        for loaded_weight_name, loaded_weight in weights_iterator:
            
            # 检查层号是否在模型范围内，如果不在则跳过
            layer_match = re.search(r'\.layers\.(\d+)\.', loaded_weight_name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                if layer_idx >= self.fd_config.model_config.num_hidden_layers:
                    continue 

            param_name = loaded_weight_name
            loader_used = None
            log_prefix = ""

            # 规则 1: MoE 专家权重 (最高优先级)
            moe_match = re.search(r"(\.layers\.\d+\.)block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name)
            if moe_match:
                prefix, expert_id, weight_type = moe_match.groups()
                suffix = "mlp.experts.up_gate_proj_weight" if weight_type in ["w1", "w3"] else "mlp.experts.down_proj_weight"
                param_name = f"model{prefix}{suffix}"
                shard_id = {"w1": "gate", "w3": "up", "w2": "down"}[weight_type]
                if param_name in params_dict:
                    log_prefix = "[MoE Loader]"
                    logger.info(f"{log_prefix} Routing '{loaded_weight_name}' to '{param_name}' for expert {expert_id}, shard '{shard_id}'.")
                    manual_moe_loader(params_dict[param_name], loaded_weight, int(expert_id), shard_id)
                    loaded_checkpoint_keys.add(loaded_weight_name)
                continue

            # 规则 2: GQA 层权重 (qkv/o_proj)
            if 'self_attn' in loaded_weight_name and any(p in loaded_weight_name for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                layer_idx = int(re.search(r'\.layers\.(\d+)\.', loaded_weight_name).group(1))
                # 仅当该层是GQA类型时应用此规则
                if self.fd_config.model_config.attn_type_list[layer_idx] == 1:
                    if 'q_proj.weight' in loaded_weight_name:
                        param_name = loaded_weight_name.replace('self_attn.q_proj.weight', 'qkv_proj.weight')
                        shard_id = 'q'
                    elif 'k_proj.weight' in loaded_weight_name:
                        param_name = loaded_weight_name.replace('self_attn.k_proj.weight', 'qkv_proj.weight')
                        shard_id = 'k'
                    elif 'v_proj.weight' in loaded_weight_name:
                        param_name = loaded_weight_name.replace('self_attn.v_proj.weight', 'qkv_proj.weight')
                        shard_id = 'v'
                    elif 'o_proj.weight' in loaded_weight_name:
                        param_name = loaded_weight_name.replace('self_attn.o_proj.weight', 'o_proj.weight')
                        shard_id = None
                    
                    if param_name in params_dict:
                        log_prefix = "[GQA Loader]"
                        logger.info(f"{log_prefix} Routing '{loaded_weight_name}' to '{param_name}' with shard_id '{shard_id}'.")
                        param = params_dict[param_name]
                        loader = getattr(param, 'weight_loader', default_weight_loader(self.fd_config))
                        if shard_id:
                            loader(param, loaded_weight, shard_id)
                        else:
                            loader(param, loaded_weight)
                        loaded_checkpoint_keys.add(loaded_weight_name)
                    continue

            # 规则 3: 简单重命名
            simple_rename_map = {
                "block_sparse_moe.gate.weight": "mlp.gate.weight",
                "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
                "lm_head.weight": "lm_head.linear.weight",
            }
            renamed = False
            for old, new in simple_rename_map.items():
                if old in param_name:
                    param_name = param_name.replace(old, new)
                    log_prefix = "[Simple Rename]"
                    renamed = True
                    break
            
            # 规则 4: 直接匹配 (包括线性注意力层)
            if not renamed:
                log_prefix = "[Direct Match]"
            
            if param_name in params_dict:
                param = params_dict[param_name]
                loader = getattr(param, 'weight_loader', default_weight_loader(self.fd_config))
                logger.info(f"{log_prefix} Loading '{loaded_weight_name}' into '{param_name}'.")
                loader(param, loaded_weight)
                loaded_checkpoint_keys.add(loaded_weight_name)
            elif loaded_weight_name not in loaded_checkpoint_keys:
                logger.warning(f"Weight '{loaded_weight_name}' was not used (tried name '{param_name}').")

        # 检查是否有模型参数未被加载
        all_loaded_param_names = {name for name, _ in weights_iterator if name in loaded_checkpoint_keys}
        for param_name, param in params_dict.items():
            # 构造可能的源权重名称进行反向检查
            # 这是一个近似检查，但对于调试很有用
            is_loaded = False
            if any(key in param_name for key in loaded_checkpoint_keys): # 简化检查
                 is_loaded = True
            
            # 更精确的检查需要反向映射，这里暂时省略
            # 如果需要，可以添加一个 `param_name` 到 `loaded_weight_name` 的映射来检查
            
            # 简单的判断：如果参数的均值仍然接近于其初始化值，则可能未加载
            # 注意：这只是一个启发式方法
            if 'embeddings' in param_name: continue # embedding权重巨大，计算sum很慢
            
            # if not is_param_loaded_heuristic(param):
            #     logger.warning(f"Model parameter '{param_name}' might not have been loaded from checkpoint.")

        # 4. process_weights_after_loading
        process_weights_after_loading_fn = process_weights_after_loading(sublayers_dict)
        for name, param in params_dict.items():
             sublayer_name = name.rsplit('.', 1)[0]
             process_weights_after_loading_fn(sublayer_name, param)

        logger.info("Finished processing weights.")


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_state_dict(self, state_dict):
        raise NotImplementedError("MiniMax-M1 uses the `load_weights` method.")

class MiniMaxM1PretrainedModel(PretrainedModel):
    config_class = FDConfig
    @classmethod
    def arch_name(cls): return "MiniMaxM1ForCausalLM"
    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.info("Bypassing automatic tensor parallel mappings for MiniMax-M1.")
        return {}