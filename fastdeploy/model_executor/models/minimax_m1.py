# fastdeploy/model_executor/models/minimax_m1.py (FINAL PRODUCTION VERSION - REVISED 5)

from __future__ import annotations
import math
import re
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
from fastdeploy.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.utils import default_weight_loader, process_weights_after_loading
# 你的 triton op 导入
from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import lightning_attention, linear_decode_forward_triton

class MiniMaxM1MLP(nn.Layer):
    def __init__( self, fd_config: FDConfig, intermediate_size: int, prefix: str = "", reduce_results: bool = True ):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear( fd_config, prefix=f"{prefix}.up_gate_proj", input_size=config.hidden_size, output_size=intermediate_size * 2, with_bias=False,)
        self.down_proj = RowParallelLinear( fd_config, prefix=f"{prefix}.down_proj", input_size=intermediate_size, output_size=config.hidden_size, with_bias=False, reduce_results=reduce_results,)
        self.act_fn = SiluAndMul()
    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_gate_proj(x)))

class MiniMaxM1MoEBlock(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.gate = ReplicatedLinear( fd_config=fd_config, prefix=f"{prefix}.gate", input_size=config.hidden_size, output_size=config.num_local_experts, with_bias=False, weight_dtype="float32",)
        weight_key_map = { "up_gate_proj_expert_weight_key": "experts.{{}}.up_gate_proj.weight", "down_proj_expert_weight_key": "experts.{{}}.down_proj.weight",}
        self.experts = FusedMoE( fd_config, moe_intermediate_size=config.intermediate_size, num_experts=config.num_local_experts, top_k=config.num_experts_per_tok, layer_idx=layer_id, weight_key_map=weight_key_map,)
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return self.experts(hidden_states, self.gate)

class MiniMaxM1StandardAttention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.fd_config = fd_config
        self.layer_id = layer_id
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=prefix, with_bias=False)
        self.o_proj = RowParallelLinear( fd_config, prefix=f"{prefix}.o_proj", input_size=fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim, output_size=fd_config.model_config.hidden_size, with_bias=False)
        self.attn = Attention( fd_config, layer_id=layer_id, prefix=prefix, use_neox_rotary_style=True)
    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = self.qkv_proj(hidden_states)
        attn_output = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        return self.o_proj(attn_output)

class MiniMaxM1LinearAttention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.fd_config = fd_config
        self.layer_id = layer_id
        config = fd_config.model_config
        hidden_inner_size = config.head_dim * config.num_attention_heads
        self.qkv_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", input_size=config.hidden_size, output_size=hidden_inner_size * 3, with_bias=False)
        self.output_gate = RowParallelLinear(fd_config, prefix=f"{prefix}.output_gate", input_size=config.hidden_size, output_size=hidden_inner_size, with_bias=False)
        self.out_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.out_proj", input_size=hidden_inner_size, output_size=config.hidden_size, with_bias=False)
        self.norm = RMSNorm(fd_config, hidden_size=hidden_inner_size, eps=1e-5, prefix=f"{prefix}.norm")
        slope_rate = self._build_slope_tensor(config.num_attention_heads)
        if config.num_hidden_layers > 1:
            slope_rate = slope_rate * (1 - layer_id / (config.num_hidden_layers - 1) + 1e-5)
        self.register_buffer("slope_rate", slope_rate, persistable=False)
    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]
            if math.log2(n).is_integer(): return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])
        return paddle.to_tensor(get_slopes(n_attention_heads), dtype='float32').reshape([n_attention_heads, 1, 1])
    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        num_tokens, _ = hidden_states.shape
        config = self.fd_config.model_config
        H = config.num_attention_heads
        D = config.head_dim

        qkv_out = self.qkv_proj(hidden_states)
        qkv_act = F.silu(qkv_out.cast("float32"))
        qkv_act_reshaped = qkv_act.reshape([num_tokens, H, 3 * D])
        q, k, v = paddle.split(qkv_act_reshaped, 3, axis=-1)

        if forward_meta.forward_mode.is_prefill():
            B = len(forward_meta.seq_lens_encoder)
            if num_tokens == 0: # Handle empty input case
                # 需要为 decode 阶段准备一个正确形状的空 cache
                # 注意：这里的 D, D 对应 vLLM 的 (d_inner, d_inner)
                updated_kv_history = paddle.zeros([B, H, D, D], dtype='float32')
                attn_hidden = paddle.empty([0, H * D], dtype=hidden_states.dtype)
            else:
                if num_tokens % B != 0:
                    raise ValueError(f"In prefill, num_tokens({num_tokens}) must be divisible by batch_size({B}).")
                N = num_tokens // B
                q = q.reshape([B, N, H, D]).transpose([0, 2, 1, 3])
                k = k.reshape([B, N, H, D]).transpose([0, 2, 1, 3])
                v = v.reshape([B, N, H, D]).transpose([0, 2, 1, 3])
                
                # FastDeploy 中 caches 是 list of tensors
                kv_history = forward_meta.caches[self.layer_id] 
                attn_hidden, updated_kv_history = lightning_attention(q, k, v, self.slope_rate, kv_history=kv_history)
                attn_hidden = attn_hidden.transpose([0, 2, 1, 3]).reshape([num_tokens, H * D])
            
            forward_meta.caches[self.layer_id] = updated_kv_history

        elif forward_meta.forward_mode.is_decode():
            B = num_tokens
            q = q.reshape([B, H, 1, D])
            k = k.reshape([B, H, 1, D])
            v = v.reshape([B, H, 1, D])
            
            kv_caches = forward_meta.caches[self.layer_id]
            slot_mapping = forward_meta.slot_mapping.flatten()
            attn_hidden = linear_decode_forward_triton(q, k, v, kv_caches, self.slope_rate.squeeze(), slot_mapping, BLOCK_SIZE=32)
        else:
            raise NotImplementedError("Mixed mode is not supported for Linear Attention yet.")
            
        hidden_norm = self.norm(attn_hidden)
        gate, _ = self.output_gate(hidden_states)
        gated_hidden = F.sigmoid(gate) * hidden_norm
        output = self.out_proj(gated_hidden.cast(hidden_states.dtype))
        return output

class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id
        attn_type = config.attn_type_list[layer_id]
        attn_prefix = f"{prefix}.self_attn"
        if attn_type == 1:
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        elif attn_type == 0:
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        else:
            raise ValueError(f"Unknown attention type for layer {layer_id}: {attn_type}")
        self.mlp = MiniMaxM1MoEBlock(fd_config, layer_id, prefix=f"{prefix}.mlp") 
        self.shared_moe = config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP( fd_config, config.shared_intermediate_size, prefix=f"{prefix}.shared_mlp", reduce_results=False)
            self.coefficient = ReplicatedLinear( fd_config, prefix=f"{prefix}.coefficient", input_size=config.hidden_size, output_size=1, with_bias=False, weight_dtype="float32")
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta
        self.input_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.post_attention_layernorm")
    def forward(self, *, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: paddle.Tensor | None):
        residual_attn = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(forward_meta=forward_meta, hidden_states=residual_attn)
        hidden_states = (hidden_states * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)
        residual_mlp = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(residual_mlp)
        if self.shared_moe:
            shared_mlp_output = self.shared_mlp(residual_mlp)
            coef, _ = self.coefficient(residual_mlp.cast("float32"))
            mlp_output = (mlp_output.cast("float32") * (1 - F.sigmoid(coef)) + shared_mlp_output.cast("float32") * F.sigmoid(coef)).cast(hidden_states.dtype)
        hidden_states = (hidden_states * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        return hidden_states, None

@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.config = fd_config.model_config
        prefix = "model"
        self.embed_tokens = VocabParallelEmbedding(fd_config, num_embeddings=self.config.vocab_size, embedding_dim=self.config.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.LayerList([MiniMaxM1DecoderLayer(fd_config, i, prefix=f"{prefix}.layers.{i}") for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(fd_config, hidden_size=self.config.hidden_size, eps=self.config.rms_norm_eps, prefix=f"{prefix}.norm")
    def forward(self, *args, **kwargs):
        ids_remove_padding = kwargs.get("ids_remove_padding")
        forward_meta = kwargs.get("forward_meta")
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(forward_meta=forward_meta, hidden_states=hidden_states, residual=residual)
        return self.norm(hidden_states)

class MiniMaxM1ForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        config = self.fd_config.model_config
        config.pretrained_config.prefix_name = "model"
        if hasattr(config, "num_local_experts") and not hasattr(config, "moe_num_experts"):
            config.moe_num_experts = config.num_local_experts
            config.n_routed_experts = config.num_local_experts
        if not hasattr(config, "n_shared_experts"):
             config.n_shared_experts = 0
        if not hasattr(config, "first_k_dense_replace"):
             config.first_k_dense_replace = 0
        if hasattr(config, "rotary_dim") and hasattr(config, "head_dim") and config.rotary_dim < config.head_dim:
            config.partial_rotary_factor = config.rotary_dim / config.head_dim
        
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, embedding_dim=config.hidden_size, num_embeddings=config.vocab_size, prefix="lm_head")

    @classmethod
    def name(cls): return "MiniMaxM1ForCausalLM"
    def forward(self, *args, **kwargs): return self.model(**kwargs)
    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs): return self.lm_head(hidden_states)

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        logger.info("Entering the final, robust load_weights for MiniMaxM1...")
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()))
        
        for loaded_weight_name, loaded_weight in weights_iterator:
            found = False
            
            # --- 1. 精确匹配 MoE 专家权重 ---
            match = re.search(r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name)
            if match:
                layer_id, expert_id_str, weight_type = match.groups()
                layer_id, expert_id = int(layer_id), int(expert_id_str)

                # 如果你的模型只有10层，跳过所有大于9的层的权重
                if layer_id >= self.fd_config.model_config.num_hidden_layers:
                    continue

                if weight_type in ['w1', 'w3']:
                    fd_param_name = f"model.layers.{layer_id}.mlp.experts.up_gate_proj.weight"
                    shard_id = "gate" if weight_type == 'w1' else "up"
                else: # w2
                    fd_param_name = f"model.layers.{layer_id}.mlp.experts.down_proj.weight"
                    shard_id = None

                if fd_param_name in params_dict:
                    param = params_dict[fd_param_name]
                    loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    found = True
                else:
                    # 这个错误日志现在非常有用，如果还有问题，会告诉我们哪里不对
                    logger.error(f"[LOAD_WEIGHTS_ERROR] MoE weight '{loaded_weight_name}' found but target param '{fd_param_name}' does not exist!")

            if found: continue

            # --- 2. 映射其他权重 ---
            name_map = {
                "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
                "lm_head.weight": "lm_head.linear.weight",
            }
            
            current_param_name = loaded_weight_name
            # 尝试完整匹配
            if current_param_name in name_map:
                fd_param_name = name_map[current_param_name]
                if fd_param_name in params_dict:
                    param = params_dict[fd_param_name]
                    loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    loader(param, loaded_weight)
                    found = True
            
            # 尝试后缀匹配 (适用于带 layer.{id} 的权重)
            # 例如: model.layers.7.self_attn.q_proj.weight -> model.layers.7.self_attn.qkv_proj.weight
            if not found:
                # 提取 layer_id
                layer_match = re.search(r"model\.layers\.(\d+)\.", current_param_name)
                if layer_match:
                    layer_id = int(layer_match.group(1))
                    if layer_id < self.fd_config.model_config.num_hidden_layers:
                        # 构造可能的FD参数名并检查
                        name_suffixes = {
                            ".self_attn.q_proj.weight": (".self_attn.qkv_proj.weight", "q"),
                            ".self_attn.k_proj.weight": (".self_attn.qkv_proj.weight", "k"),
                            ".self_attn.v_proj.weight": (".self_attn.qkv_proj.weight", "v"),
                            ".self_attn.o_proj.weight": (".self_attn.o_proj.weight", None),
                            ".mlp.gate.weight": (".mlp.gate.weight", None), # 之前叫 block_sparse_moe
                            # Linear Attention
                            ".self_attn.qkv_proj.weight": (".self_attn.qkv_proj.weight", None), 
                            ".self_attn.output_gate.weight": (".self_attn.output_gate.weight", None), 
                        }
                        for ckpt_suffix, (fd_suffix, shard_id) in name_suffixes.items():
                            if current_param_name.endswith(ckpt_suffix):
                                base_name = current_param_name[:-len(ckpt_suffix)]
                                fd_param_name = base_name + fd_suffix
                                if fd_param_name in params_dict:
                                    param = params_dict[fd_param_name]
                                    loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                                    loader(param, loaded_weight, shard_id)
                                    found = True
                                    break
            if found: continue
            
            # 尝试直接匹配 (适用于 layernorm 等)
            if not found and current_param_name in params_dict:
                param = params_dict[current_param_name]
                loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                loader(param, loaded_weight)
                found = True
            
            if not found and "rotary_emb.inv_freq" not in loaded_weight_name:
                logger.warning(f"[LOAD_WEIGHTS] Weight '{loaded_weight_name}' was NOT FOUND and NOT MAPPED.")
            
            if found:
                process_weights_after_loading_fn(re.sub(r"\.(weight|bias)$", "", param.name), param)

    def set_state_dict(self, state_dict):
        raise NotImplementedError("MiniMax-M1 uses the `load_weights` method.")

class MiniMaxM1PretrainedModel(PretrainedModel):
    config_class = FDConfig
    @classmethod
    def arch_name(cls): return "MiniMaxM1ForCausalLM"
    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.warning("Tensor parallel mappings for MiniMax-M1 are placeholders.")
        return {}