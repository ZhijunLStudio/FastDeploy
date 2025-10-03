# fastdeploy/model_executor/models/minimax_m1.py (THE ULTIMATE FIX - FINAL & COMPLETE)

from __future__ import annotations
import math
import re
import inspect
from functools import partial
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
    ReplicatedLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm, ParallelRMSNorm # Import both
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.utils import default_weight_loader, get_tensor

try:
    from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import (
        lightning_attention, linear_decode_forward_triton)
except ImportError:
    logger.error("Could not import minimax_mamba_ops. Please ensure Triton kernels are compiled.")
    lightning_attention = None
    linear_decode_forward_triton = None

# --- Helper to handle inconsistent return values ---
def _get_output(output):
    """Safely extracts the tensor from a layer's output, which could be a tuple or a tensor."""
    if isinstance(output, tuple):
        return output[0]
    return output

# --- Submodules ---

class MiniMaxM1MLP(nn.Layer):
    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = "", reduce_results: bool = True):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear(fd_config, prefix=f"{prefix}.up_gate_proj", input_size=config.hidden_size, output_size=intermediate_size * 2, with_bias=False)
        self.down_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.down_proj", input_size=intermediate_size, output_size=config.hidden_size, with_bias=False, reduce_results=reduce_results)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = _get_output(self.up_gate_proj(x))
        x = self.act_fn(gate_up)
        return _get_output(self.down_proj(x))

class MiniMaxM1MoEBlock(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.gate = ReplicatedLinear(fd_config, prefix=f"{prefix}.gate", input_size=config.hidden_size, output_size=config.num_local_experts, with_bias=False, weight_dtype="float32")
        self.experts = FusedMoE(
            fd_config,
            moe_intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            layer_idx=layer_id,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        router_logits = _get_output(self.gate(hidden_states))
        return self.experts(hidden_states, router_logits)

class MiniMaxM1StandardAttention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
        self.o_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.o_proj", input_size=config.num_attention_heads * config.head_dim, output_size=config.hidden_size, with_bias=False)
        self.attn = Attention(fd_config, layer_id=layer_id, prefix=prefix, use_neox_rotary_style=True)

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = _get_output(self.qkv_proj(hidden_states))
        attn_output = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        return _get_output(self.o_proj(attn_output))

class MiniMaxM1LinearAttention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id
        self.hidden_inner_size = config.num_attention_heads * config.head_dim
        self.head_dim = config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        tp_rank = fd_config.parallel_config.tensor_parallel_rank
        self.num_heads_tp = config.num_attention_heads // tp_size
        self.qkv_proj = ColumnParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", input_size=config.hidden_size, output_size=self.hidden_inner_size * 3, with_bias=False)
        self.output_gate = ColumnParallelLinear(fd_config, prefix=f"{prefix}.output_gate", input_size=config.hidden_size, output_size=self.hidden_inner_size, with_bias=False)
        self.out_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.out_proj", input_size=self.hidden_inner_size, output_size=config.hidden_size, with_bias=False)
        
        # Use the new ParallelRMSNorm for sharded inputs
        self.norm = ParallelRMSNorm(fd_config, hidden_size=self.hidden_inner_size, eps=1e-5, prefix=f"{prefix}.norm")
        
        full_slope_rate = self._build_slope_tensor(config.num_attention_heads)
        if config.num_hidden_layers <= 1:
            full_slope_rate = full_slope_rate * (1 + 1e-5)
        else:
            full_slope_rate = full_slope_rate * (1 - layer_id / (config.num_hidden_layers - 1) + 1e-5)
        tp_slope = full_slope_rate[tp_rank * self.num_heads_tp:(tp_rank + 1) * self.num_heads_tp]
        self.register_buffer("slope_rate", tp_slope.contiguous())

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])
        slopes = paddle.to_tensor(get_slopes(n_attention_heads), dtype='float32').reshape([n_attention_heads, 1, 1])
        return slopes

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        if lightning_attention is None: raise ImportError("Triton kernels for minimax_mamba_ops not available.")
        
        qkv = _get_output(self.qkv_proj(hidden_states))
        output_gated = _get_output(self.output_gate(hidden_states))
        
        qkv = qkv.reshape([-1, self.num_heads_tp, 3 * self.head_dim])
        q, k, v = paddle.split(qkv, 3, axis=-1)

        if forward_meta.forward_mode.is_prefill() or forward_meta.forward_mode.is_mixed():
            num_tokens = q.shape[0]
            q = q.reshape([1, num_tokens, self.num_heads_tp, self.head_dim]).transpose([0, 2, 1, 3])
            k = k.reshape([1, num_tokens, self.num_heads_tp, self.head_dim]).transpose([0, 2, 1, 3])
            v = v.reshape([1, num_tokens, self.num_heads_tp, self.head_dim]).transpose([0, 2, 1, 3])
            kv_history_in = forward_meta.linear_attn_kv_history[self.layer_id]
            attn_output, kv_history_out = lightning_attention(q, k, v, self.slope_rate, kv_history=kv_history_in)
            forward_meta.linear_attn_kv_history[self.layer_id] = kv_history_out
            attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([-1, self.num_heads_tp * self.head_dim])
        elif forward_meta.forward_mode.is_decode():
            bsz = q.shape[0]
            q = q.reshape([bsz, self.num_heads_tp, 1, self.head_dim])
            k = k.reshape([bsz, self.num_heads_tp, 1, self.head_dim])
            v = v.reshape([bsz, self.num_heads_tp, 1, self.head_dim])
            kv_caches = forward_meta.linear_attn_kv_history[self.layer_id]
            slot_idx = paddle.arange(0, bsz, dtype='int32')
            attn_output = linear_decode_forward_triton(q, k, v, kv_caches, self.slope_rate.squeeze(), slot_idx, BLOCK_SIZE=32)
        else:
            raise ValueError(f"Unsupported forward mode for Linear Attention: {forward_meta.forward_mode}")

        # attn_output is sharded, so we use ParallelRMSNorm
        norm_output = self.norm(attn_output)
        
        # output_gated is also sharded, so this works element-wise
        gated_output = norm_output * F.silu(output_gated)
        
        # out_proj is RowParallelLinear, expects a sharded input
        final_output = _get_output(self.out_proj(gated_output))
        return final_output

class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id
        attn_prefix = f"{prefix}.self_attn"
        attn_type = config.attn_type_list[layer_id]
        if attn_type == 0:
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        elif attn_type == 1:
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id, prefix=attn_prefix)
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        else:
            raise ValueError(f"Unknown attention type {attn_type}")
        self.mlp = MiniMaxM1MoEBlock(fd_config, layer_id, prefix=f"{prefix}.mlp")
        self.shared_moe = config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(fd_config, config.shared_intermediate_size, prefix=f"{prefix}.shared_mlp", reduce_results=False)
            self.coefficient = ReplicatedLinear(fd_config, prefix=f"{prefix}.coefficient", input_size=config.hidden_size, output_size=1, with_bias=False, weight_dtype="float32")
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta
        self.input_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.post_attention_layernorm")

    def forward(self, *, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: paddle.Tensor | None):
        residual_main = hidden_states
        residual_attn = _get_output(self.input_layernorm(hidden_states))
        attn_output = self.self_attn(forward_meta=forward_meta, hidden_states=residual_attn)
        hidden_states = (residual_main * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)
        residual_mlp = _get_output(self.post_attention_layernorm(hidden_states))
        mlp_output = self.mlp(residual_mlp)
        if self.shared_moe:
            shared_mlp_output = self.shared_mlp(residual_mlp)
            coef = _get_output(self.coefficient(residual_mlp.cast("float32")))
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
        hidden_states = _get_output(self.norm(hidden_states))
        return hidden_states

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
    
    def forward(self, *args, **kwargs): 
        forward_meta = kwargs.get("forward_meta")
        # Check if any layer uses linear attention
        is_linear_attn_used = 0 in self.fd_config.model_config.attn_type_list
        if is_linear_attn_used and not hasattr(forward_meta, "linear_attn_kv_history"):
            self.init_linear_attn_kv_history(forward_meta)
        return self.model(**kwargs)
    
    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs): return self.lm_head(hidden_states)

    def init_linear_attn_kv_history(self, forward_meta: ForwardMeta):
        config = self.fd_config.model_config
        num_layers = config.num_hidden_layers
        bsz = self.fd_config.parallel_config.max_num_seqs
        H_tp = config.num_attention_heads // self.fd_config.parallel_config.tensor_parallel_size
        D = config.head_dim
        forward_meta.linear_attn_kv_history = [paddle.zeros((bsz, H_tp, D, D), dtype='float32') for _ in range(num_layers)]
        
    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        logger.info("Entering Final V-LAST Manual MoE Loader for MiniMax-M1...")
        all_param_mappings = self._build_weight_mappings()

        for loaded_weight_name, loaded_weight in weights_iterator:
            if loaded_weight_name in all_param_mappings:
                param_info = all_param_mappings[loaded_weight_name]
                param = param_info["param"]
                loader = param_info["loader"]
                kwargs = param_info.get("kwargs", {})
                
                try:
                    loader(param, loaded_weight, **kwargs)
                except Exception:
                    import traceback
                    logger.error(f"Error loading weight '{loaded_weight_name}' into '{param.name}':\n{traceback.format_exc()}")
            elif "rotary_emb.inv_freq" not in loaded_weight_name:
                # This warning is now more accurate because we built a full map.
                layer_match = re.search(r"model\.layers\.(\d+)\.", loaded_weight_name)
                if not (layer_match and int(layer_match.group(1)) >= self.fd_config.model_config.num_hidden_layers):
                    logger.warning(f"[LOAD_WEIGHTS] Weight '{loaded_weight_name}' has no mapping and was skipped.")

        logger.info("Weight loading finished.")

    def _build_weight_mappings(self):
        mappings = {}
        params_dict = dict(self.named_parameters())
        tp_rank = self.fd_config.parallel_config.tensor_parallel_rank
        tp_size = self.fd_config.parallel_config.tensor_parallel_size
        model_format = self.fd_config.model_config.model_format
        num_hidden_layers = self.fd_config.model_config.num_hidden_layers
        num_experts = self.fd_config.model_config.num_local_experts

        def manual_moe_loader(param, loaded_weight, expert_id, shard_id):
            if not param._is_initialized():
                param.set_value(paddle.zeros(param.shape, dtype=param.dtype))
            loaded_weight_tensor = get_tensor(loaded_weight)
            if model_format == "torch":
                loaded_weight_tensor = loaded_weight_tensor.transpose([1, 0])
            if shard_id in ["gate", "up"]:
                block_size = loaded_weight_tensor.shape[1] // tp_size
                start, end = tp_rank * block_size, (tp_rank + 1) * block_size
                loaded_weight_shard = loaded_weight_tensor[:, start:end]
                target_slice_full = param[expert_id]
                output_size_slice = target_slice_full.shape[1] // 2
                if shard_id == "gate": target_slice = target_slice_full[:, :output_size_slice]
                else: target_slice = target_slice_full[:, output_size_slice:]
                target_slice.set_value(loaded_weight_shard)
            else:
                block_size = loaded_weight_tensor.shape[0] // tp_size
                start, end = tp_rank * block_size, (tp_rank + 1) * block_size
                loaded_weight_shard = loaded_weight_tensor[start:end, :]
                target_slice = param[expert_id]
                assert target_slice.shape == loaded_weight_shard.shape, f"Shape mismatch for expert {expert_id} {shard_id}: {target_slice.shape} vs {loaded_weight_shard.shape}"
                target_slice.set_value(loaded_weight_shard)

        for name, param in params_dict.items():
            if name in [p_info["param"].name for p_info in mappings.values() if "param" in p_info]: continue

            loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            hf_name = name
            kwargs = {}
            
            moe_match = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(up_gate_proj_weight|down_proj_weight)", name)
            if moe_match:
                layer_id = int(moe_match.group(1))
                param_type = moe_match.group(2)
                for expert_id in range(num_experts):
                    if param_type == "up_gate_proj_weight":
                        hf_w1 = f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w1.weight"
                        mappings[hf_w1] = {"param": param, "loader": manual_moe_loader, "kwargs": {"expert_id": expert_id, "shard_id": "gate"}}
                        hf_w3 = f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w3.weight"
                        mappings[hf_w3] = {"param": param, "loader": manual_moe_loader, "kwargs": {"expert_id": expert_id, "shard_id": "up"}}
                    else:
                        hf_w2 = f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w2.weight"
                        mappings[hf_w2] = {"param": param, "loader": manual_moe_loader, "kwargs": {"expert_id": expert_id, "shard_id": None}}
                continue 

            if name.endswith("embed_tokens.embeddings.weight"): hf_name = "model.embed_tokens.weight"
            elif name.endswith("lm_head.linear.weight"): hf_name = "lm_head.weight"
            elif ".mlp.gate.weight" in name: hf_name = name.replace("mlp.gate", "block_sparse_moe.gate")
            
            mappings[hf_name] = {"param": param, "loader": loader, "kwargs": kwargs}
            
        return mappings

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