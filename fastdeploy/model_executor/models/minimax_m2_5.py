"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations

import os
import re
from typing import Dict

import numpy as np

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import QKRMSNorm, RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    process_weights_after_loading,
)


def _numpy_int4_quant_and_pack(wt_np):
    """
    Quantize a float32 numpy array to int4 and pack as int8.

    Args:
        wt_np: 2D float32 numpy array [out_dim, in_dim] (or 3D [E, K, N] for MoE).

    Returns:
        packed: int8 numpy array, shape [out_dim, in_dim//8] (2D) or [E, K//8, N] (3D).
        scale:  float16 numpy array, shape [out_dim] (2D) or [E, 1, N] (3D).
    """
    max_bound = 7  # int4 range: [-7, 7]
    is_3d = wt_np.ndim == 3

    if is_3d:
        E, K, N = wt_np.shape
        assert K % 8 == 0, f"K ({K}) must be divisible by 8 for int4 packing"
        # Per-output-channel quantization (along K axis)
        ch_max = np.abs(wt_np).max(axis=1, keepdims=True)  # [E, 1, N]
        scale = ch_max / max_bound  # [E, 1, N]
        # Avoid division by zero
        scale = np.where(scale == 0, 1.0, scale)
        quanted = np.round(wt_np / scale).astype(np.int32)
        quanted = np.clip(quanted, -7, 7) + 8  # shift to [0, 15]
        # Pack 8 int4 values into 1 int32 along K
        quanted = quanted.reshape([E, K // 8, 8, N])
        packed = np.zeros([E, K // 8, N], dtype=np.int32)
        for j in range(8):
            packed |= quanted[:, :, j, :] << (j * 4)
        # Pack int32 → int8 (4 int8 per int32)
        packed = packed.view(np.int8).reshape([E, K // 8, N * 4])
        scale_out = scale.astype(np.float16).squeeze(axis=1)  # [E, N]
    else:
        out_dim, in_dim = wt_np.shape
        assert in_dim % 8 == 0, f"in_dim ({in_dim}) must be divisible by 8 for int4 packing"
        ch_max = np.abs(wt_np).max(axis=0)  # [in_dim]
        scale = ch_max / max_bound  # [in_dim]
        scale = np.where(scale == 0, 1.0, scale)
        quanted = np.round(wt_np / scale[np.newaxis, :]).astype(np.int32)
        quanted = np.clip(quanted, -7, 7) + 8
        # Pack 8 int4 values into 1 int32 along in_dim
        quanted = quanted.reshape([out_dim, in_dim // 8, 8])
        packed = np.zeros([out_dim, in_dim // 8], dtype=np.int32)
        for j in range(8):
            packed |= quanted[:, :, j] << (j * 4)
        # Pack int32 → int8 (4 int8 per int32)
        packed = packed.view(np.int8).reshape([out_dim, (in_dim // 8) * 4])
        scale_out = scale.astype(np.float16)  # [in_dim]

    return packed, scale_out


class MiniMaxRMSNorm(paddle.nn.Layer):
    """
    MiniMax-M2.5 QK-RMSNorm: per-token normalization across the FULL Q or K vector.

    This matches vLLM's MiniMaxText01RMSNormTP behavior:
      variance = mean(x^2, axis=-1)  # across all features (all heads concatenated)
      x_normed = x * rsqrt(variance + eps) * weight
    """

    def __init__(self, hidden_size: int, tp_size: int, eps: float = 1e-6,
                 weight_key: str = ""):
        super().__init__()
        self.tp_size    = tp_size
        self.shard_size = hidden_size // tp_size
        self.eps        = eps
        self.weight_key = weight_key
        self.weight = self.create_parameter(
            shape=[self.shard_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )

    def weight_loader(self, param, loaded_weight):
        """Load the TP shard of this weight."""
        from fastdeploy.model_executor.layers.utils import get_tensor
        w = get_tensor(loaded_weight).cast("float32")
        shard_size = w.shape[0] // self.tp_size
        tp_rank    = paddle.distributed.get_rank() if self.tp_size > 1 else 0
        shard      = w[tp_rank * shard_size:(tp_rank + 1) * shard_size]
        param.set_value(shard)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """x: [..., shard_size]"""
        orig_dtype = x.dtype
        x = x.cast("float32")
        if self.tp_size > 1:
            # All-reduce variance across TP ranks
            var_local = x.pow(2).mean(axis=-1, keepdim=True)
            from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce
            var = tensor_model_parallel_all_reduce(var_local) / self.tp_size
        else:
            var = x.pow(2).mean(axis=-1, keepdim=True)
        x = x * paddle.rsqrt(var + self.eps) * self.weight
        return x.cast(orig_dtype)


class MiniMaxM2_5Attention(nn.Layer):
    """
    MiniMax-M2.5 Attention with GQA and QK-Norm.

    Architecture:
    - GQA: 48 query heads, 8 KV heads, head_dim=128
    - Partial RoPE: rotary_dim=64 (partial_rotary_factor=0.5)
    - QK Norm: per-token full-vector RMSNorm on Q and K (MiniMaxText01RMSNormTP style)
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        self.fd_config = fd_config
        self.head_dim = fd_config.model_config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        num_kv_heads_replicas = max(1, tp_size // fd_config.model_config.num_key_value_heads)
        self.q_size = fd_config.model_config.num_attention_heads * self.head_dim // tp_size
        self.kv_size = fd_config.model_config.num_key_value_heads * self.head_dim * num_kv_heads_replicas // tp_size

        # QKV projection
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)

        # Output projection
        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim * fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
            layer_id=layer_id,
        )

        # Attention backend (handles RoPE internally)
        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
        )

        # QK Norm: per-token full-vector RMSNorm, matching MiniMaxText01RMSNormTP
        # Total Q features = num_attention_heads * head_dim (before TP split)
        total_q = fd_config.model_config.num_attention_heads * self.head_dim
        total_k = fd_config.model_config.num_key_value_heads * self.head_dim
        self.q_norm = MiniMaxRMSNorm(
            hidden_size=total_q,
            tp_size=tp_size,
            eps=fd_config.model_config.rms_norm_eps,
            weight_key=f"{prefix}.q_norm.weight",
        )
        self.k_norm = MiniMaxRMSNorm(
            hidden_size=total_k,
            tp_size=tp_size if fd_config.model_config.num_key_value_heads >= tp_size else 1,
            eps=fd_config.model_config.rms_norm_eps,
            weight_key=f"{prefix}.k_norm.weight",
        )

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        # q_norm and k_norm loaded via load_weights in MiniMaxM2ForCausalLM
        self.attn.load_state_dict(state_dict)

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        qkv_out = self.qkv_proj(hidden_states)
        # Split QKV and apply per-token QK norm
        q = qkv_out[:, :self.q_size]
        k = qkv_out[:, self.q_size:self.q_size + self.kv_size]
        v = qkv_out[:, self.q_size + self.kv_size:]
        q = self.q_norm(q)
        k = self.k_norm(k)
        qkv_normed = paddle.concat([q, k, v], axis=-1)
        attn_out = self.attn(qkv=qkv_normed, forward_meta=forward_meta)
        output = self.o_proj(attn_out)
        return output


class MiniMaxM2_5MoE(nn.Layer):
    """
    MiniMax-M2.5 MoE Block.

    All 62 layers use MoE (no dense FFN layers).
    256 experts, top-8 routing, sigmoid scoring.
    Has e_score_correction_bias for routing bias correction.
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        num_experts = fd_config.model_config.num_local_experts

        # Weight key map: MiniMax uses w1/w2/w3 naming in checkpoint + correction bias
        weight_key_map = {
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            "gate_correction_bias_key": f"{prefix}.gate.e_score_correction_bias",
        }

        # Gate projects hidden_size -> num_experts (float32, no quant)
        # Must create gate BEFORE FusedMoE so e_score_correction_bias is available
        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=num_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )

        # MiniMax has e_score_correction_bias for routing bias correction
        # (used by noaux_tc routing: topk by score+bias, weight by score)
        if getattr(fd_config.model_config, "use_routing_bias", True):
            self.gate.e_score_correction_bias = self.create_parameter(
                shape=[1, num_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
        else:
            self.gate.e_score_correction_bias = None

        # Create FusedMoE with the correction bias reference (only once)
        self.experts = FusedMoE(
            fd_config,
            moe_intermediate_size=fd_config.model_config.intermediate_size,
            num_experts=num_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method="noaux_tc",   # MiniMax uses sigmoid routing with correction bias
            n_group=1,                # No grouping in MiniMax (unlike DeepSeek)
            topk_group=1,
            routed_scaling_factor=1.0,
            layer_idx=layer_id,
            gate_correction_bias=self.gate.e_score_correction_bias,
            weight_key_map=weight_key_map,
        )

    def forward(self, x, forward_meta):
        return self.experts(x, self.gate, forward_meta)

    def load_state_dict(self, state_dict):
        self.gate.load_state_dict(state_dict)
        self.experts.load_state_dict(state_dict)


class MiniMaxM2_5DecoderLayer(nn.Layer):
    """
    MiniMax-M2.5 Decoder Layer.

    All layers are MoE layers (no dense FFN).
    """

    def __init__(self, fd_config: FDConfig, prefix: str = "") -> None:
        super().__init__()

        layer_id = int(prefix.split(".")[-1])

        self.self_attn = MiniMaxM2_5Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = MiniMaxM2_5MoE(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            layer_id=layer_id,
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        hidden_states = self.self_attn(
            forward_meta=forward_meta,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states, forward_meta)

        return hidden_states, residual


@support_graph_optimization
class MiniMaxM2_5Model(nn.Layer):
    """
    MiniMax-M2.5 Transformer Model (62 decoder layers, all MoE).
    """

    def __init__(self, fd_config: FDConfig = None):
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        # Use "model" as the prefix (matches HF checkpoint structure)
        fd_config.model_config.pretrained_config.prefix_name = "model"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                MiniMaxM2_5DecoderLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Loading layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.embed_tokens(
            ids_remove_padding=ids_remove_padding, forward_meta=forward_meta
        )

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        if self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out


@ModelRegistry.register_model_class(
    architecture="MiniMaxM2ForCausalLM",
    module_name="minimax_m2_5",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniMaxM2ForCausalLM(ModelForCasualLM):
    """
    MiniMax-M2.5 Causal Language Model.

    Registered as "MiniMaxM2ForCausalLM" to match the architecture name
    in the model's config.json.
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        # Ensure num_local_experts is accessible as num_experts for FusedMoE
        if not hasattr(fd_config.model_config, "num_experts") or fd_config.model_config.num_experts is None:
            fd_config.model_config.num_experts = fd_config.model_config.num_local_experts

        # Set partial_rotary_factor from rotary_dim / head_dim (MiniMax: 64/128 = 0.5)
        if hasattr(fd_config.model_config, "rotary_dim") and fd_config.model_config.partial_rotary_factor == 1.0:
            fd_config.model_config.partial_rotary_factor = (
                fd_config.model_config.rotary_dim / fd_config.model_config.head_dim
            )

        # moe_intermediate_size field: FD uses this name, but MiniMax config has intermediate_size
        # for expert FFN (different from a dense model's intermediate_size).
        # They are the same here: expert intermediate_size=1536.
        # FusedMoE reads fd_config.model_config.moe_intermediate_size if set,
        # but we pass it explicitly, so this is just informational.

        self.model = MiniMaxM2_5Model(fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(cls):
        return "MiniMaxM2ForCausalLM"

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """
        Map MiniMax checkpoint weight names to FD internal names.

        MiniMax checkpoint uses:
          model.layers.{i}.mlp.experts.{j}.w1.weight  -> gate_proj (shard_id="gate")
          model.layers.{i}.mlp.experts.{j}.w2.weight  -> down_proj (shard_id="down")
          model.layers.{i}.mlp.experts.{j}.w3.weight  -> up_proj   (shard_id="up")
        """
        return FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.num_local_experts,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )

    def _wint4_quantize_linear_layers(self):
        """Quantize Linear layers (q/k/v/o_proj) from BF16 to WINT4 in-place."""
        from paddle.nn.quant import weight_quantize as _wq
        from fastdeploy.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
        from fastdeploy.model_executor.layers.quantization.weight_only import GPUWeightOnlyLinearMethod, WINT4Config

        logger.info("WINT4: Quantizing Linear layers to int4 ...")
        count = 0
        for name, sublayer in self.named_sublayers():
            if not isinstance(sublayer, (ColumnParallelLinear, RowParallelLinear)):
                continue
            if getattr(sublayer, '_wint4_quantized', False):
                continue
            if not hasattr(sublayer, 'weight') or sublayer.weight is None:
                continue
            if sublayer.weight.dtype != paddle.bfloat16:
                continue
            if "mlp." in name and "experts" not in name:
                continue
            if "lm_head" in name:
                continue

            w = sublayer.weight
            if self.fd_config.model_config.model_format == "torch":
                w = w.transpose([1, 0])
            wt_int4, wt_scale = _wq(w, algo="weight_only_int4")

            sublayer.weight = sublayer.create_parameter(
                shape=wt_int4.shape, dtype="int8",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            sublayer.weight.copy_(wt_int4, False)
            sublayer.weight_scale = sublayer.create_parameter(
                shape=wt_scale.shape, dtype=wt_scale.dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            sublayer.weight_scale.copy_(wt_scale, False)
            sublayer._wint4_quantized = True
            sublayer._torch_weight_transposed = True

            wint4_cfg = WINT4Config(is_checkpoint_bf16=True)
            sublayer.quant_method = GPUWeightOnlyLinearMethod(wint4_cfg)
            def _wint4_apply(layer, x):
                from paddle.nn.quant import weight_only_linear
                return weight_only_linear(
                    x.cast("bfloat16"), weight=layer.weight,
                    bias=layer.bias if layer.with_bias else None,
                    weight_scale=layer.weight_scale, weight_dtype="int4",
                )
            sublayer.quant_method.apply = _wint4_apply
            count += 1
            del wt_int4, wt_scale, w

        logger.info(f"WINT4: Quantized {count} Linear layers to int4")

    def _wint4_quantize_moe_experts(self):
        """Quantize MoE expert weights from BF16 to WINT4, one layer at a time.

        Key optimization: delete BF16 param + empty_cache BEFORE creating int8 param
        to avoid double GPU memory allocation.
        """
        from fastdeploy.model_executor.layers.moe.moe import FusedMoE
        from paddle.nn.quant import weight_quantize as _wq

        logger.info("WINT4: Quantizing MoE expert weights to int4 ...")
        for name, sublayer in self.named_sublayers():
            if not isinstance(sublayer, FusedMoE):
                continue
            if not hasattr(sublayer, 'up_gate_proj_weight'):
                continue
            if getattr(sublayer, '_wint4_quantized', False):
                continue

            E = sublayer.num_local_experts
            for wname, sname in [
                ("up_gate_proj_weight", "up_gate_proj_weight_scale"),
                ("down_proj_weight", "down_proj_weight_scale"),
            ]:
                orig = getattr(sublayer, wname)  # [E, K, N] BF16

                # Quantize all experts to CPU int8 + scale first
                packed_list = []
                scale_list = []
                for e in range(E):
                    wi, ws = _wq(orig[e].cuda(), algo="weight_only_int4")
                    packed_list.append(wi.cpu())
                    scale_list.append(ws.cpu())
                    del wi, ws

                # Use actual output shape from weight_quantize (not assumed)
                packed_shape = list(packed_list[0].shape)  # e.g. [K//2, N] or [N, K//2]
                scale_shape = list(scale_list[0].shape)     # e.g. [N]

                # Free BF16 param BEFORE allocating new int8 param
                if wname in sublayer._parameters:
                    sublayer._parameters.pop(wname)
                if hasattr(sublayer, wname):
                    delattr(sublayer, wname)
                if sname in sublayer._parameters:
                    sublayer._parameters.pop(sname)
                if hasattr(sublayer, sname):
                    delattr(sublayer, sname)
                del orig
                paddle.device.cuda.empty_cache()

                # Now create int8 param with correct shape
                new_w = sublayer.create_parameter(
                    shape=[E] + packed_shape, dtype="int8",
                    default_initializer=paddle.nn.initializer.Constant(0),
                )
                new_s = sublayer.create_parameter(
                    shape=[E] + scale_shape, dtype="bfloat16",
                    default_initializer=paddle.nn.initializer.Constant(0),
                )
                for e in range(E):
                    new_w[e].set_value(packed_list[e])
                    new_s[e].set_value(scale_list[e])

                setattr(sublayer, wname, new_w)
                setattr(sublayer, sname, new_s)
                paddle.device.cuda.empty_cache()

            sublayer._wint4_quantized = True
            # Replace quant_method so moe_expert_ffn uses int4 kernel
            from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import CutlassWeightOnlyMoEMethod
            from fastdeploy.model_executor.layers.quantization.weight_only import WINT4Config
            wint4_cfg = WINT4Config(is_checkpoint_bf16=True)
            sublayer.quant_method = CutlassWeightOnlyMoEMethod(wint4_cfg)
            mem = paddle.device.cuda.memory_allocated() / (1024**3)
            logger.info(f"WINT4: MoE layer '{name}' quantized to int4, GPU: {mem:.1f} GB")

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from weights_iterator.

        Handles:
        - q_proj/k_proj/v_proj  -> qkv_proj  (stacked)
        - gate_proj(w1)/up_proj(w3) -> up_gate_proj  (expert, stacked)
        - down_proj(w2)             -> down_proj      (expert)
        - embed_tokens / lm_head   -> direct load
        - q_norm / k_norm          -> qk_norm weights
        - MTP layers (model.layers.62+) -> skipped
        """
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            # MiniMax e_score_correction_bias: checkpoint has ".gate.e_score_correction_bias"
            # (after block_sparse_moe→mlp rename), FD FusedMoE has "experts.gate_correction_bias"
            ("experts.gate_correction_bias", "gate.e_score_correction_bias", None),
        ]

        expert_params_mapping = self.get_expert_mapping()
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(
            dict(self.named_sublayers()), self.fd_config
        )

        num_main_layers = self.fd_config.model_config.num_hidden_layers  # 62

        # Accumulate FP8 expert weights and their scales for deferred dequantization
        fp8_weights = {}   # name → fp8 tensor
        fp8_scales  = {}   # name (weight_scale_inv) → scale tensor

        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")

            # Skip MTP layers (model.layers.62, 63, 64, ...)
            # MTP layer indices start at num_hidden_layers
            skip = False
            if "model.layers." in loaded_weight_name:
                # Extract layer index
                parts = loaded_weight_name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                    if layer_idx >= num_main_layers:
                        skip = True
                except (ValueError, IndexError):
                    pass
            if skip:
                continue

            # MiniMax checkpoint uses "block_sparse_moe" but FD model uses "mlp"
            loaded_weight_name = loaded_weight_name.replace(
                ".block_sparse_moe.", ".mlp."
            )

            # MiniMax checkpoint has e_score_correction_bias at ".mlp.e_score_correction_bias"
            # but FD FusedMoE stores it as ".mlp.gate.e_score_correction_bias"
            # (since gate_correction_bias_key = "{prefix}.gate.e_score_correction_bias")
            loaded_weight_name = loaded_weight_name.replace(
                ".mlp.e_score_correction_bias",
                ".mlp.gate.e_score_correction_bias",
            )

            # MiniMax FP8 checkpoint stores per-block scales as "*.weight_scale_inv".
            # Collect scale for later dequantization.
            # NOTE: In TP>1, the scale shape may not match the parameter shape
            # (parameter is TP-sliced, scale is full). We only collect it for
            # dequant, not directly set it.
            if ".weight_scale_inv" in loaded_weight_name:
                fp8_scales[loaded_weight_name] = loaded_weight
                continue

            # For ALL FP8 weights (both expert and linear), defer loading for dequantization.
            if (loaded_weight_name.endswith(".weight") and
                    hasattr(loaded_weight, "dtype") and
                    "float8" in str(loaded_weight.dtype).lower()):
                fp8_weights[loaded_weight_name] = loaded_weight
                continue

            # Special handling for q_norm / k_norm weights.
            # Now using MiniMaxRMSNorm with shard-wise weight [total_size/tp].
            # Use the norm's weight_loader to properly TP-shard the weight.
            if ".q_norm.weight" in loaded_weight_name:
                # e.g. "model.layers.0.self_attn.q_norm.weight"
                # prefix = "model.layers.0.self_attn"
                prefix = loaded_weight_name.replace(".q_norm.weight", "")
                q_norm_key = f"{prefix}.q_norm.weight"   # = "model.layers.0.self_attn.q_norm.weight"
                if q_norm_key in params_dict:
                    param = params_dict[q_norm_key]
                    wl = getattr(param, "weight_loader", None)
                    if wl is not None:
                        wl(param, loaded_weight)
                    else:
                        w = get_tensor(loaded_weight).cast("float32")
                        tp = self.fd_config.parallel_config.tensor_parallel_size
                        shard = w.shape[0] // tp
                        rank  = paddle.distributed.get_rank() if tp > 1 else 0
                        param.set_value(w[rank*shard:(rank+1)*shard])
                continue
            if ".k_norm.weight" in loaded_weight_name:
                prefix = loaded_weight_name.replace(".k_norm.weight", "")
                k_norm_key = f"{prefix}.k_norm.weight"
                if k_norm_key in params_dict:
                    param = params_dict[k_norm_key]
                    wl = getattr(param, "weight_loader", None)
                    if wl is not None:
                        wl(param, loaded_weight)
                    else:
                        w = get_tensor(loaded_weight).cast("float32")
                        if w.shape[0] == param.shape[0]:
                            param.set_value(w)
                        else:
                            tp = self.fd_config.parallel_config.tensor_parallel_size
                            shard = max(1, w.shape[0] // tp)
                            rank  = paddle.distributed.get_rank() if tp > 1 else 0
                            end   = min((rank+1)*shard, w.shape[0])
                            param.set_value(w[rank*shard:end])
                continue

            # Try stacked parameter mappings
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                # Expert weights are handled separately
                if "mlp.experts" in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                matched = True
                break

            if matched:
                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                continue

            # Try expert parameter mappings
            matched_expert = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                matched_expert = True
                break

            if matched_expert:
                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                continue

            # Direct load for remaining parameters
            model_param_name = loaded_weight_name
            if model_param_name not in params_dict:
                continue
            param = params_dict[model_param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(
                r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
            )
            process_weights_after_loading_fn(model_sublayer_name, param)

        # WINT4 mode: all params (Linear + MoE) are int8.
        # FP8 dequant → int4 quantize → pack int8 → write directly to int8 params.
        _enable_wint4 = os.environ.get("FD_WINT4_QUANTIZE", "0") == "1"

        # Dequantize FP8 weights using block-wise scales.
        if fp8_weights:
            logger.info(f"Dequantizing {len(fp8_weights)} FP8 weights (CPU numpy"
                        f"{', WINT4 int4 quantize' if _enable_wint4 else ''}) ...")
            BLOCK_SIZE = 128
            tp_size = self.fd_config.parallel_config.tensor_parallel_size
            tp_rank = self.fd_config.parallel_config.tensor_parallel_rank

            for wname, wt in fp8_weights.items():
                scale_name = wname.replace(".weight", ".weight_scale_inv")
                scale = fp8_scales.get(scale_name)

                if scale is None:
                    logger.warning(f"No scale for {wname}, loading raw fp8 as bf16")
                    wt_dq = get_tensor(wt).cast("bfloat16")
                else:
                    wt_f32_t = get_tensor(wt).cast("float32")
                    sc_np = get_tensor(scale).numpy()
                    wt_np = wt_f32_t.numpy()
                    del wt_f32_t

                    out_dim, in_dim = wt_np.shape
                    n_blocks_r = (out_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
                    n_blocks_c = (in_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
                    pad_r = n_blocks_r * BLOCK_SIZE - out_dim
                    pad_c = n_blocks_c * BLOCK_SIZE - in_dim
                    if pad_r > 0 or pad_c > 0:
                        wt_np = np.pad(wt_np, ((0, pad_r), (0, pad_c)))
                    wt_blocked = wt_np.reshape([n_blocks_r, BLOCK_SIZE, n_blocks_c, BLOCK_SIZE])
                    sc_expanded = sc_np.reshape([n_blocks_r, n_blocks_c])[:, np.newaxis, :, np.newaxis]
                    wt_dequant = (wt_blocked * sc_expanded).reshape(
                        [n_blocks_r * BLOCK_SIZE, n_blocks_c * BLOCK_SIZE])[:out_dim, :in_dim]
                    del wt_np, sc_np, wt_blocked, sc_expanded

                    wt_dq = paddle.to_tensor(wt_dequant, dtype="bfloat16")
                    del wt_dequant

                # Load immediately
                matched = False
                for mapping in expert_params_mapping:
                    param_name_e, weight_name_e, expert_id, shard_id = mapping
                    if weight_name_e not in wname:
                        continue
                    model_param_name = wname.replace(weight_name_e, param_name_e)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, wt_dq, shard_id=shard_id, expert_id=expert_id)
                    msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$",
                                 "", model_param_name)
                    process_weights_after_loading_fn(msn, param)
                    matched = True
                    break

                if not matched:
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in wname or "mlp.experts" in wname:
                            continue
                        model_param_name = wname.replace(weight_name, param_name)
                        if model_param_name not in params_dict:
                            continue
                        param = params_dict[model_param_name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader(self.fd_config))
                        weight_loader(param, wt_dq, shard_id)
                        msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$",
                                     "", model_param_name)
                        process_weights_after_loading_fn(msn, param)
                        matched = True
                        break

                if not matched:
                    if wname in params_dict:
                        param = params_dict[wname]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader(self.fd_config))
                        weight_loader(param, wt_dq)
                        msn = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$",
                                     "", wname)
                        process_weights_after_loading_fn(msn, param)

                del wt_dq

            logger.info(f"FP8 dequant {'+ WINT4' if _enable_wint4 else ''} complete")

        # WINT4: quantize Linear layers after loading
        if _enable_wint4:
            self._wint4_quantize_linear_layers()
            paddle.device.cuda.empty_cache()
            self._wint4_quantize_moe_experts()
            paddle.device.cuda.empty_cache()

        # Transpose all Linear weights after loading.
        # ColumnParallelLinear/RowParallelLinear: BlockWiseFP8LinearMethod does NOT
        # transpose for torch format + FP8 checkpoint, so we must do it here.
        # ReplicatedLinear (e.g. MoE gate): UnquantizedLinearMethod would also
        # transpose, but that happens in process_final_after_loading. We include it
        # here so that direct callers (e.g. run_fd_gen.py) work. process_final_after_loading
        # is aware and will skip re-transposing via a guard.
        if self.fd_config.model_config.model_format == "torch":
            from fastdeploy.model_executor.utils import process_weight_transpose
            from fastdeploy.model_executor.layers.linear import (
                ColumnParallelLinear, RowParallelLinear, ReplicatedLinear,
            )
            for name, sublayer in self.named_sublayers():
                if isinstance(sublayer, (ColumnParallelLinear, RowParallelLinear, ReplicatedLinear)):
                    if getattr(sublayer, "_torch_weight_transposed", False):
                        continue  # Already handled (e.g. WINT4 quantized layers)
                    if hasattr(sublayer, "weight") and sublayer.weight is not None:
                        if sublayer.weight.ndim == 2:
                            logger.info(f"Transposing {name} weight {sublayer.weight.shape}")
                            process_weight_transpose(sublayer, "weight")
                            sublayer._torch_weight_transposed = True

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def empty_input_forward(self, forward_meta):
        """Warm-up empty forward for MoE expert routing initialization."""
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(self.fd_config.model_config.num_hidden_layers):
            self.model.layers[i].mlp.experts(
                fake_hidden_states, self.model.layers[i].mlp.gate, forward_meta
            )

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        ids_remove_padding = inputs["ids_remove_padding"]
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding, forward_meta=forward_meta
        )
        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend."""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)


class MiniMaxM2PretrainedModel(PretrainedModel):
    """Pretrained model wrapper for MiniMax-M2.5."""

    config_class = FDConfig

    def _init_weight(self, layer):
        return None

    @classmethod
    def arch_name(cls):
        return "MiniMaxM2ForCausalLM"
