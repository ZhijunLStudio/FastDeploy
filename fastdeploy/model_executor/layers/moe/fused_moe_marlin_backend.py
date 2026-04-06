"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Callable

import paddle
from paddle import nn

import fastdeploy
from fastdeploy.model_executor.ops.gpu import (
    MoeWna16MarlinGemmApi,
    tritonmoe_preprocess_func,
)

from ..quantization.quant_base import QuantMethodBase


def gptq_marlin_moe_repack(
    b_q_weight: paddle.Tensor,
    perm: paddle.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> paddle.Tensor:
    """
    Util function.
    """
    from fastdeploy.model_executor.ops.gpu import gptq_marlin_repack

    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = paddle.empty(
        [num_experts, size_k // 16, size_n * (num_bits // 2)],
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = gptq_marlin_repack(b_q_weight[e], perm[e], size_k, size_n, num_bits)
    return output


def get_scale_perms():
    """
    Util function.
    """
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: paddle.Tensor, size_k: int, size_n: int, group_size: int) -> paddle.Tensor:
    """
    Util function.
    """
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape([-1, len(scale_perm)])[:, scale_perm]
    else:
        s = s.reshape([-1, len(scale_perm_single)])[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_moe_permute_scales(
    s: paddle.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    """
    Util function.
    """
    num_experts = s.shape[0]
    output = paddle.empty(
        [num_experts, s.shape[1], s.shape[2]],
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def pack_fp8_to_int32(fp8_tensor: paddle.Tensor, size_k_first: bool = True) -> paddle.Tensor:
    """Pack FP8 tensor to int32 (4 FP8 per int32).

    Args:
        fp8_tensor: [M, K] float8_e4m3fn tensor
        size_k_first: if True, K is the first dimension after transpose
    Returns:
        int32_tensor: [M, K//4] if size_k_first, else [M//4, K]
    """
    if size_k_first:
        fp8_tensor = fp8_tensor.T
    fp8_tensor = fp8_tensor.contiguous()
    int32_tensor = fp8_tensor.view("int32")
    if size_k_first:
        int32_tensor = int32_tensor.T.contiguous()
    return int32_tensor


class MarlinWeightOnlyMoEMethod(QuantMethodBase):
    """
    Use Marlin Group Gemm to compute Fused MoE.
    Supports both INT4 (uint4b8) and FP8 (float8_e4m3fn) weight types.
    """

    def __init__(self, quant_method=None):
        """
        Marlin Group Gemm to compute Fused MoE.
        """
        self.quant_method = quant_method
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]
        self.added_zeros_attrs = ["zeros0", "zeros1"]

        # Determine weight type from quant_method
        if quant_method is not None:
            # quant_method could be a QuantConfig (e.g. BlockWiseFP8Config) or a QuantMethod
            if hasattr(quant_method, 'weight_block_size'):
                bs = quant_method.weight_block_size
                if bs[0] > 0 and bs[1] > 0:
                    self.weight_type = "fp8"
                    self.block_size = bs[0]
                else:
                    self.weight_type = "int4"
                    self.block_size = None
            elif hasattr(quant_method, 'quant_config') and hasattr(quant_method.quant_config, 'weight_block_size'):
                bs = quant_method.quant_config.weight_block_size
                if bs[0] > 0 and bs[1] > 0:
                    self.weight_type = "fp8"
                    self.block_size = bs[0]
                else:
                    self.weight_type = "int4"
                    self.block_size = None
            else:
                self.weight_type = "int4"
                self.block_size = None
        else:
            self.weight_type = "int4"
            self.block_size = None

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        self.default_dtype = layer._helper.get_default_dtype()
        self.weight_dtype = "int32"

        up_gate_proj_weight_name = self.added_weight_attrs[0]
        down_proj_weight_name = self.added_weight_attrs[1]

        if self.weight_type == "fp8":
            # FP8: num_bits=8, pack_factor=4
            # weight shape: [num_experts, size_k // 16, size_n * (num_bits // 2)]
            #             = [num_experts, size_k // 16, size_n * 4]
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size // 16,
                layer.moe_intermediate_size * 4 * 2,  # *2 for gate+up
            ]
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size // 16,
                layer.hidden_size * 4,
            ]
        else:
            # INT4: num_bits=4, pack_factor=8
            # weight shape: [num_experts, size_k // 16, size_n * (num_bits // 2)]
            #             = [num_experts, size_k // 16, size_n * 2]
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size // 16,
                layer.moe_intermediate_size * 4,
            ]
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size // 16,
                layer.hidden_size * 2,
            ]

        setattr(
            layer,
            up_gate_proj_weight_name,
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            down_proj_weight_name,
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        # weight_scale shape depends on weight type
        if self.weight_type == "fp8":
            # FP8 block quantization: scale per block
            # For FP8 Marlin, scales are permuted to [n_blocks_k, size_n]
            n_blocks_k_up = (layer.hidden_size + self.block_size - 1) // self.block_size
            n_blocks_k_down = (layer.moe_intermediate_size + self.block_size - 1) // self.block_size
            scale_shape_up = [layer.num_local_experts, n_blocks_k_up, layer.moe_intermediate_size * 2]
            scale_shape_down = [layer.num_local_experts, n_blocks_k_down, layer.hidden_size]
        else:
            # INT4 channel-wise: [num_experts, 1, size_n]
            scale_shape_up = [layer.num_local_experts, 1, layer.moe_intermediate_size * 2]
            scale_shape_down = [layer.num_local_experts, 1, layer.hidden_size]

        setattr(
            layer,
            self.added_scale_attrs[0],
            layer.create_parameter(
                shape=scale_shape_up,
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_scale_attrs[1],
            layer.create_parameter(
                shape=scale_shape_down,
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Marlin MoE load weight process.
        Supports both INT4 (BF16 input) and FP8 (float8_e4m3fn input).
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

        up_gate_proj_tensor = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_tensor = paddle.stack(down_proj_weights, axis=0)

        # Detect weight type
        is_fp8 = str(up_gate_proj_tensor.dtype).find("float8") >= 0
        if is_fp8:
            self.weight_type = "fp8"
            num_bits = 8
            # FP8 block quantization: need scales
            if self.block_size is None:
                self.block_size = 128  # default for MiniMax
        else:
            self.weight_type = "int4"
            num_bits = 4

        for idx, weight_tensor in enumerate([up_gate_proj_tensor, down_proj_tensor]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            if is_fp8:
                # FP8 path: pack FP8 to int32, repack to Marlin format
                weight_scale = self._process_fp8_weights(
                    layer, weight_tensor, weight_name, scale_name, num_bits
                )
            else:
                # INT4 path: existing logic
                weight_scale = self._process_int4_weights(
                    weight_tensor, weight_name, scale_name
                )

    def _process_fp8_weights(self, layer, weight_tensor, weight_name, scale_name, num_bits):
        """Process FP8 weights for Marlin kernel.

        Args:
            weight_tensor: [E, K, N] float8_e4m3fn tensor
            weight_name: name of the weight parameter
            scale_name: name of the scale parameter
            num_bits: 8 for FP8
        """
        from fastdeploy.model_executor.ops.gpu import gptq_marlin_repack

        E, K, N = weight_tensor.shape
        group_size = self.block_size
        n_blocks_k = (K + group_size - 1) // group_size
        n_blocks_n = (N + group_size - 1) // group_size

        # Process each expert
        marlin_qweights = []
        marlin_scales = []

        for i in range(E):
            # Pack FP8 to int32: [K, N] -> [K, N//4]
            qweight = pack_fp8_to_int32(weight_tensor[i], size_k_first=False)
            # Transpose: [N//4, K]
            qweight = qweight.T.contiguous()

            # Repack to Marlin format
            perm = paddle.empty([0], dtype="int32")
            marlin_qw = gptq_marlin_repack(qweight, perm, K, N, num_bits)
            marlin_qweights.append(marlin_qw)

            # For FP8, we need to compute scales from the weight
            # The FP8 weight is already quantized, so we need to reconstruct scales
            # from the block-wise quantization. But we don't have the original scales
            # here - they come from the checkpoint as weight_scale_inv.
            # So we'll create a placeholder scale and it will be set separately.
            # For now, create identity-like scales
            s_placeholder = paddle.ones([n_blocks_k, N], dtype="float32")
            marlin_s = marlin_permute_scales(s_placeholder, K, N, group_size)
            marlin_scales.append(marlin_s)

        marlin_qweight = paddle.stack(marlin_qweights, axis=0)
        marlin_scale = paddle.stack(marlin_scales, axis=0)

        # Set weight
        getattr(layer, weight_name).set_value(marlin_qweight)
        # Scale will be set separately via set_fp8_scales
        getattr(layer, scale_name).set_value(marlin_scale.cast(self.default_dtype))

        return marlin_scale

    def set_fp8_scales(self, layer, up_gate_scale, down_scale):
        """Set FP8 scales for Marlin kernel.

        Args:
            layer: the MoE layer
            up_gate_scale: [E, n_blocks_k_up, n_blocks_n_up] float32 tensor
            down_scale: [E, n_blocks_k_down, n_blocks_n_down] float32 tensor
        """
        if up_gate_scale is None or down_scale is None:
            return

        E, K, N_up = getattr(layer, self.added_weight_attrs[0]).shape
        # N_up is in Marlin format, need to compute actual N
        # For FP8: actual_n = N * 4 / (num_bits // 2) ... but this is complex
        # Instead, use the scale shape directly
        group_size = self.block_size

        for idx, (scale_tensor, scale_name) in enumerate([
            (up_gate_scale, self.added_scale_attrs[0]),
            (down_scale, self.added_scale_attrs[1]),
        ]):
            if idx == 0:
                # up_gate: size_k = hidden_size, size_n = moe_intermediate_size * 2
                size_k = layer.hidden_size
                size_n = layer.moe_intermediate_size * 2
            else:
                # down: size_k = moe_intermediate_size, size_n = hidden_size
                size_k = layer.moe_intermediate_size
                size_n = layer.hidden_size

            n_blocks_k = scale_tensor.shape[1]
            n_blocks_n = scale_tensor.shape[2]

            # Expand scales from block-wise to group-wise
            # [E, n_blocks_k, n_blocks_n] -> [E, n_blocks_k, size_n]
            marlin_scales = []
            for e in range(scale_tensor.shape[0]):
                s = scale_tensor[e]  # [n_blocks_k, n_blocks_n]
                block_n = self.block_size
                s_expanded = s.unsqueeze(2).expand(
                    [s.shape[0], s.shape[1], block_n]
                ).reshape([s.shape[0], s.shape[1] * block_n])
                s_expanded = s_expanded[:, :size_n]
                marlin_s = marlin_permute_scales(s_expanded, size_k, size_n, group_size)
                marlin_scales.append(marlin_s)

            marlin_scale = paddle.stack(marlin_scales, axis=0)
            getattr(layer, scale_name).set_value(marlin_scale.cast(self.default_dtype))

    def _process_int4_weights(self, weight_tensor, weight_name, scale_name):
        """Process INT4 weights for Marlin kernel (existing logic)."""
        max_bound = 7

        weight_scale = weight_tensor.abs().max(axis=1)
        quanted_weight = weight_tensor / weight_scale[:, None, :] * max_bound
        quanted_weight = paddle.round(quanted_weight).astype("int32")

        quanted_weight[quanted_weight > 7] = 7
        quanted_weight[quanted_weight < -7] = -7
        quanted_weight += 8

        E, K, N = quanted_weight.shape
        quanted_weight = quanted_weight.reshape([0, K // 8, 8, N])
        res = paddle.zeros([E, K // 8, N], dtype="int32")
        for j in range(8):
            tmp = quanted_weight[:, :, j, :]
            res = res | (tmp << (j * 4))
        quanted_weight = paddle.assign(res)
        weight_scale = weight_scale / max_bound
        weight_scale = weight_scale[:, None, :]

        group_size = -1  # means per_channel

        g_idx_sort_indices = paddle.empty([E, 0], dtype="int32")
        quanted_weight = gptq_marlin_moe_repack(
            quanted_weight,
            g_idx_sort_indices,
            K,
            N,
            4,
        )

        weight_scale = marlin_moe_permute_scales(
            weight_scale,
            size_k=K,
            size_n=N,
            group_size=group_size,
        )

        for name, tensor in [
            (weight_name, quanted_weight),
            (scale_name, weight_scale),
        ]:
            getattr(layer, name).set_value(tensor)

        return weight_scale

    def init_ep(self, layer):
        """Initialize NCCL-based EP runner for SM80 (no deep_ep required)."""
        from .nccl_ep_runner import NCCLEPPrefillRunner
        fd_config = layer.fd_config
        ep_size = layer.ep_size
        ep_rank = fd_config.parallel_config.expert_parallel_rank
        ep_group = fd_config.parallel_config.ep_group
        num_local_experts = layer.num_local_experts
        layer._nccl_ep_runner = NCCLEPPrefillRunner(
            ep_size, ep_rank, ep_group, num_local_experts
        )

    def apply_ep(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Marlin FP8 MoE with Expert Parallel via NCCL all-to-all (SM80 compatible).

        Flow:
          1. Routing: compute global top-k expert IDs
          2. Dispatch: NCCL all-to-all to send tokens to expert-owner ranks
          3. Local Marlin GEMM: run only local experts on received tokens
          4. Combine: NCCL all-to-all to return results to originating ranks
        """
        from fastdeploy.model_executor.layers.moe.moe import get_moe_scores

        M, hidden_size = x.shape
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts

        # Step 1: Routing
        gate_out = gate(x).cast("float32")
        _, topk_weights, topk_ids = get_moe_scores(
            gate_out,
            layer.n_group,
            layer.topk_group,
            top_k,
            layer.routed_scaling_factor,
            layer.gate_correction_bias,
            getattr(layer, "renormalize", True),
        )
        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        # Step 2: Dispatch via NCCL
        runner = layer._nccl_ep_runner
        recv_x, recv_local_eids, recv_ws, recv_orig, send_counts, recv_counts = runner.dispatch(
            x, topk_ids, topk_weights
        )
        # Ensure correct dtype for downstream ops
        recv_local_eids = recv_local_eids.cast("int64")

        R = recv_x.shape[0]
        ffn_outs = paddle.zeros([R, hidden_size], dtype=x.dtype)

        if R > 0:
            # Max number of local experts any received token has on this rank
            valid_mask = (recv_local_eids >= 0)
            max_local_k = int(valid_mask.sum(axis=1).max().item())
            if max_local_k == 0:
                max_local_k = 1

            # Clip to max_local_k, pad -1 with 0 and zero weights
            local_eids = recv_local_eids[:, :max_local_k].clone()
            local_ws = recv_ws[:, :max_local_k].clone()
            pad_mask = (local_eids < 0)
            if pad_mask.any():
                local_eids = paddle.where(pad_mask, paddle.zeros_like(local_eids), local_eids)
                local_ws = paddle.where(
                    pad_mask.cast("float32") > 0,
                    paddle.zeros_like(local_ws), local_ws
                )

            block_size_m = 64
            for m in [8, 16, 32, 48, 64]:
                if R * max_local_k / num_local_experts / m < 0.9:
                    block_size_m = m
                    break

            sorted_token_ids, expert_ids, num_tokens_pp = tritonmoe_preprocess_func(
                local_eids.cast("int64"), num_local_experts, block_size_m
            )

            up_gate_weight = layer.up_gate_proj_weight
            down_weight = layer.down_proj_weight
            actual_size_k_up = up_gate_weight.shape[1] * 16
            actual_size_n_up = up_gate_weight.shape[2] // 4
            actual_size_k_down = down_weight.shape[1] * 16
            actual_size_n_down = down_weight.shape[2] // 4

            b_q_type_str = "float8_e4m3fn" if self.weight_type == "fp8" else "uint4b8"
            workspace = paddle.empty([528], dtype="int32")

            ffn_out = MoeWna16MarlinGemmApi(
                recv_x, None,
                b_q_weight=up_gate_weight,
                b_scales=layer.up_gate_proj_weight_scale,
                global_scale_or_none=None, b_zeros_or_none=None,
                g_idx_or_none=None, perm_or_none=None,
                workspace=workspace,
                sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_pp,
                topk_weights=local_ws, moe_block_size=block_size_m,
                top_k=max_local_k, mul_topk_weights=False, is_ep=False,
                b_q_type_str=b_q_type_str,
                size_m=R, size_n=actual_size_n_up, size_k=actual_size_k_up,
                is_k_full=True, use_atomic_add=True, use_fp32_reduce=True, is_zp_float=False,
            )[0]

            swiglu_out = paddle.nn.functional.swiglu(ffn_out)

            ffn_outs = MoeWna16MarlinGemmApi(
                swiglu_out, None,
                b_q_weight=down_weight,
                b_scales=layer.down_proj_weight_scale,
                global_scale_or_none=None, b_zeros_or_none=None,
                g_idx_or_none=None, perm_or_none=None,
                workspace=workspace,
                sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_pp,
                topk_weights=local_ws, moe_block_size=block_size_m,
                top_k=1, mul_topk_weights=True, is_ep=False,
                b_q_type_str=b_q_type_str,
                size_m=R * max_local_k, size_n=actual_size_n_down, size_k=actual_size_k_down,
                is_k_full=True, use_atomic_add=True, use_fp32_reduce=True, is_zp_float=False,
            )[0]

            ffn_outs = ffn_outs.reshape([R, max_local_k, hidden_size]).sum(axis=1)

        # Step 4: Combine via NCCL
        output = runner.combine(M, ffn_outs, recv_ws, recv_orig, send_counts, recv_counts)
        return output

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Marlin compute Fused MoE. Routes to apply_ep() when ep_size > 1.
        """
        if getattr(layer, 'ep_size', 1) > 1:
            return self.apply_ep(layer, x, gate, topk_ids_hookfunc, shared_experts)

        gate_out = gate(x)
        gate_out = gate_out.cast("float32")
        token_num = x.shape[0]
        top_k = layer.top_k
        top_k = layer.top_k
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size
        num_experts = layer.num_experts
        topk_method = layer.topk_method

        if topk_method == "noaux_tc":
            from fastdeploy.model_executor.layers.moe.moe import get_moe_scores

            _, topk_weights, topk_ids = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
            )
        else:
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                top_k,
                True,  # apply_norm_weight,
                False,
            )

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        block_size_m = 64

        for m in [8, 16, 32, 48, 64]:
            if token_num * top_k / num_experts / m < 0.9:
                block_size_m = m
                break

        topk = top_k

        # for H100 132 sms
        workspace = paddle.empty([528], dtype="int32")

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
            topk_ids, num_experts, block_size_m
        )

        # Determine b_q_type_str based on weight type
        if self.weight_type == "fp8":
            b_q_type_str = "float8_e4m3fn"
        else:
            b_q_type_str = "uint4b8"

        # For FP8, size_n and size_k are derived from actual weight shapes
        # to handle TP-sharded vs full expert weights correctly.
        if self.weight_type == "fp8":
            # FP8: weight shape is [E, K//16, N*4], so:
            #   actual K = weight.shape[1] * 16
            #   actual N = weight.shape[2] // 4
            up_gate_weight = layer.up_gate_proj_weight
            actual_size_k_up = up_gate_weight.shape[1] * 16
            actual_size_n_up = up_gate_weight.shape[2] // 4
            down_weight = layer.down_proj_weight
            actual_size_k_down = down_weight.shape[1] * 16
            actual_size_n_down = down_weight.shape[2] // 4
        else:
            actual_size_k_up = hidden_size
            actual_size_n_up = moe_intermediate_size * 2
            actual_size_k_down = moe_intermediate_size
            actual_size_n_down = hidden_size

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Marlin FP8 apply: b_q_type={b_q_type_str}, "
                     f"up_gate: size_n={actual_size_n_up}, size_k={actual_size_k_up}, "
                     f"down: size_n={actual_size_n_down}, size_k={actual_size_k_down}, "
                     f"up_gate_weight={layer.up_gate_proj_weight.shape}, "
                     f"up_gate_scale={layer.up_gate_proj_weight_scale.shape}, "
                     f"down_weight={layer.down_proj_weight.shape}, "
                     f"down_scale={layer.down_proj_weight_scale.shape}, "
                     f"x={x.shape}, token_num={token_num}")

        ffn_out = MoeWna16MarlinGemmApi(
            x,
            c_or_none=None,
            b_q_weight=layer.up_gate_proj_weight,
            b_scales=layer.up_gate_proj_weight_scale,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type_str=b_q_type_str,
            size_m=token_num,
            size_n=actual_size_n_up,
            size_k=actual_size_k_up,
            is_k_full=True,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        swiglu_out = paddle.nn.functional.swiglu(ffn_out)

        ffn_out = MoeWna16MarlinGemmApi(
            swiglu_out,
            c_or_none=None,
            b_q_weight=layer.down_proj_weight,
            b_scales=layer.down_proj_weight_scale,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type_str=b_q_type_str,
            size_m=token_num * topk,
            size_n=actual_size_n_down,
            size_k=actual_size_k_down,
            is_k_full=True,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        ffn_out.reshape_([token_num, -1, hidden_size])
        ffn_out = ffn_out.sum(axis=1)

        return ffn_out
