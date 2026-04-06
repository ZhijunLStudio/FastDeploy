"""
NCCL-based Expert Parallel (EP) Runner for SM80 (A100/A800).

Implements token dispatch and combine using paddle.distributed.alltoall_single(),
without requiring the deep_ep library (which needs SM90+/Hopper).

Design:
- dispatch():  all-to-all send tokens to ranks owning the selected experts
- combine():   all-to-all send results back to originating ranks, weighted sum
- apply_marlin_ep(): full EP forward pass with Marlin FP8 GEMM on local experts

Usage (in MarlinWeightOnlyMoEMethod.apply()):
    if layer.ep_size > 1:
        return self._apply_ep_nccl(layer, x, gate, ...)
"""

import paddle
import paddle.distributed as dist
from paddleformers.utils.log import logger


def _alltoall_variable(send_data, send_counts, recv_counts, group):
    """
    Variable-length all-to-all using paddle.distributed.alltoall_single.

    Args:
        send_data: [total_send, ...] flat tensor to send
        send_counts: list of int, how many rows to send to each rank
        recv_counts: list of int, how many rows to receive from each rank
        group: process group

    Returns:
        recv_data: [total_recv, ...] flat tensor received
    """
    total_recv = sum(recv_counts)
    recv_data = paddle.empty(
        [total_recv] + list(send_data.shape[1:]),
        dtype=send_data.dtype
    )
    dist.alltoall_single(
        recv_data, send_data,
        out_split_sizes=recv_counts,
        in_split_sizes=send_counts,
        group=group,
    )
    return recv_data


class NCCLEPPrefillRunner:
    """
    Simple NCCL-based EP runner for prefill (encoder) mode.

    Each GPU owns num_local_experts = num_experts / ep_size experts.
    Token dispatch sends each token's hidden state to every rank that owns
    at least one of its selected experts.

    Memory layout:
        dispatch sends: flat [send_tokens, hidden_size]
        Each sent row has associated metadata: which local experts to run, weights.
    """

    def __init__(self, ep_size: int, ep_rank: int, ep_group, num_local_experts: int):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.expert_id_offset = ep_rank * num_local_experts

    def dispatch(self, x, topk_ids, topk_weights):
        """
        Send each token to all ranks that own at least one of its selected experts.

        Args:
            x: [M, hidden_size] token hidden states (bfloat16)
            topk_ids: [M, top_k] global expert IDs (int64)
            topk_weights: [M, top_k] expert weights (float32)

        Returns:
            recv_x: [R, hidden_size] received token hidden states
            recv_local_eids: [R, top_k] local expert IDs per received token
                             (-1 for padding when token has fewer than top_k experts on this rank)
            recv_weights: [R, top_k] weights per received token-expert pair
            recv_orig_token: [R] original token index on source rank (for combine)
            send_counts: list of send counts per rank
            recv_counts: list of recv counts per rank
        """
        M, H = x.shape
        top_k = topk_ids.shape[1]

        topk_ids_np = topk_ids.numpy()
        topk_weights_np = topk_weights.numpy()

        # Build per-rank send lists
        # to_send[r] = list of (token_idx, [local_exp_ids], [weights])
        to_send = [[] for _ in range(self.ep_size)]

        for i in range(M):
            rank_to_exps = {}  # rank -> ([local_ids], [weights])
            for k in range(top_k):
                eid = int(topk_ids_np[i, k])
                r = eid // self.num_local_experts
                local_eid = eid % self.num_local_experts
                w = float(topk_weights_np[i, k])
                if r not in rank_to_exps:
                    rank_to_exps[r] = ([], [])
                rank_to_exps[r][0].append(local_eid)
                rank_to_exps[r][1].append(w)

            for r, (eids, ws) in rank_to_exps.items():
                to_send[r].append((i, eids, ws))

        send_counts = [len(to_send[r]) for r in range(self.ep_size)]

        # Exchange counts
        send_count_t = paddle.to_tensor(send_counts, dtype="int32")
        recv_count_t = paddle.zeros([self.ep_size], dtype="int32")
        dist.alltoall(
            [recv_count_t[i:i+1] for i in range(self.ep_size)],
            [send_count_t[i:i+1] for i in range(self.ep_size)],
            group=self.ep_group,
        )
        recv_counts = recv_count_t.tolist()

        total_send = sum(send_counts)
        total_recv = sum(recv_counts)

        # Pack send buffers
        send_x = paddle.zeros([total_send, H], dtype=x.dtype)
        send_eids = paddle.full([total_send, top_k], -1, dtype="int32")
        send_ws = paddle.zeros([total_send, top_k], dtype="float32")
        send_orig = paddle.zeros([total_send], dtype="int32")

        offset = 0
        for r in range(self.ep_size):
            for (token_idx, eids, ws) in to_send[r]:
                send_x[offset].set_value(x[token_idx])
                for j, (e, w) in enumerate(zip(eids, ws)):
                    send_eids[offset, j] = e
                    send_ws[offset, j] = w
                send_orig[offset] = token_idx
                offset += 1

        # All-to-all exchange
        recv_x = _alltoall_variable(send_x, send_counts, recv_counts, self.ep_group)
        recv_eids = _alltoall_variable(send_eids, send_counts, recv_counts, self.ep_group)
        recv_ws = _alltoall_variable(send_ws, send_counts, recv_counts, self.ep_group)
        recv_orig = _alltoall_variable(send_orig, send_counts, recv_counts, self.ep_group)

        return recv_x, recv_eids, recv_ws, recv_orig, send_counts, recv_counts

    def combine(self, M, ffn_outs, recv_ws, recv_orig, send_counts, recv_counts):
        """
        Send FFN results back to originating ranks and accumulate.

        Args:
            M: original number of tokens on this rank
            ffn_outs: [R, hidden_size] local FFN outputs, already weighted
                      (mul_topk_weights=True was used in Marlin GEMM, then sum over local experts)
            recv_ws: [R, top_k] weights (for debugging/reference, not used for re-weighting)
            recv_orig: [R] original token indices on source rank
            send_counts: how many tokens this rank sent (during dispatch)
            recv_counts: how many tokens this rank received (during dispatch)

        Returns:
            output: [M, hidden_size] sum of weighted expert outputs per token
        """
        H = ffn_outs.shape[1]

        # Send FFN results back to originating ranks (reverse all-to-all)
        # Note: recv_counts/send_counts are swapped: what we received during dispatch
        # is what we send back, and vice versa.
        send_back = _alltoall_variable(ffn_outs, recv_counts, send_counts, self.ep_group)
        recv_orig_back = _alltoall_variable(recv_orig, recv_counts, send_counts, self.ep_group)

        # Accumulate contributions using vectorized scatter-add
        # recv_orig_back: [total_send] int32, tells us which token each result belongs to
        # send_back: [total_send, H] float - pre-weighted expert outputs
        output = paddle.zeros([M, H], dtype=ffn_outs.dtype)
        if send_back.shape[0] > 0:
            # Use scatter add: output[recv_orig_back[i]] += send_back[i]
            # paddle.scatter only supports float32, so use index_add_ via loop but
            # convert token indices first to avoid .item() overhead
            idx = recv_orig_back.cast("int64")  # [total_send]
            # Vectorized scatter-add: output = scatter_add(output, idx, send_back)
            # PaddlePaddle scatter_nd_add equivalent:
            idx_expanded = idx.unsqueeze(1).expand(send_back.shape)  # [total_send, H]
            output = paddle.put_along_axis(output, idx_expanded, send_back, axis=0, reduce='add')

        return output


def apply_marlin_ep(layer, x, gate, ep_runner, marlin_apply_fn):
    """
    Full EP forward pass using NCCL dispatch/combine with Marlin GEMM.

    Args:
        layer: FusedMoE layer with Marlin weights loaded
        x: [M, hidden_size] token hidden states
        gate: gate network
        ep_runner: NCCLEPPrefillRunner instance
        marlin_apply_fn: function to run local Marlin GEMM
                         signature: (x_local, local_eids, local_weights, layer) -> output

    Returns:
        [M, hidden_size] MoE output
    """
    import numpy as np

    M, H = x.shape

    # Step 1: Compute routing
    gate_out = gate(x).cast("float32")  # [M, num_experts]

    from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import get_moe_scores
    gate_scores, topk_weights, topk_ids = get_moe_scores(
        gate_out,
        layer.n_group,
        layer.topk_group,
        layer.top_k,
        layer.routed_scaling_factor,
        layer.gate_correction_bias,
        getattr(layer, "renormalize", True),
    )

    # Step 2: Dispatch tokens to expert-owning ranks
    recv_x, recv_eids, recv_ws, recv_orig, send_counts, recv_counts = ep_runner.dispatch(
        x, topk_ids, topk_weights
    )

    # Step 3: Run local Marlin GEMM for received tokens
    # recv_eids: [R, top_k] local expert IDs (-1 = no expert on this rank for that slot)
    R = recv_x.shape[0]
    ffn_outs = paddle.zeros([R, H], dtype=x.dtype)

    if R > 0:
        ffn_outs = marlin_apply_fn(recv_x, recv_eids, recv_ws, layer)

    # Step 4: Combine results back
    output = ep_runner.combine(M, ffn_outs, recv_ws, recv_orig, send_counts, recv_counts)

    # Step 5: All-reduce across TP for attention (MoE doesn't need all-reduce since EP)
    return output
