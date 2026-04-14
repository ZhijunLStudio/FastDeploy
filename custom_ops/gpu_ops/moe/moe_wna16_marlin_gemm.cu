/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/api/include/api.h"

#include "moe/moe_wna16_marlin_utils/kernel.h"
#include "moe/moe_wna16_marlin_utils/core/scalar_type.hpp"
#include "moe/moe_wna16_marlin_gemm.h"
#include "helper.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace MARLIN_NAMESPACE_NAME {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m,
    int size_k,
    int top_k) {
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
  int32_t block_sorted_ids[moe_block_size];
  int block_num_valid_tokens = 0;
  int64_t old_expert_id = 0;
  int64_t expert_id = 0;
  int row_stride = size_k * sizeof(half) / 16;

  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] =
          ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= size_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }
  };

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int in_offset = (row / top_k) * row_stride;
    int out_offset = row * row_stride;

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + in_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
    old_expert_id = expert_id;
    int tmp_expert_id = expert_ids_ptr[index];
    if (tmp_expert_id == -1) continue;
    expert_id = tmp_expert_id;
    perm_int_ptr += (expert_id - old_expert_id) * size_k;
    read_moe_block_data(index);

    for (int i = 0; i < block_num_valid_tokens; i++)
      permute_row(block_sorted_ids[i]);
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * stages * 2;          // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, bool m_block_size_8,
                          int thread_m_blocks, int prob_m, int prob_n,
                          int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int has_zp,
                          int is_zp_float, bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size +
                   sh_g_idx_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, bool m_block_size_8,
                     int thread_m_blocks, int prob_m, int prob_n, int prob_k,
                     int num_bits, int group_size, bool has_act_order,
                     bool is_k_full, int has_zp, int is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size =
      get_kernel_cache_size(th_config, m_block_size_8, thread_m_blocks, prob_m,
                            prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  // Extract type ids for kernel_selector.h dispatch
  auto a_type_id = a_type.id();
  auto b_type_id = b_type.id();
  auto c_type_id = c_type.id();
  auto s_type_id = s_type.id();

#include "moe/moe_wna16_marlin_utils/kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int num_experts, int top_k, int thread_m_blocks,
    bool m_block_size_8, int num_bits, int group_size, bool has_act_order,
    bool is_k_full, bool has_zp, bool is_zp_float, bool is_a_8bit, int stages,
    int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, m_block_size_8, thread_m_blocks, prob_m,
                         prob_n, prob_k, num_bits, group_size, has_act_order,
                         is_k_full, has_zp, is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, m_block_size_8, thread_m_blocks, prob_m, prob_n, prob_k,
        num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float,
        is_a_8bit, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : (group_size / 16);
    }

    auto kernel =
        get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                          th_config.thread_n / 16, th_config.thread_k / 16,
                          m_block_size_8, has_act_order, has_zp, group_blocks,
                          th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
    int allow_count = min(device_max_reg_size / reg_size,
                          max_shared_mem / (cache_size + 1536));
    if (thread_m_blocks == 1)
      allow_count = max(min(allow_count, 4), 1);
    else
      allow_count = max(min(allow_count, 2), 1);

    if (prob_n / th_config.thread_n * prob_m * top_k * 4 < sms * allow_count) {
      allow_count =
          max(prob_n / th_config.thread_n * prob_m * top_k * 4 / sms, 1);
    }

    if (allow_count > count) {
      count = allow_count;
      exec_cfg = {count, th_config};
    };
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, void* sorted_token_ids,
               void* expert_ids, void* num_tokens_past_padded,
               void* topk_weights, int moe_block_size, int num_experts,
               int top_k, bool mul_topk_weights, int prob_m, int prob_n,
               int prob_k, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;
  bool is_a_8bit = a_type.size_bits() == 8;

  PADDLE_ENFORCE(prob_m > 0 && prob_n > 0 && prob_k > 0,
                 "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      PADDLE_ENFORCE(group_size != -1, "group_size must != -1");
      group_blocks = group_size / 16;
      PADDLE_ENFORCE(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                     " is not divisible by group_blocks = ", group_blocks);
    } else {
      PADDLE_ENFORCE(group_size == 0, "group_size must == 0");
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      PADDLE_ENFORCE(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                     " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const uint16_t* g_s_ptr = (const uint16_t*)g_s;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr =
      (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    auto kernel = permute_cols_kernel<8>;
    if (moe_block_size == 8) {
    } else if (moe_block_size == 16)
      kernel = permute_cols_kernel<16>;
    else if (moe_block_size == 32)
      kernel = permute_cols_kernel<32>;
    else if (moe_block_size == 48)
      kernel = permute_cols_kernel<48>;
    else if (moe_block_size == 64)
      kernel = permute_cols_kernel<64>;
    else
      PADDLE_ENFORCE(false, "unsupported moe_block_size ", moe_block_size);

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
    // clang-format on
    A_ptr = a_tmp_ptr;
    prob_m = prob_m * top_k;
    top_k = 1;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  PADDLE_ENFORCE(max_shared_mem > 0, "max_shared_mem must > 0");

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  PADDLE_ENFORCE(major_capability * 10 + minor_capability >= 75,
                 "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    PADDLE_ENFORCE(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                   "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    PADDLE_ENFORCE(major_capability * 10 + minor_capability >= 89,
                   "FP8 only support Ada Lovelace or newer GPUs.");
    PADDLE_ENFORCE(
        major_capability * 10 + minor_capability == 89 ||
            major_capability * 10 + minor_capability == 120,
        "Marlin W4A8-FP8 only support SM89 or SM120 device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, thread_k * thread_n / 64};
    if (blocks_per_sm == -1) blocks_per_sm = 1;
    exec_cfg = exec_config_t{blocks_per_sm, thread_tfg};
    PADDLE_ENFORCE(prob_n % thread_n == 0, "prob_n = ", prob_n,
                   " is not divisible by thread_n = ", thread_n);
    PADDLE_ENFORCE(prob_k % thread_k == 0, "prob_k = ", prob_k,
                   " is not divisible by thread_k = ", thread_k);
  } else {
    // Auto config
    exec_cfg = determine_exec_config(
        a_type, b_type, c_type, s_type, prob_m, prob_n, prob_k, num_experts,
        top_k, thread_m_blocks, m_block_size_8, num_bits, group_size,
        has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages,
        max_shared_mem, sms);
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1)
    max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  PADDLE_ENFORCE(is_valid_config(thread_tfg, m_block_size_8, thread_m_blocks,
                                 prob_m, prob_n, prob_k, num_bits, group_size,
                                 has_act_order, is_k_full, has_zp, is_zp_float,
                                 is_a_8bit, stages, max_shared_mem),
                 "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
                 ", thread_k = ", thread_tfg.thread_k,
                 ", thread_n = ", thread_tfg.thread_n,
                 ", num_threads = ", thread_tfg.num_threads, " for MKN = [",
                 prob_m, ", ", prob_k, ", ", prob_n,
                 "] and num_bits = ", num_bits, ", group_size = ", group_size,
                 ", has_act_order = ", has_act_order,
                 ", is_k_full = ", is_k_full, ", has_zp = ", has_zp,
                 ", is_zp_float = ", is_zp_float,
                 ", max_shared_mem = ", max_shared_mem);

  int sh_cache_size =
      get_kernel_cache_size(thread_tfg, m_block_size_8, thread_m_blocks, prob_m,
                            prob_n, prob_k, num_bits, group_size, has_act_order,
                            is_k_full, has_zp, is_zp_float, is_a_8bit, stages);

  auto kernel = get_marlin_kernel(
      a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
      thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
      num_threads, is_zp_float, stages);

  if (kernel == MarlinDefault) {
    PADDLE_ENFORCE(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                   ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                   ", num_groups = ", num_groups,
                   ", group_size = ", group_size,
                   ", thread_m_blocks = ", thread_m_blocks,
                   ", thread_n_blocks = ", thread_n_blocks,
                   ", thread_k_blocks = ", thread_k_blocks,
                   ", num_bits = ", num_bits);
  }

  cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  // avoid ">>>" being formatted to "> > >"
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr, g_idx_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, num_groups, prob_m,
      prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce);
  // clang-format on
}

}  // namespace MARLIN_NAMESPACE_NAME

MARLIN_NAMESPACE_NAME::Tensor ConvertPaddleTensorToDetailTensor(
    const paddle::Tensor& tensor) {
  MARLIN_NAMESPACE_NAME::Tensor res(tensor);
  return res;
}

paddle::Tensor ConvertDetailTensorToPaddleTensor(
    const MARLIN_NAMESPACE_NAME::Tensor& tensor) {
  return tensor.raw_tensor();
}

const paddle::optional<MARLIN_NAMESPACE_NAME::Tensor>
ConvertOptionalPaddleTensorToDetailTensor(
    const paddle::optional<paddle::Tensor>& tensor) {
  paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> res;
  if (tensor) {
    res = ConvertPaddleTensorToDetailTensor(tensor.get());
  }
  return res;
}

MARLIN_NAMESPACE_NAME::Tensor moe_wna16_marlin_gemm(
    MARLIN_NAMESPACE_NAME::Tensor& a,
    paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> const& c_or_none,
    MARLIN_NAMESPACE_NAME::Tensor& b_q_weight,
    MARLIN_NAMESPACE_NAME::Tensor& b_scales,
    paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> const& global_scale_or_none,
    paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> const& b_zeros_or_none,
    paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> const& g_idx_or_none,
    paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> const& perm_or_none,
    MARLIN_NAMESPACE_NAME::Tensor& workspace,
    MARLIN_NAMESPACE_NAME::Tensor& sorted_token_ids,
    MARLIN_NAMESPACE_NAME::Tensor& expert_ids,
    MARLIN_NAMESPACE_NAME::Tensor& num_tokens_past_padded,
    MARLIN_NAMESPACE_NAME::Tensor& topk_weights,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    MARLIN_NAMESPACE_NAME::ScalarTypeId const& b_q_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  // Determine a_type, c_type, s_type from input tensor dtypes (matching vLLM)
  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  if (a.dtype() == paddle::DataType::FLOAT16) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.dtype() == MARLIN_NAMESPACE_NAME::kBFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    // W4A8 or W8A8 activation (INT8 or FP8)
    if (b_scales.dtype() == paddle::DataType::FLOAT16) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.dtype() == MARLIN_NAMESPACE_NAME::kBFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      PADDLE_ENFORCE(c_or_none, "c must be passed for W4A8-FP4");
      MARLIN_NAMESPACE_NAME::Tensor c_check = c_or_none.get();
      if (c_check.dtype() == paddle::DataType::FLOAT16) {
        c_type_id = vllm::kFloat16.id();
      } else if (c_check.dtype() == MARLIN_NAMESPACE_NAME::kBFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        PADDLE_ENFORCE(false, "unsupported c dtype");
      }
    }

    if (a.dtype() == paddle::DataType::FLOAT8_E4M3FN) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.dtype() == paddle::DataType::INT8) {
      a_type_id = vllm::kS8.id();
    } else {
      PADDLE_ENFORCE(false, "unsupported a scalar_type");
    }
  }

  s_type_id = c_type_id;
  // FP4 (e2m1) scale type override - not currently used in FD but kept for
  // completeness
  vllm::ScalarType b_type_check = vllm::ScalarType::from_id(b_q_type_id);
  if (b_type_check == vllm::kFE2M1f) {
    if (b_scales.dtype() == paddle::DataType::FLOAT8_E4M3FN) {
      s_type_id = vllm::kFE4M3fn.id();
    } else {
      PADDLE_ENFORCE(false,
                     "When b_type = float4_e2m1f, b_scale scalar type must be "
                     "float8_e4m3fn (for NVFP4).");
    }
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_q_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  int pack_factor = 32 / b_type.size_bits();
  int num_experts = b_q_weight.size(0);

  if (moe_block_size != 8) {
    PADDLE_ENFORCE(moe_block_size % 16 == 0,
                   "unsupported moe_block_size=", moe_block_size);
    PADDLE_ENFORCE(moe_block_size >= 16 && moe_block_size <= 64,
                   "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  PADDLE_ENFORCE(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
                 ", size_m = ", size_m);
  PADDLE_ENFORCE(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
                 ", size_k = ", size_k);

  // Verify B
  PADDLE_ENFORCE(size_k % MARLIN_NAMESPACE_NAME::tile_size == 0,
                 "size_k = ", size_k,
                 " is not divisible by tile_size = ",
                 MARLIN_NAMESPACE_NAME::tile_size);
  PADDLE_ENFORCE(
      (size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(1),
      "Shape mismatch: b_q_weight.size(1) = ", b_q_weight.size(1),
      ", size_k = ", size_k, ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  PADDLE_ENFORCE(b_q_weight.size(2) % MARLIN_NAMESPACE_NAME::tile_size == 0,
                 "b_q_weight.size(2) = ", b_q_weight.size(2),
                 " is not divisible by tile_size = ",
                 MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(2) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  PADDLE_ENFORCE(size_n == actual_size_n, "size_n = ", size_n,
                 ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  PADDLE_ENFORCE(a.is_gpu(), "A is not on GPU");
  PADDLE_ENFORCE(a.is_contiguous(), "A is not contiguous");

  PADDLE_ENFORCE(b_q_weight.is_gpu(), "b_q_weight is not on GPU");
  PADDLE_ENFORCE(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  PADDLE_ENFORCE(b_scales.is_gpu(), "b_scales is not on GPU");
  PADDLE_ENFORCE(b_scales.is_contiguous(), "b_scales is not contiguous");

  int device_id = a.place().GetDeviceId();

  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id);

  // Alloc buffers
  phi::GPUPlace gpu_place(device_id);
  MARLIN_NAMESPACE_NAME::Tensor c;
  if (c_or_none) {
    c = c_or_none.get();
    PADDLE_ENFORCE(c.is_gpu(), "c is not on GPU");
    PADDLE_ENFORCE(c.is_contiguous(), "c is not contiguous");
    PADDLE_ENFORCE(c.size(0) == size_m * top_k,
                   "Shape mismatch: c.size(0) = ", c.size(0),
                   ", size_m * topk = ", size_m * top_k);
    PADDLE_ENFORCE(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                   ", size_n = ", size_n);
  } else {
    c = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
        {size_m * top_k, size_n}, a.dtype(), phi::GPUPlace(device_id)));
  }

  // Create dummy a_scales tensor (PaddlePaddle API does not pass a_scales yet)
  MARLIN_NAMESPACE_NAME::Tensor a_scales =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {0}, paddle::DataType::FLOAT32, phi::GPUPlace(device_id)));

  // Alloc C tmp buffer that is going to be used for the global reduce
  MARLIN_NAMESPACE_NAME::Tensor c_tmp;
  if (use_fp32_reduce && !use_atomic_add) {
    // max num of threadblocks is sms * 4
    long max_c_tmp_size = std::min(
        (long)size_n * sorted_token_ids.size(0),
        (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
    if (moe_block_size == 8) max_c_tmp_size *= 2;
    c_tmp = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({max_c_tmp_size},
                                    MARLIN_NAMESPACE_NAME::kFloat32,
                                    phi::GPUPlace(device_id)));
  } else {
    c_tmp = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
        {0}, MARLIN_NAMESPACE_NAME::kFloat32, phi::GPUPlace(device_id)));
  }

  // Detect groupsize and act_order
  int group_size = -1;

  int rank = b_scales.dim();
  PADDLE_ENFORCE(rank == 3, "b_scales rank must be 3");
  PADDLE_ENFORCE(b_scales.size(2) == size_n,
                 "b_scales dim 2 = ", b_scales.size(2),
                 " is not size_n = ", size_n);
  int num_groups = b_scales.size(1);

  bool has_act_order = false;
  MARLIN_NAMESPACE_NAME::Tensor g_idx, perm, a_tmp;
  if (g_idx_or_none && perm_or_none) {
    g_idx = g_idx_or_none.get();
    perm = perm_or_none.get();

    PADDLE_ENFORCE(g_idx.is_gpu(), "g_idx is not on GPU");
    PADDLE_ENFORCE(g_idx.is_contiguous(), "g_idx is not contiguous");
    PADDLE_ENFORCE(perm.is_gpu(), "perm is not on GPU");
    PADDLE_ENFORCE(perm.is_contiguous(), "perm is not contiguous");

    // Verify g_idx and perm
    PADDLE_ENFORCE((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                       (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                   "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                   " and perm.size(-1) = ", perm.size(-1),
                   ", where size_k = ", size_k);

    has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;
  } else {
    g_idx = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({0}, a.dtype(), phi::GPUPlace(device_id)));

    perm = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({0}, a.dtype(), phi::GPUPlace(device_id)));

    a_tmp = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({0}, a.dtype(), phi::GPUPlace(device_id)));
  }

  if (has_act_order) {
    a_tmp = ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
        {size_m * top_k, size_k}, a.dtype(), phi::GPUPlace(device_id)));

    if (is_k_full) {
      PADDLE_ENFORCE(num_groups > 1, "For act_order, num_groups must be > 1");
      PADDLE_ENFORCE(size_k % num_groups == 0, "size_k = ", size_k,
                     ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }

  } else {
    a_tmp = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({0}, a.dtype(), phi::GPUPlace(device_id)));

    if (num_groups > 1) {
      PADDLE_ENFORCE(size_k % num_groups == 0, "size_k = ", size_k,
                     ", is not divisible by b_scales.size(1) = ",
                     b_scales.size(1));
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  MARLIN_NAMESPACE_NAME::Tensor global_scale;
  if (global_scale_or_none) {
    global_scale = global_scale_or_none.get();
    PADDLE_ENFORCE(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                   "global_scale can only be used for nvfp4 format.");
  } else {
    global_scale = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({0}, a.dtype(), phi::GPUPlace(device_id)));

    PADDLE_ENFORCE(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                   "the global_scale parameter must be passed for nvfp4 format.");
  }

  // Create dummy b_bias (PaddlePaddle API does not pass b_bias yet)
  bool has_bias = false;
  MARLIN_NAMESPACE_NAME::Tensor b_bias =
      ConvertPaddleTensorToDetailTensor(paddle::experimental::empty(
          {0}, a.dtype(), phi::GPUPlace(device_id)));

  MARLIN_NAMESPACE_NAME::Tensor b_zeros;
  if (b_zeros_or_none) {
    b_zeros = b_zeros_or_none.get();
    PADDLE_ENFORCE(b_zeros.is_gpu(), "b_zeros is not on GPU");
    PADDLE_ENFORCE(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = ConvertPaddleTensorToDetailTensor(
        paddle::experimental::empty({0}, a.dtype(), phi::GPUPlace(device_id)));
  }
  bool has_zp = b_zeros.numel() > 0;

  if (has_zp) {
    PADDLE_ENFORCE(b_type == vllm::kU4 || b_type == vllm::kU8,
                   "b_type must be u4 or u8 when has_zp = True. Got = ",
                   b_type.str());
  } else {
    PADDLE_ENFORCE(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 ||
                       b_type == vllm::kS4 || b_type == vllm::kS8 ||
                       b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                   "b_type must be uint4b8, uint8b128, int4, int8, "
                   "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                   b_type.str());
  }

  if (has_zp && is_zp_float) {
    PADDLE_ENFORCE(
        a.dtype() == paddle::DataType::FLOAT16,
        "Computation type must be float16 (half) when using float zero "
        "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.dim();
    PADDLE_ENFORCE(rank == 3, "b_zeros rank must be 3");
    if (is_zp_float) {
      PADDLE_ENFORCE(b_zeros.size(2) == size_n,
                     "b_zeros dim 2 = ", b_zeros.size(2),
                     " is not size_n = ", size_n);
      PADDLE_ENFORCE(num_groups == b_zeros.size(1),
                     "b_zeros dim 1 = ", b_zeros.size(1),
                     " is not num_groups = ", num_groups);
      PADDLE_ENFORCE(num_groups != -1, "num_groups must be != -1");
    } else {
      PADDLE_ENFORCE(b_zeros.size(1) == num_groups,
                     "b_zeros dim 1 = ", b_zeros.size(1),
                     " is not num_groups = ", num_groups);
      PADDLE_ENFORCE(b_zeros.size(2) == size_n / pack_factor,
                     "b_zeros dim 2 = ", b_zeros.size(2),
                     " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  // Verify workspace size
  PADDLE_ENFORCE(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
                 "size_n = ", size_n, ", is not divisible by min_thread_n = ",
                 MARLIN_NAMESPACE_NAME::min_thread_n);

  int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
  int min_workspace_size = std::min(
      max_n_tiles * (int)(sorted_token_ids.size(0) / moe_block_size), sms * 4);
  PADDLE_ENFORCE(workspace.numel() >= min_workspace_size,
                 "workspace.numel = ", workspace.numel(),
                 " is below min_workspace_size = ", min_workspace_size);

  PADDLE_ENFORCE(a_scales.dtype() == paddle::DataType::FLOAT32,
                 "scalar type of a_scales must be float");

  if (a_type.size_bits() == 16) {
    PADDLE_ENFORCE(a.dtype() == c.dtype(),
                   "scalar type of a must be the same with c for 16 bit activation");
  }

  // Call marlin_mm without scalar_t template - use runtime type dispatch
  MARLIN_NAMESPACE_NAME::marlin_mm(
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(),
      b_bias.data_ptr(), a_scales.data_ptr(), b_scales.data_ptr(),
      global_scale.data_ptr(), b_zeros.data_ptr(), g_idx.data_ptr(),
      perm.data_ptr(), a_tmp.data_ptr(), sorted_token_ids.data_ptr(),
      expert_ids.data_ptr(), num_tokens_past_padded.data_ptr(),
      topk_weights.data_ptr(), moe_block_size, num_experts, top_k,
      mul_topk_weights, size_m, size_n, size_k, workspace.data_ptr(), a_type,
      b_type, c_type, s_type, has_bias, has_act_order, is_k_full, has_zp,
      num_groups, group_size, device_id, a.raw_tensor_.stream(), -1, -1, sms,
      -1, use_atomic_add, use_fp32_reduce, is_zp_float);

  return c;
}

std::vector<paddle::Tensor> MoeWna16MarlinGemmApi(
    const paddle::Tensor& a,
    const paddle::optional<paddle::Tensor>& c_or_none,
    const paddle::Tensor& b_q_weight,
    const paddle::Tensor& b_scales,
    const paddle::optional<paddle::Tensor>& global_scale_or_none,
    const paddle::optional<paddle::Tensor>& b_zeros_or_none,
    const paddle::optional<paddle::Tensor>& g_idx_or_none,
    const paddle::optional<paddle::Tensor>& perm_or_none,
    const paddle::Tensor& workspace,
    const paddle::Tensor& sorted_token_ids,
    const paddle::Tensor& expert_ids,
    const paddle::Tensor& num_tokens_post_padded,
    const paddle::Tensor& topk_weights,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    const std::string& b_q_type_str,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  auto a_ = ConvertPaddleTensorToDetailTensor(a);
  auto b_q_weight_ = ConvertPaddleTensorToDetailTensor(b_q_weight);
  auto b_scales_ = ConvertPaddleTensorToDetailTensor(b_scales);
  auto workspace_ = ConvertPaddleTensorToDetailTensor(workspace);
  auto sorted_token_ids_ = ConvertPaddleTensorToDetailTensor(sorted_token_ids);
  auto expert_ids_ = ConvertPaddleTensorToDetailTensor(expert_ids);
  auto num_tokens_padded_ =
      ConvertPaddleTensorToDetailTensor(num_tokens_post_padded);
  auto topk_weights_ = ConvertPaddleTensorToDetailTensor(topk_weights);

  paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> c_opt_ =
      ConvertOptionalPaddleTensorToDetailTensor(c_or_none);
  paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> global_scale_opt_ =
      ConvertOptionalPaddleTensorToDetailTensor(global_scale_or_none);
  paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> b_zeros_opt_ =
      ConvertOptionalPaddleTensorToDetailTensor(b_zeros_or_none);
  paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> g_idx_opt_ =
      ConvertOptionalPaddleTensorToDetailTensor(g_idx_or_none);
  paddle::optional<MARLIN_NAMESPACE_NAME::Tensor> perm_opt_ =
      ConvertOptionalPaddleTensorToDetailTensor(perm_or_none);

  MARLIN_NAMESPACE_NAME::ScalarTypeId b_q_type_id;
  if (b_q_type_str == "uint4") {
    b_q_type_id = MARLIN_NAMESPACE_NAME::kU4.id();
  } else if (b_q_type_str == "uint4b8") {
    b_q_type_id = MARLIN_NAMESPACE_NAME::kU4B8.id();
  } else if (b_q_type_str == "float8_e4m3fn") {
    b_q_type_id = MARLIN_NAMESPACE_NAME::kFE4M3fn.id();
  } else {
    PADDLE_ENFORCE(false, "b_q_type_str not supported!");
  }

  MARLIN_NAMESPACE_NAME::Tensor out_detail =
      moe_wna16_marlin_gemm(a_,
                            c_opt_,
                            b_q_weight_,
                            b_scales_,
                            global_scale_opt_,
                            b_zeros_opt_,
                            g_idx_opt_,
                            perm_opt_,
                            workspace_,
                            sorted_token_ids_,
                            expert_ids_,
                            num_tokens_padded_,
                            topk_weights_,
                            moe_block_size,
                            top_k,
                            mul_topk_weights,
                            is_ep,
                            b_q_type_id,
                            size_m,
                            size_n,
                            size_k,
                            is_k_full,
                            use_atomic_add,
                            use_fp32_reduce,
                            is_zp_float);
  paddle::Tensor out = ConvertDetailTensorToPaddleTensor(out_detail);
  return {out};
}

std::vector<std::vector<int64_t>> MoeWna16MarlinGemmInferShape(
    const std::vector<int64_t>& a_shape,
    const paddle::optional<std::vector<int64_t>>& c_shape,
    const std::vector<int64_t>& b_q_weight_shape,
    const std::vector<int64_t>& b_scales_shape,
    const paddle::optional<std::vector<int64_t>>& global_scale_shape,
    const paddle::optional<std::vector<int64_t>>& b_zeros_shape,
    const paddle::optional<std::vector<int64_t>>& g_idx_shape,
    const paddle::optional<std::vector<int64_t>>& perm_shape,
    const std::vector<int64_t>& workspace_shape,
    const std::vector<int64_t>& sorted_token_ids_shape,
    const std::vector<int64_t>& expert_ids_shape,
    const std::vector<int64_t>& num_tokens_post_padded_shape,
    const std::vector<int64_t>& topk_weights_shape,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    const std::string& b_q_type_str,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  return {{size_m * top_k, size_n}};
}

std::vector<paddle::DataType> MoeWna16MarlinGemmInferDtype(
    const paddle::DataType& a_dtype,
    const paddle::optional<paddle::DataType>& c_dtype,
    const paddle::DataType& b_q_weight_dtype,
    const paddle::DataType& b_scales_dtype,
    const paddle::optional<paddle::DataType>& global_scale_dtype,
    const paddle::optional<paddle::DataType>& b_zeros_dtype,
    const paddle::optional<paddle::DataType>& g_idx_dtype,
    const paddle::optional<paddle::DataType>& perm_dtype,
    const paddle::DataType& workspace_dtype,
    const paddle::DataType& sorted_token_ids_dtype,
    const paddle::DataType& expert_ids_dtype,
    const paddle::DataType& num_tokens_post_padded_dtype,
    const paddle::DataType& topk_weights_dtype) {
  return {a_dtype};
}
PD_BUILD_STATIC_OP(moe_wna16_marlin_gemm)
    .Inputs({"a",
             paddle::Optional("c_or_none"),
             "b_q_weight",
             "b_scales",
             paddle::Optional("global_scale_or_none"),
             paddle::Optional("b_zeros_or_none"),
             paddle::Optional("g_idx_or_none"),
             paddle::Optional("perm_or_none"),
             "workspace",
             "sorted_token_ids",
             "expert_ids",
             "num_tokens_post_padded",
             "topk_weights"})
    .Outputs({"out"})
    .Attrs({"moe_block_size:int64_t",
            "top_k:int64_t",
            "mul_topk_weights:bool",
            "is_ep:bool",
            "b_q_type_str:std::string",
            "size_m:int64_t",
            "size_n:int64_t",
            "size_k:int64_t",
            "is_k_full:bool",
            "use_atomic_add:bool",
            "use_fp32_reduce:bool",
            "is_zp_float:bool"})
    .SetKernelFn(PD_KERNEL(MoeWna16MarlinGemmApi))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeWna16MarlinGemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeWna16MarlinGemmInferDtype));
