# MiniMax-M2.5 FastDeploy 复现工作报告

## 1. 项目目标

在 FastDeploy (FD) 框架中复现 MiniMax-M2.5 模型（`MiniMaxM2ForCausalLM`），实现权重加载、前向传播、token 生成，并与 vLLM 对齐精度。

**模型关键参数**：
| 参数 | 值 |
|------|-----|
| hidden_size | 3072 |
| num_hidden_layers | 62 |
| num_attention_heads | 48 |
| num_key_value_heads | 8 (GQA) |
| head_dim | 128 |
| num_local_experts | 256 |
| num_experts_per_tok | 8 (top-8) |
| intermediate_size | 1536 |
| rotary_dim | 64 (partial_rotary_factor=0.5) |
| scoring_func | sigmoid |
| vocab_size | 200064 |
| 量化格式 | FP8 (float8_e4m3fn, block_size=128x128) |

---

## 2. 已完成的工作

### 2.1 模型架构实现
- 完整实现了 `MiniMaxM2_5DecoderLayer` (GQA + QK-Norm + Partial RoPE + 256 Expert MoE)
- 支持 `e_score_correction_bias` 路由校正
- 注册为 `@ModelRegistry.register_model_class(architecture="MiniMaxM2ForCausalLM")`

### 2.2 Marlin FP8 MoE 集成
- 通过 `FD_MARLIN_FP8=1` 环境变量启用 Marlin FP8 MoE backend
- SM80 (A100) 支持 FP8 weight-only 量化（Marlin kernel）
- EP=4 模式：4 卡各加载 64 experts，~80 GB/卡

### 2.3 Scale 修复（2026-04-11，commit f0accda78）
- 修复 `_process_fp8_marlin_weights` 的 scale 处理顺序
- 改为先 `_marlin_permute_scales` 再 `2^120` bias（与 vLLM 一致）
- 修复后 scales 与 vLLM 100% 一致（max_diff=0）
- packed weights byte-identical

### 2.4 配置兼容性修改
- 添加 `partial_rotary_factor = rotary_dim / head_dim` 自动计算
- 添加 `num_local_experts` → `num_experts` 映射
- 添加 `model_format="torch"` 检测
- `pad_token_id=None` 处理

### 2.5 FP8 权重反量化
- CPU numpy 反量化（避免 GPU OOM）
- 支持所有权重类型（Linear + Expert）
- block_size=128 的 scale 展开
- 命名映射：`block_sparse_moe` → `mlp`，`w1/w2/w3` → `up_gate_proj/down_proj`

### 2.6 SM80 兼容性修复
- DeepGEMM 不支持 SM80 时的 BF16 dequant fallback
- `fp8_quant_blockwise()` 的 `using_ue8m0_scale` 参数兼容性修复

---

## 3. 当前状态（2026-04-12）

### 3.1 EP=4 已验证正确的组件
- ✅ Marlin packed weights: byte-identical with vLLM
- ✅ Scales: 100% 一致（max_diff=0）
- ✅ Gate routing: cos=1.0
- ✅ Expert mapping: 正确
- ✅ sorted_token_ids: 正确的 flat index encoding
- ✅ AllReduce: 正确求和
- ✅ 62 层完整加载和推理：端到端运行，无崩溃

### 3.2 EP=4 输出质量问题
- ❌ **Marlin kernel 输出退化**：n≤15 层正确，n≥20 层开始退化
  - n=15: top-1=`'\n\n'` ✓
  - n=20: top-1=`'\xa0'` ✗
  - n=62: top-1=`'斯基'` ✗

### 3.3 根因分析（2026-04-12 深入对比）

#### 3.3.1 FD 的 Marlin kernel vs vLLM 的差异

FD: `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/marlin_template.h` (75KB)
vLLM: `vllm/csrc/moe/marlin_moe_wna16/marlin_template.h` (85KB)

2216 行差异，但**核心计算逻辑（dequant、scale、MMA 指令）完全一致**：

| 差异点 | 内容 | FP8 weight-only 精度影响 |
|--------|------|--------------------------|
| mma() 函数 | FD 不支持 FP8 native MMA | **无影响** — SM80 上都用 FP16 MMA |
| dequant FP8 | byte-identical | **无影响** |
| scale 函数 | 逻辑一致 | **无影响** |
| 调度策略 | FD 简单 stripe vs vLLM DP+SK | **潜在影响** — FP32 累积顺序不同 |
| `should_load_a`/`pipe_a` | FD 独有，vLLM 没有 | **潜在影响** — pipeline 管理不同 |
| `is_ep` 参数 | FD 有，vLLM 已移除 | **无影响** — 仅控制 valid block 过滤 |
| kernel dispatch | FD 手写宏 vs vLLM 自动生成 | **潜在影响** — 可能选到不同配置 |
| `stages` 参数 | FD 硬编码 4，vLLM SM75=2 | **无影响** — SM80 都是 4 |

**结论：** 核心计算 bit-identical，差异主要在调度和类型系统。调度差异在 EP=4 小 batch 下影响很小，但理论上 FP32 累积顺序不同可能导致微小误差。

#### 3.3.2 当前 SM80 workaround（float32 反量化 + 朴素 GEMM）

在 `minimax_m2_5.py` 的 `_load_fp8_marlin_layer` 中：
- FP8 权重 CPU numpy 反量化到 **float32**（因为当时以为 bf16 坏了）
- 保存到 `_sm80_gate_bf16`/`_sm80_up_bf16`/`_sm80_down_bf16`（实际是 float32）

在 `fused_moe_marlin_backend.py` 的 `_apply_ep_sm80_bf16` 中：
- 每个 expert 用 float32 权重 + float32 GEMM
- n=2, n=10 输出正确 ✅
- n=20+ OOM（float32 权重占 4x 显存）❌

#### 3.3.3 关键发现：bfloat16 cast 实际是正常的

2026-04-12 验证：
```
paddle.to_tensor(numpy_f32, dtype="bfloat16").numpy()  → 返回 uint16（显示乱码）
paddle.to_tensor(numpy_f32, dtype="bfloat16").cast("float32").numpy()  → 值正确
paddle.matmul(x_bf16, w_bf16)  → BF16 GEMM 正常工作，error=9.7e-4
```

**之前认为"bfloat16 cast 坏了"是误判。** 实际是 `.numpy()` 对 bf16 返回 uint16 原始值，`cast("float32")` 是正确的。

#### 3.3.4 之前 bf16 反量化方案出乱码的原因

当时（04-11）的代码和现在的差距：
1. **Scale 处理顺序还没修复** — 这是最关键的
2. FP8 反量化逻辑还没对齐 vLLM
3. `.numpy()` 对 bf16 的 uint16 输出导致误判

**现在 scale 已修复，bf16 反量化方案应该能正确工作。**

### 3.4 下一步方案

#### 方案 A：bf16 反量化 + cuBLAS GEMM（推荐，改动小）
- 将 float32 workaround 改为 **bfloat16**
- 显存减半：float32 → bf16 = 4x → 2x
- 4 卡 EP=4 全量 bf16 权重：~3.6 GB/卡，完全可以跑
- 不需要逐层反量化
- 性能：cuBLAS BF16 GEMM 比 Marlin 慢，但足够用
- 改动：修改 `minimax_m2_5.py` 和 `fused_moe_marlin_backend.py` 两三个地方

#### 方案 B：同步 vLLM Marlin kernel（长期方案）
- 替换 FD 的 `marlin_template.h` 等 6+ 个文件
- 工作量大（2216 行 diff + 300KB 自动生成 dispatch 表）
- 需要创建 PaddlePaddle 适配的 `scalar_type.hpp`
- 编译调试周期长
- 优点：性能最优，不占额外显存

#### 方案 C：float16 反量化（备选）
- 和方案 A 类似，但用 float16 代替 bfloat16
- FP16 GEMM error=1.06e-4（比 BF16 更精确）
- 但 FP16 表示范围更小，对大权重值可能溢出

### 3.5 vLLM 成功运行 MiniMax-M2.5
vLLM 已成功在 8 卡 A100 (SM80) 上以 EP=8 模式运行 MiniMax-M2.5 FP8 模型，输出正确的中文。

```bash
# vLLM 启动命令
SAFETENSORS_FAST_GPU=1 vllm serve \
    /data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5 --trust-remote-code \
    --enable_expert_parallel --tensor-parallel-size 8 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

---

## 4. 关键文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `fastdeploy/model_executor/models/minimax_m2_5.py` | **新建** | MiniMax-M2.5 模型实现，含 SM80 float32 workaround（待改为 bf16） |
| `fastdeploy/config.py` | **修改** | partial_rotary_factor, num_local_experts, model_format 检测 |
| `fastdeploy/engine/engine.py` | **修改** | pad_token_id=None 处理 |
| `fastdeploy/input/base_processor.py` | **修改** | get_pad_id() 处理 pad_token_id=None |
| `fastdeploy/model_executor/layers/quantization/block_wise_fp8.py` | **修改** | SM80 BF16 dequant fallback |
| `fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py` | **修改** | Scale 修复，SM80 float32 workaround，debug dump 代码（待清理） |
| `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/marlin_template.h` | **待更新** | 需同步到 vLLM 版本（长期） |
| `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/core/scalar_type.hpp` | **已创建** | vLLM ScalarType 适配版（为长期方案准备） |
| `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/marlin_mma.h` | **已复制** | vLLM MMA 指令封装（为长期方案准备） |

---

## 5. 运行命令

### 5.1 FD EP=4 Marlin FP8 推理
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate paddle
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices=0,1,2,3 /tmp/ep4_bisect.py --n-layers 62
```

### 5.2 FD EP=4 SM80 workaround 推理（当前）
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate paddle
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices=1,3,4,5 /tmp/ep4_bisect.py --n-layers 10
```

### 5.3 vLLM 8 卡部署（已验证成功）
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vllm17
SAFETENSORS_FAST_GPU=1 vllm serve \
    /data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5 --trust-remote-code \
    --enable_expert_parallel --tensor-parallel-size 8
```

---

## 6. 环境信息

- **GPU**: 8 × A100 80GB (SM 8.0)
- **FD 环境**: conda activate `paddle` (Python 3.10, PaddlePaddle 3.3.0)
- **vLLM 环境**: conda activate `vllm17` (Python 3.10, PyTorch, vLLM v0.17 dev)
- **模型路径**: `/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5`
- **Checkpoint**: 125 个 safetensors 文件，总计 230 GB (FP8)

---

## 7. WINT4 复现与 EP=4 FP8 的关系

**WINT4 不受 Marlin kernel 精度问题影响。** WINT4 使用 `uint4b8` 量化（`COMMON_GET_IF(kU4B8)` 路径），不是 FP8（`BIGGROUP_GET_IF(kFE4M3fn)` 路径）。WINT4 的 scale 修复已完成，精度对齐正确。

---

## 8. SM80 BF16 反量化修复（2026-04-12）

### 8.1 关键 bug：`paddle.Tensor.numpy()` 对 bfloat16 返回 uint16

**根因：** `_apply_ep_sm80_bf16` 中的 scatter-add 使用 `ffn_out.numpy()` 对 bf16 tensor 操作，PaddlePaddle 返回 `uint16` numpy 数组（而非 `float32`）。`uint16 += uint16` 产生乱码，导致后续层输出完全退化。

**修复：** 在 `.numpy()` 前显式 `.cast("float32")`：
```python
# 修复前：
ffn_out_np = ffn_out.numpy()           # → uint16
weighted_out_np = weighted_out.numpy() # → uint16

# 修复后：
ffn_out_np = ffn_out.cast("float32").numpy()
weighted_out_np = weighted_out.cast("float32").numpy()
```

### 8.2 验证结果

SM80 BF16 反量化 + cuBLAS GEMM，EP=4，4×A100 (GPU 4,5,6,7)：

| n_layers | top-1 | 状态 |
|----------|-------|------|
| 2 | `'\n\n'` (18.14%) | ✅ |
| 20 | `'\n\n'` (5.62%) | ✅ |
| 30 | `'\n\n'` (5.80%) | ✅ |
| 40 | `'\n\n'` (5.23%) | ✅ |
| 41+ | OOM | ⚠️ 显存不足 |

### 8.3 显存限制

当前瓶颈：非 expert 权重（attention QKV, o_proj, norms）在 `load_weights` 时全部反量化到 BF16 并常驻显存。
- 每层 ~0.8 GB（BF16 非 expert + FP8 expert 权重）
- 40 层 ≈ 48 GB → 勉强 80 GB
- 62 层 ≈ 74 GB → OOM

**解决方案：** 非 expert 权重也需要逐层反量化（在 forward 时反量化当前层，用完释放）。

### 8.4 修改的文件

| 文件 | 修改 |
|------|------|
| `fused_moe_marlin_backend.py` | `_apply_ep_sm80_bf16`: numpy uint16 bug 修复；allreduce 改用 float32 |
| `moe.py` | `forward_normal`: 移除 debug 日志 |

---

## 9. SM80 优化：FP8 非 expert + CPU expert（2026-04-12 下午/晚）

### 9.1 非 expert 权重 FP8 保留

- `_dequant_fp8_weights` 添加 `sm80_keep_fp8=True` 参数
- SM80 上非 expert 权重（qkv_proj, o_proj）保持 FP8 格式，不反量化到 BF16
- `BlockWiseFP8LinearMethod.apply()` SM80 路径在 forward 时逐层反量化
- **显存节省：** 每层非 expert 权重从 ~0.5 GB BF16 降到 ~0.05 GB FP8
- **scale 布局修复：** weight_loader 将权重从 `[out,in]` 转置为 `[in,out]`，但 scale_inv 保持 torch 布局。`apply()` 中自动检测并转置 scale。

### 9.2 Expert 权重 CPU 离驻

- SM80 不创建 Marlin packed int32 格式（节省 ~1.6 GB/层）
- Raw FP8 expert 权重存储在 CPU，forward 时 `.cuda()` 拷贝到 GPU
- `_apply_ep_sm80_bf16` 在 dequant 前从 CPU 拷贝到 GPU，计算后释放 GPU 副本

### 9.3 验证结果

| n_layers | top-1 | 状态 | GPU 内存 |
|----------|-------|------|----------|
| 10 | `'\n\n'` (28.64%) | ✅ | 11.4 GB |
| 20 | `'\n\n'` (21.12%) | ✅ | 20.6 GB |
| 40 | `'\n\n'` (1.45%) | ✅ | 38.9 GB |
| 50 | `'\n\n'` (1.80%) | ✅ | 48.0 GB |
| 52 | `'\n\n'` (2.55%) | ✅ | 49.8 GB |
| 54 | `'\n\n'` (6.32%) | ✅ | 51.7 GB |
| 56 | `'\n\n'` (6.60%) | ✅ | 53.6 GB |
| 58 | `'\n\n'` (6.14%) | ✅ | 55.4 GB |
| 60 | `'\n\n'` (7.28%) | ✅ | 57.2 GB |
| 61 | `'\n\n'` (7.05%) | ✅ | 58.1 GB |
| 62 | `':'` (6.92%) | ⚠️ | 59.0 GB |

**结论：** SM80 BF16 dequant workaround 支持 **61 层**正确输出。n=62 可能是 BF16 GEMM 累积误差的最后 1 层边界。

### 9.3.1 非 expert FP8 dequant scale expansion bug 修复（2026-04-12）

**根因：** `block_wise_fp8.py` 的 `apply()` 方法中，PaddlePaddle 的 `paddle.expand` + `paddle.reshape` 对 block-wise scale 的展开产生错误的内存布局。

**Bug 细节：**
- Scale shape: `[n_blocks_out, n_blocks_in]`，需展开到 `[n_blocks_out*BLOCK, n_blocks_in*BLOCK]`
- `paddle.expand` 返回 non-contiguous view `[n_blocks_out, n_blocks_in, 128, 128]`
- 直接 `reshape` 到 `[n_blocks_out*128, n_blocks_in*128]` 时，PaddlePaddle 按 C order 展开，产生错误的 block-interleaved 布局
- 正确做法：先 `transpose([0, 2, 1, 3])` 为 `[n_blocks_out, 128, n_blocks_in, 128]`，再 reshape

**修复：** 在 `block_wise_fp8.py` 的 `apply()` SM80 路径中，`expand` 后添加 `transpose([0, 2, 1, 3])`：
```python
sc_exp = paddle.expand(sc_exp, [scale.shape[0], scale.shape[1], BLOCK, BLOCK])
sc_exp = sc_exp.transpose([0, 2, 1, 3])  # ← 新增
sc_exp = sc_exp.reshape([scale.shape[0] * BLOCK, scale.shape[1] * BLOCK])[:out_d, :in_d]
```

**影响：** 之前 n=54+ 的精度退化并非 BF16 GEMM 累积误差，而是非 expert 权重的 FP8 dequant scale 展开错误。修复后 n=54~61 全部正确。

### 9.4 n=62 边界问题（2026-04-12）

n=61 正确、n=62 错误。logit mean 从 -6.40 跳到 -3.75，不是微小精度误差而是质变。
可能原因：62 层 BF16 GEMM 累积误差 + lm_head 放大效应，或非 MoE 组件（embed_tokens、lm_head）的 BF16 精度。
vLLM SM80 使用相同的 BF16 GEMM 路径，可能 PyTorch cuBLAS 和 PaddlePaddle cuBLAS 的精度行为略有差异。

### 9.5 修改的文件

| 文件 | 修改 |
|------|------|
| `minimax_m2_5.py` | `_dequant_fp8_weights` 添加 `sm80_keep_fp8`；SM80 不创建 Marlin packed；expert FP8 存 CPU |
| `block_wise_fp8.py` | `apply()` SM80 路径添加 scale 布局自动检测和转置 |
| `fused_moe_marlin_backend.py` | `_apply_ep_sm80_bf16`: CPU→GPU 拷贝；清理 debug 代码 |
| `linear.py` | `UnquantizedLinearMethod.apply()` 清理 debug dump 代码 |

---

## 10. 详细失败案例与实验记录（2026-04-12）

### 10.1 实验 1：FD Marlin kernel 在 SM80 上的精度问题

**目标：** 验证 FD 的 Marlin FP8 kernel 在 SM80 (A100) 上是否能正确运行。

**方法：**
```bash
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices=1,3,4,5 /tmp/ep4_bisect.py --n-layers 2
```

**结果：**
| n_layers | top-1 token | 预期 | 状态 |
|----------|------------|------|------|
| 2 | `'水分'` (2.58%) | `'\n\n'` | ❌ |

**结论：** FD 的 Marlin FP8 kernel 在 SM80 上从第 2 层就出错，完全不可用。

---

### 10.2 实验 2：禁用 `should_load_a` 优化

**假设：** FD 独有的 `should_load_a`/`pipe_a` A 矩阵缓存优化导致某些 tile 读取 stale 数据。

**修改：** `marlin_template.h` 行 1963-1968：
```cpp
// 原代码：
if (slice_col == 0 || old_slice_row ||
    prob_k > thread_k_blocks * 16 * stages * max_num_stage_groups) {
  should_load_a = true;
} else {
  should_load_a = false;  // ← 强制为 true
}

// 修改为：
should_load_a = true;
```

同时行 923-925 强制 `max_num_stage_groups = 1`：
```cpp
// 原代码：
int max_num_stage_groups =
    ((sh_a_max_row - moe_block_size) / moe_block_size + 1) / stages;
max_num_stage_groups = max(max_num_stage_groups, 1);

// 修改为：
int max_num_stage_groups = 1;
```

**编译：** `cd custom_ops && python setup_ops.py build`（成功）

**结果：** n=2 仍然是 `'水分'`，**完全无效**。

**回退：** 所有 `marlin_template.h` 修改已回退。

---

### 10.3 实验 3：修复 `sh_new` 的 `+moe_block_size` padding

**假设：** FD 的 `shm_size_used` 计算有额外的 `+ moe_block_size`，压缩了 `sh_a` 空间。

**修改：** `marlin_template.h` 中 `sh_new` 调用处，移除 `+ moe_block_size`。

**结果：** n=2 仍然是 `'水分'`，**完全无效**。

**回退：** 修改已回退。

---

### 10.4 实验 4：BF16 numpy 反量化 + GPU BF16 GEMM（scatter-add uint16 bug）

**背景：** vLLM SM80 上不支持 FP8 Marlin（`TORCH_CHECK(major >= 89)`），使用 W8A16（BF16 activation）。FD 的 SM80 BF16 dequant workaround 理论上等价。

**实现：** 在 `fused_moe_marlin_backend.py` 的 `_apply_ep_sm80_bf16` 中：
- 每个 expert 的 FP8 权重用 numpy 反量化到 BF16
- 使用 `paddle.nn.functional.linear` 做 BF16 GEMM
- Scatter-add 汇总所有 expert 输出

**问题：** 首次运行 n=2 输出完全退化（不是 `'\n\n'`）。

**根因：** `ffn_out.numpy()` 对 bfloat16 tensor 返回 `uint16` numpy 数组，而非 `float32`。`uint16 += uint16` 做的是整数加法，产生乱码。

```
# 验证：
paddle.to_tensor(numpy_f32, dtype="bfloat16").numpy()  → uint16 [16256, 16128, ...]
paddle.to_tensor(numpy_f32, dtype="bfloat16").cast("float32").numpy()  → float32 [1.0, 0.5, ...]
```

**修复：** 在 `.numpy()` 前显式 `.cast("float32")`：
```python
ffn_out_np = ffn_out.cast("float32").numpy()
weighted_out_np = weighted_out.cast("float32").numpy()
```

**验证：** n=2 ✅, n=20 ✅, n=30 ✅, n=40 ✅, n=41+ OOM。

---

### 10.5 实验 5：显存优化 — 非 expert FP8 保留 + CPU expert 离驻

**问题：** n=41+ OOM，因为 float32 expert 权重占 ~4x 显存。

**方案：**
1. 非 expert 权重（qkv_proj, o_proj）保持 FP8 格式，不反量化到 BF16
2. Expert FP8 权重存储在 CPU，forward 时 `.cuda()` 拷贝到 GPU
3. `BlockWiseFP8LinearMethod.apply()` SM80 路径在 forward 时逐层反量化

**显存节省计算：**
- 非 expert 权重：~0.5 GB BF16 → ~0.05 GB FP8（每层）
- Expert 权重：~3.2 GB BF16 → 0 GB GPU（CPU 离驻，按需拷贝）
- 总计：62 层 ~59 GB GPU（可接受）

**结果：**
| n_layers | top-1 | 状态 | GPU 内存 |
|----------|-------|------|----------|
| 10 | `'\n\n'` (28.64%) | ✅ | 11.4 GB |
| 20 | `'\n\n'` (21.12%) | ✅ | 20.6 GB |
| 40 | `'\n\n'` (1.45%) | ✅ | 38.9 GB |
| 50 | `'\n\n'` (1.80%) | ✅ | 48.0 GB |
| 52 | `'\n\n'` (2.55%) | ✅ | 49.8 GB |
| 54 | ❌ 退化 | ❌ | 51.7 GB |

**新问题：** n=54+ 开始退化，但不是 OOM！这说明退化原因是精度问题，不是显存问题。

---

### 10.6 实验 6：定位 scale expansion bug（关键突破）

**假设：** 非 expert FP8 dequant 的 scale 展开有问题。

**验证方法：** 对比 FD 和 vLLM 的 FP8 dequant 输出：
```python
# 测试 block-wise scale expansion
scale = paddle.randn([24, 24])  # [n_blocks_out, n_blocks_in]
BLOCK = 128

# FD 原始实现（错误）：
sc_exp = scale.unsqueeze(2).unsqueeze(3)
sc_exp = paddle.expand(sc_exp, [24, 24, 128, 128])
result_wrong = sc_exp.reshape([3072, 3072])

# 正确实现：
sc_exp = scale.unsqueeze(2).unsqueeze(3)
sc_exp = paddle.expand(sc_exp, [24, 24, 128, 128])
sc_exp = sc_exp.transpose([0, 2, 1, 3])
result_correct = sc_exp.reshape([3072, 3072])
```

**关键发现：**
- `result_wrong[127, 127]` = 23（应为 0，因为 block [0,0] 的 scale）
- `result_correct[127, 127]` = 0（正确）
- `paddle.expand` 返回 non-contiguous view，`reshape` 按 C order 展开，不是 block-interleaved

**根因：** PaddlePaddle 的 `paddle.expand` + `paddle.reshape` 对 4D tensor `[n_blocks_out, n_blocks_in, BLOCK, BLOCK]` 的展开不按 block-interleaved 顺序。需要先 `transpose([0, 2, 1, 3])` 将维度排列为 `[n_blocks_out, BLOCK, n_blocks_in, BLOCK]`，再 reshape 才能得到正确的 block-wise scale 布局。

**也测试了 `paddle.tile`：** 有同样的问题。

**修复：** `block_wise_fp8.py` 行 375 后添加 `sc_exp = sc_exp.transpose([0, 2, 1, 3])`。

**验证：** dequant float32 error = 0.00e+00（完全一致）。

**结果：** n=54~61 全部正确（之前 n=54+ 全部错误）。

---

### 10.7 实验 7：n=62 边界问题

**现象：** n=61 正确（`'\n\n'`, logit mean = -6.40），n=62 错误（`':'`, logit mean = -3.75）。

**关键观察：**
- logit mean 从 -6.40 跳到 -3.75，不是微小精度误差（~1e-3），而是质变（~2.65）
- n=60, 61 的 logit mean 都是 -6.40，非常稳定
- n=62 的 top-1 是 `':'`（6.92%），不是随机噪声

**可能原因：**
1. 62 层 BF16 GEMM 累积误差 + lm_head 放大效应
2. embed_tokens 或 lm_head 的 FP8 dequant 也有同样的 scale expansion bug（已修复，但可能还有其他路径）
3. PaddlePaddle cuBLAS 和 PyTorch cuBLAS 的 BF16 GEMM 精度行为略有差异
4. 第 62 层（最后一层）的权重可能有特殊的数值范围

**待调查：** 检查 lm_head 和 embed_tokens 是否也使用 `BlockWiseFP8LinearMethod.apply()` 的 SM80 路径。

---

### 10.8 实验 8（失败）：SM80 Marlin packed 权重替代 CPU FP8

**假设：** 也许 SM80 上可以用 Marlin packed weights（而非 CPU FP8 offload），只是之前的测试有问题。

**修改：** `minimax_m2_5.py` 中 SM80 路径改为创建 Marlin packed weights。

**结果：** n=2 = `'水分'`，和实验 1 一致。**Marlin kernel 在 SM80 上确实不工作。**

**回退：** 修改已回退。

---

### 10.9 失败模式总结

| 实验 | 假设 | 修改 | 结果 | 结论 |
|------|------|------|------|------|
| 1 | Marlin SM80 可用 | 无 | n=2 ❌ | Marlin 在 SM80 不可用 |
| 2 | `should_load_a` 是根因 | 禁用 `should_load_a` | n=2 ❌ | 无效 |
| 3 | shm padding 是根因 | 移除 `+moe_block_size` | n=2 ❌ | 无效 |
| 4 | BF16 dequant workaround | numpy dequant + cuBLAS | n=2~40 ✅, 41+ OOM | workaround 可行但显存不足 |
| 5 | 显存优化 | FP8 非 expert + CPU expert | n=2~52 ✅, 54+ ❌ | 显存够了，但精度退化 |
| 6 | Scale expansion bug | transpose([0,2,1,3]) | n=2~61 ✅, 62 ❌ | **关键突破** |
| 7 | n=62 边界 | 待调查 | n=62 ❌ | 可能是 BF16 累积误差 |
| 8 | SM80 Marlin 可用 | 恢复 Marlin packed | n=2 ❌ | 确认不可用 |

---

### 10.10 Marlin template.h 已回退的修改

以下修改已尝试并**全部回退**（文件恢复到 git 版本）：
1. `should_load_a = true` 强制设置（行 1963-1968）
2. `max_num_stage_groups = 1` 强制设置（行 923-925）
3. `sh_new` 移除 `+ moe_block_size` padding
4. SM80 Marlin packed weights 路径（`minimax_m2_5.py`）
