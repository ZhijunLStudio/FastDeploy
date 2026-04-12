# MiniMax-M2.5 FP8 EP=4 修复进展

## 日期：2026-04-12

### 重大突破：非 expert FP8 dequant scale expansion bug 修复

**根因：** `block_wise_fp8.py` 的 `apply()` SM80 路径中，PaddlePaddle 的 `paddle.expand` + `paddle.reshape` 对 block-wise scale 的展开产生错误的内存布局。

**Bug 详情：**
- Scale shape `[n_blocks_out, n_blocks_in]`，需展开到 `[n_blocks_out*BLOCK, n_blocks_in*BLOCK]`
- `paddle.expand` 返回 non-contiguous view，直接 `reshape` 按 C order 展开，不是 block-interleaved
- 正确做法：`expand` 后添加 `transpose([0, 2, 1, 3])`，再 reshape

**修复：** 在 `block_wise_fp8.py` 行 375 后添加 `sc_exp = sc_exp.transpose([0, 2, 1, 3])`

**验证结果：**

SM80 BF16 dequant workaround，EP=4，4×A100：

| n_layers | top-1 | 状态 | GPU 内存 |
|----------|-------|------|----------|
| 10 | `'\n\n'` (28.64%) | ✅ | 11.4 GB |
| 20 | `'\n\n'` (21.12%) | ✅ | 20.6 GB |
| 40 | `'\n\n'` (1.45%) | ✅ | 38.9 GB |
| 50 | `'\n\n'` (1.80%) | ✅ | 48.0 GB |
| 52 | `'\n\n'` (6.07%) | ✅ | 49.8 GB |
| 54 | `'\n\n'` (6.32%) | ✅ | 51.7 GB |
| 56 | `'\n\n'` (6.60%) | ✅ | 53.6 GB |
| 58 | `'\n\n'` (6.14%) | ✅ | 55.4 GB |
| 60 | `'\n\n'` (7.28%) | ✅ | 57.2 GB |
| 61 | `'\n\n'` (7.05%) | ✅ | 58.1 GB |
| 62 | `':'` (6.92%) | ⚠️ | 59.0 GB |

**结论：** n=54~61 全部正确（之前 n=54+ 全部错误）。n=62 仍然出错，可能是 BF16 GEMM 最后 1 层累积误差。

---

## 失败实验详细记录

### 实验 1：FD Marlin kernel SM80 路径

**目标：** 验证 FD 的 Marlin FP8 kernel 在 SM80 (A100) 上是否能正确运行。

**方法：** 直接使用 FD Marlin kernel，不做任何修改。

**结果：**
| n_layers | top-1 token | 预期 | 状态 |
|----------|------------|------|------|
| 2 | `'水分'` (2.58%) | `'\n\n'` | ❌ |

**结论：** FD 的 Marlin FP8 kernel 在 SM80 上从第 2 层就出错。vLLM 也不支持 SM80 FP8 Marlin（`TORCH_CHECK(major >= 89)` for W8A8），这是硬件限制，不是 bug。

### 实验 2：禁用 `should_load_a` 优化

**假设：** FD 独有的 `should_load_a`/`pipe_a` A 矩阵缓存优化（vLLM 已移除）导致某些 tile 读取 stale 激活数据。

**修改：** `marlin_template.h`
1. 行 1963-1968：强制 `should_load_a = true`
2. 行 923-925：强制 `max_num_stage_groups = 1`

**编译：** `cd custom_ops && python setup_ops.py build`（成功）

**结果：** n=2 仍然是 `'水分'`，**完全无效**。

**结论：** `should_load_a` 不是 SM80 退化的原因。Marlin kernel 在 SM80 上有更根本的问题（FP8 activation 路径选择错误）。

**回退：** 所有 `marlin_template.h` 修改已回退。

### 实验 3：修复 `sh_new` 的 `+moe_block_size` padding

**假设：** FD 的 `shm_size_used` 计算有额外的 `+ moe_block_size`，压缩了 `sh_a` 缓存空间。

**修改：** `marlin_template.h` 中 `sh_new` 调用处，移除 `+ moe_block_size`。

**结果：** n=2 仍然是 `'水分'`，**完全无效**。

**回退：** 修改已回退。

### 实验 4：BF16 numpy 反量化 + cuBLAS GEMM（scatter-add uint16 bug）

**背景：** vLLM SM80 使用 Marlin W8A16（BF16 activation）。FD 的 SM80 BF16 dequant workaround 理论上等价。

**实现：**
- `fused_moe_marlin_backend.py` 新增 `_apply_ep_sm80_bf16` 方法
- 每个 expert FP8 权重用 numpy 反量化到 BF16
- `paddle.nn.functional.linear` 做 BF16 GEMM
- Scatter-add 汇总所有 expert 输出

**问题：** 首次运行 n=2 输出完全退化。

**根因：** `ffn_out.numpy()` 对 bfloat16 tensor 返回 `uint16` numpy 数组（而非 `float32`）。`uint16 += uint16` 做的是整数加法，产生乱码。

```python
# 验证：
paddle.to_tensor(numpy_f32, dtype="bfloat16").numpy()           # → uint16 [16256, 16128, ...]
paddle.to_tensor(numpy_f32, dtype="bfloat16").cast("float32").numpy()  # → float32 [1.0, 0.5, ...]
```

**修复：** 在 `.numpy()` 前显式 `.cast("float32")`：
```python
ffn_out_np = ffn_out.cast("float32").numpy()
weighted_out_np = weighted_out.cast("float32").numpy()
```

**验证：** n=2 ✅, n=20 ✅, n=30 ✅, n=40 ✅, n=41+ OOM（float32 权重占 4x 显存）。

### 实验 5：显存优化 — 非 expert FP8 保留 + CPU expert 离驻

**问题：** float32 expert 权重占 ~4x 显存，n=41+ OOM。

**方案：**
1. 非 expert 权重保持 FP8 格式（`sm80_keep_fp8=True`）
2. Expert FP8 权重存储在 CPU，forward 时 `.cuda()` 拷贝到 GPU
3. `BlockWiseFP8LinearMethod.apply()` SM80 路径在 forward 时逐层反量化

**结果：**
| n_layers | top-1 | 状态 | GPU 内存 |
|----------|-------|------|----------|
| 10 | `'\n\n'` (28.64%) | ✅ | 11.4 GB |
| 20 | `'\n\n'` (21.12%) | ✅ | 20.6 GB |
| 40 | `'\n\n'` (1.45%) | ✅ | 38.9 GB |
| 50 | `'\n\n'` (1.80%) | ✅ | 48.0 GB |
| 52 | `'\n\n'` (2.55%) | ✅ | 49.8 GB |
| 54 | ❌ 退化 | ❌ | 51.7 GB |

**新问题：** n=54+ 开始退化，但不是 OOM。说明退化原因是精度问题。

### 实验 6：定位 scale expansion bug（关键突破）

**假设：** 非 expert FP8 dequant 的 scale 展开有 bug。

**验证：** 对比 FD 和 vLLM 的 scale expansion 输出：
```python
scale = paddle.randn([24, 24])
BLOCK = 128

# 错误实现：
sc_exp = paddle.expand(scale.unsqueeze(2).unsqueeze(3), [24, 24, 128, 128])
result_wrong = sc_exp.reshape([3072, 3072])
print(result_wrong[127, 127])  # → 23（应为 0）

# 正确实现：
sc_exp = paddle.expand(scale.unsqueeze(2).unsqueeze(3), [24, 24, 128, 128])
sc_exp = sc_exp.transpose([0, 2, 1, 3])
result_correct = sc_exp.reshape([3072, 3072])
print(result_correct[127, 127])  # → 0（正确）
```

**根因：** `paddle.expand` 返回 non-contiguous view。4D tensor `[n_blocks_out, n_blocks_in, BLOCK, BLOCK]` 直接 `reshape` 按 C order 展开，不是 block-interleaved 顺序。`paddle.tile` 也有同样的问题。

**修复：** `block_wise_fp8.py` 添加 `sc_exp = sc_exp.transpose([0, 2, 1, 3])`

**结果：** n=54~61 全部正确（之前 n=54+ 全部错误）。

### 实验 7：n=62 边界问题（待解决）

**现象：** n=61 正确（`'\n\n'`, logit mean = -6.40），n=62 错误（`':'`, logit mean = -3.75）。

**关键观察：**
- logit mean 从 -6.40 跳到 -3.75（差 2.65），不是微小精度误差（~1e-3），而是质变
- n=60, 61 的 logit mean 都是 -6.40，非常稳定
- n=62 的 top-1 是 `':'`（6.92%），不是随机噪声

**可能原因：**
1. 62 层 BF16 GEMM 累积误差 + lm_head 放大效应
2. embed_tokens 或 lm_head 的 FP8 dequant 可能还有其他问题
3. PaddlePaddle cuBLAS 和 PyTorch cuBLAS 的 BF16 GEMM 精度行为差异
4. 第 62 层（最后一层）的权重可能有特殊的数值范围

**待调查：** 检查 lm_head 和 embed_tokens 是否也使用 `BlockWiseFP8LinearMethod.apply()` 的 SM80 路径。

### 实验 8（失败）：SM80 恢复 Marlin packed 权重

**假设：** 也许 SM80 上可以用 Marlin packed weights。

**修改：** `minimax_m2_5.py` SM80 路径改为创建 Marlin packed weights。

**结果：** n=2 = `'水分'`，和实验 1 一致。**确认 Marlin kernel 在 SM80 上不可用。**

**回退：** 修改已回退。

---

## 失败模式总结

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

## Marlin template.h 已回退的修改

以下修改已尝试并**全部回退**（文件恢复到 git 版本）：
1. `should_load_a = true` 强制设置（行 1963-1968）
2. `max_num_stage_groups = 1` 强制设置（行 923-925）
3. `sh_new` 移除 `+ moe_block_size` padding

---

## 修改的文件（最终状态）

| 文件 | 修改 |
|------|------|
| `block_wise_fp8.py` | `apply()` SM80 路径：`expand` 后添加 `transpose([0, 2, 1, 3])` 修复 scale 展开布局 + scale 转置自动检测 |
| `minimax_m2_5.py` | `_dequant_fp8_weights` 添加 `sm80_keep_fp8`；SM80 不创建 Marlin packed；expert FP8 存 CPU |
| `fused_moe_marlin_backend.py` | `_apply_ep_sm80_bf16`: CPU→GPU 拷贝；numpy uint16 bug 修复；清理 debug 代码 |
| `linear.py` | `UnquantizedLinearMethod.apply()` 清理 debug dump 代码 |
| `marlin_template.h` | **未修改**（所有尝试已回退） |

## 下一步

- n=62 仍然出错（logit mean 从 -6.40 跳到 -3.75），可能是 BF16 GEMM 最后 1 层累积误差
- 可能需要检查 lm_head 的 BF16 精度，或 embed_tokens 的输出
- 或者尝试 FP32 lm_head computation
