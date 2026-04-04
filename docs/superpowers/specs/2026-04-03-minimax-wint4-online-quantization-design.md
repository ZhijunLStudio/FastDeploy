# MiniMax-M2.5 WINT4 在线量化设计

## 背景

MiniMax-M2.5 (227B, 62层, 256 experts) checkpoint 是 FP8 block-wise 量化。在 SM80 (A800) 上 FP8 不支持，FD 做 FP8→BF16 反量化后显存翻倍，TP=4 约 108 GB/卡超出 A800 80GB 限制。

**目标**：FP8→BF16 反量化后，对所有 Linear 层做 BF16→WINT4 在线量化，将显存降到 TP=4 ~56 GB/卡。

## 方案概述

在 `minimax_m2_5.py` 的 `load_weights` 中，FP8→BF16 反量化后立即调用 `paddle.nn.quant.weight_quantize(algo="weight_only_int4")` 做 int4 量化。不修改 FD 的 `quant_config` 框架。

## 量化范围

| 层 | 量化方式 | 推理 kernel |
|----|---------|------------|
| q_proj, k_proj, v_proj, o_proj | WINT4 | `weight_only_linear` (PaddlePaddle 原生) |
| MoE experts (w1/w2) | WINT4 Marlin | `MoeWna16MarlinGemmApi` |
| MoE gate | 保持 BF16 | — |
| lm_head | 保持 BF16 | — |
| Embedding | 保持 BF16 | — |

## 显存估算 (TP=4)

| 组件 | BF16 | WINT4 |
|------|------|-------|
| Attention q/k/v/o (62层) | ~4.6 GB | ~2.3 GB |
| MoE experts (62×256) | ~200 GB | ~100 GB |
| 其他 (embed, gate, norm) | ~1.4 GB | ~1.4 GB |
| **总计 (全量)** | **~206 GB** | **~104 GB** |
| **每卡 TP=4** | **~108 GB ❌** | **~56 GB ✅** |

## 实现步骤

### Step 1: Linear 层 WINT4 (在 FP8 反量化循环中)

在 `load_weights` 的 FP8 反量化循环 (line 635-729) 中，对 Linear 层权重：

1. FP8→BF16 反量化 (已有)
2. 如果是 torch 格式，先 transpose
3. 调用 `weight_quantize(wt_dq, algo="weight_only_int4")` 得到 int4 packed weight (int8 存储) + scale
4. 释放 BF16 权重
5. 将 int4 weight 设置到 param，创建 `weight_scale` param

判断是否是 Linear 层：通过 `params_dict` 中的 sublayer 类型 (`ColumnParallelLinear`, `RowParallelLinear`)。

### Step 2: MoE Expert 层 WINT4 Marlin (在 load_weights 末尾)

所有权重加载完成后，对每个 MoE 层：

1. 提取 `up_gate_proj_weight` (shape: `[256, 3072, 3072]`) 和 `down_proj_weight` (shape: `[256, 1536, 3072]`)
2. 逐 expert 做 int4 量化 + Marlin repacking (复用 `MarlinWeightOnlyMoEMethod.process_loaded_weights` 的逻辑)
3. 替换 MoE 层的 `quant_method` 为 `MarlinWeightOnlyMoEMethod`
4. 释放 BF16 expert weights

### Step 3: 修改 block_wise_fp8.py 的 apply 方法

SM80 上 WINT4 量化的 Linear 层使用 `weight_only_linear` 推理：

```python
if get_sm_version() < 90:
    if getattr(layer, '_wint4_quantized', False):
        linear_out = weight_only_linear(x, weight=layer.weight,
                                         weight_scale=layer.weight_scale,
                                         weight_dtype="int4")
    else:
        linear_out = F.linear(x.cast("bfloat16"), layer.weight)
```

### Step 4: MoE 推理使用 Marlin kernel

MoE 层的 `quant_method` 替换为 `MarlinWeightOnlyMoEMethod` 后，`FusedMoE.forward` 会自动调用 `MarlinWeightOnlyMoEMethod.apply`，使用 `MoeWna16MarlinGemmApi` 做 int4 MoE GEMM。

## 关键代码位置

- `minimax_m2_5.py:635-729` — FP8 反量化循环 (Linear 层 WINT4 插入点)
- `minimax_m2_5.py:731-749` — torch 格式 weight 转置 (WINT4 前需先转置)
- `minimax_m2_5.py:load_weights 末尾` — MoE Marlin WINT4 后处理
- `block_wise_fp8.py:337-350` — SM80 apply 方法 (需添加 WINT4 分支)
- `fused_moe_marlin_backend.py:172-237` — Marlin MoE WINT4 量化逻辑 (复用)

## 风险

| 风险 | 缓解 |
|------|------|
| WINT4 精度损失 | 先用 5 层模型验证 |
| Marlin kernel SM80 兼容性 | 已检查无 SM 版本限制 |
| weight_quantize shape 限制 | int4 要求 out_dim 能被 2 整除 |
