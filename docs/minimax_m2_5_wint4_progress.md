# MiniMax-M2.5 WINT4 推理复现报告

## 1. 项目目标

在 FastDeploy (FD) 框架中复现 MiniMax-M2.5 模型（`MiniMaxM2ForCausalLM`），实现权重加载、前向推理、token 生成。核心目标是在 **4×A800 (SM80)** 上以 **WINT4 量化** 方式运行全量 62 层模型。

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

**环境信息**：
- **GPU**: 8 × A100 80GB (SM 8.0)
- **FD 环境**: conda activate `paddle` (Python 3.10, PaddlePaddle 3.3.0)
- **vLLM 环境**: conda activate `vllm17` (Python 3.10, PyTorch, vLLM v0.17 dev)
- **模型路径**: `/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5`
- **Checkpoint**: 125 个 safetensors 文件，总计 230 GB (FP8)
- **代码仓库**: `github.com:ZhijunLStudio/FastDeploy.git` 分支 `feat/minimax-m2.5-wint4`

---

## 2. 已完成的工作

### 2.1 模型架构实现
- 完整实现了 `MiniMaxM2_5DecoderLayer` (GQA + QK-Norm + Partial RoPE + 256 Expert MoE)
- 支持 FP8 权重反量化 (block-wise dequant)
- 支持 `e_score_correction_bias` 路由校正

### 2.2 内存优化
- **修复 `FusedMoE` 重复创建 Bug**：初始化时创建了两次 FusedMoE，导致显存翻倍。修复后每层节省约 6.75GB。
- **FP8 反量化**：在 CPU 上进行 FP8 -> BF16 反量化，避免 GPU OOM。

### 2.3 WINT4 实现 (进行中)
实现了自动化的 WINT4 量化流程：
- **Linear 层** (`qkv_proj`, `o_proj`): 使用 `paddle.nn.quant.weight_quantize` 进行 int4 量化。
- **MoE 层** (`experts`): 逐层进行 int4 量化，量化后删除 BF16 权重并释放显存，替换 `quant_method` 为 `CutlassWeightOnlyMoEMethod(WINT4Config(...))` 以支持 SM80 MoE Kernel。

### 2.4 当前状态
- **FP8 模式**：在单卡 A800 上成功运行 8 层模型，生成正常 token（57GB 显存）。
- **WINT4 模式**：
  - 8 层模型显存仅需 **16GB** (远优于 FP8 的 57GB)。
  - **但是输出全为 0**。
  - 代码已推送到远程分支。

---

## 3. 核心 Blocker：WINT4 输出全 0

### 3.1 问题现象
开启 `FD_WINT4_QUANTIZE=1` 后，模型加载和量化过程正常（无报错），但推理生成的 token 全部为 0。

### 3.2 根因分析
问题出在 **Linear 层**的 `weight_only_linear` 调用上（MoE 层也可能存在类似问题，但 Linear 层最先暴露）。

#### 形状不匹配
PaddlePaddle 的 `weight_quantize` 和 `weight_only_linear` API 存在形状定义的混淆：

1. **标准 Linear 权重布局**：`[out_features, in_features]` (N, K)
2. **`weight_quantize(algo="weight_only_int4")` 行为**：
   - 输入：`[out_features, in_features]`
   - 输出 Packed Weight：`[out_features / 2, in_features]` (int8)
   - 输出 Scale：`[out_features]`
   - *(注：部分源码和测试用例暗示它期望输入是转置的 `[in, out]`，但 PaddlePaddle 文档和标准 Linear 布局是 `[out, in]`)*

3. **`weight_only_linear` 行为**：
   - 根据 `weight_only_linear.cc` 推断，它期望 Weight 形状为 `[out_features/2, in_features]`。
   - 如果传入的形状不匹配，kernel 可能会越界读取或计算错误，导致输出 0。

#### 错误的实现
在 `minimax_m2_5.py` 的 `_wint4_quantize_linear_layers` 中，当前的实现方式如下：

```python
# 当前代码 (有问题)
w = sublayer.weight
if self.fd_config.model_config.model_format == "torch":
    w = w.transpose([1, 0]) # [in, out] -> [out, in]
wt_int4, wt_scale = _wq(w, algo="weight_only_int4")

sublayer.weight = sublayer.create_parameter(
    shape=wt_int4.shape, dtype="int8", # 直接使用了量化后的形状
    ...
)
```

这里我们直接把 `wt_int4` 赋值给 `layer.weight`。虽然形状本身可能对上了，但问题是 PaddlePaddle 的 `weight_only_linear` 内部实现可能对 Layout 有特殊要求（例如要求 Column-Major 或特殊的 Packing 方式），而 `weight_quantize` 输出的 Packing 方式在 SM80 上可能与 Kernel 期望的不一致。

**更重要的是**，我们之前尝试直接调用 `weight_only_linear` 失败了，因为在 SM80 (A800) 上，PaddlePaddle 的 `paddle.nn.quant` 模块下的 CUDA Kernel 可能并未完整支持 int4 这种非标准布局，或者 `arch=80` 参数并没有正确生效。

### 3.3 下一步排查计划

#### 方向 A：确认 `weight_only_linear` 是否真的支持 SM80 int4
- 编写一个最小化的单元测试：创建一个 `[1024, 1024]` 的随机 BF16 权重，量化为 int4，然后用 `weight_only_linear` 计算，对比 BF16 的 `matmul` 结果。
- 如果单元测试也输出全 0 或报错，说明 `paddle.nn.quant.weight_only_linear` 在 SM80 上不支持 int4（或者我们的调用姿势完全错误）。

#### 方向 B：检查 MoE Kernel (`moe_expert_ffn`)
- MoE 层使用的是 `CutlassWeightOnlyMoEMethod`，它调用的是 `moe_expert_ffn` 这个自定义算子。
- 这个算子在 FastDeploy 中有 XPU/CUDA 实现。需要确认 CUDA 实现是否真的支持 `weight_only_int4`。
- 如果 `moe_expert_ffn` 支持 int4，那么我们需要看看它的输入要求是什么，是否和 `weight_quantize` 的输出一致。

#### 方向 C：手动实现 CPU int4 GEMM (备选)
- 如果 PaddlePaddle 的 CUDA Kernel 确实不支持 SM80 int4，我们可以退回到 CPU 计算（虽然慢，但能跑通全量模型验证效果）。
- 或者手动写一个 CUDA Kernel 来解包 int4 权重并执行 GEMM。

---

## 4. 关键文件清单

| 文件路径 | 说明 |
|----------|------|
| `FastDeploy/fastdeploy/model_executor/models/minimax_m2_5.py` | 模型定义 + WINT4 量化逻辑 (核心文件) |
| `FastDeploy/fastdeploy/model_executor/layers/quantization/weight_only.py` | Linear WINT4 实现 |
| `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py` | MoE WINT4 实现 |
| `my-tools/run_fd_gen_8layer.py` | 8 层推理测试脚本 |
| `my-tools/tp4_memory_test.py` | TP=4 内存测试脚本 |
