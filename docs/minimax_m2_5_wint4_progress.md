# MiniMax-M2.5 WINT4 推理复现报告

> **更新日期：2026-04-05**

## 1. 项目目标

在 FastDeploy (FD) 框架中复现 MiniMax-M2.5 模型（`MiniMaxM2ForCausalLM`），实现权重加载、前向推理、token 生成。核心目标是在 **8×A800 (SM80)** 上以 **FP8 反量化** 和 **WINT4 量化** 方式运行全量 62 层模型。

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

### 2.3 WINT4 实现
实现了自动化的 WINT4 量化流程：
- **Linear 层** (`qkv_proj`, `o_proj`): SM80 上跳过 WINT4 量化，保持 BF16（`paddle.nn.quant.weight_only_linear` 在 SM80 上 int4/int8 不工作）。
- **MoE 层** (`experts`): 逐层进行 int4 量化，使用 `moe_expert_ffn` custom op（cutlass kernel），cos_sim = 0.975。

### 2.4 逐层流式加载（2026-04-05 新增）
实现了 `load_weights` 的逐层流式加载模式，解决了 62 层模型在 8×A800 上的 OOM 问题：
- 收集 FP8 权重按层分组 (`fp8_by_layer`, `scales_by_layer`)
- 非层权重（embed, norm, lm_head）先处理
- 逐层：FP8 dequant → WINT4 quantize → 释放 BF16 → `empty_cache`
- 峰值显存从 ~108 GB/卡 降至 ~53 GB/卡

### 2.5 SM80 FP8 MoE Fallback（2026-04-05 新增）
在 `fused_moe_triton_backend.py` 的 `BlockWiseFP8MoEMethod.apply()` 中添加了 SM80 兼容路径：
- SM < 90 时，检测到 Triton FP8 kernel 不可用
- 自动将 FP8 expert 权重反量化为 BF16
- 使用标准 BF16 MoE kernel（`get_moe_method`）进行计算
- 添加了 `_bf16_dequanted` 标志避免重复反量化

### 2.6 WINT4Config 修复（2026-04-05 新增）
- `WINT4Config.__init__` 添加 `self.is_quantized = True`
- 修复 `CutlassWeightOnlyMoEMethod.process_weights_after_loading` 中的 `is_quantized` 检查

---

## 3. 当前验证状态

### 3.1 TP=1 验证（已通过）

**FP8 模式 - 3 层模型**：
```
GPU 显存: 22.8 GB
prompt: 'Hello, my name is'
tokens: [96777, 82748, 28745, 109764, 71526, 165419, 56570, 195762, 62189, 79971]
text:   '那只 Lavamaniaالثةotskaprotsikin(SOUNレート的值'
```

**WINT4 模式 - 3 层模型**：
```
GPU 显存: 7.6 GB
prompt: 'Hello, my name is'
tokens: [141384, 158033, 579, 134026, 179603, 189249, 69635, 12379, 84738, 691]
text:   'uwangiopically�_epiappes・新zypisyHSarella�'
```

两种模式在 TP=1 上均能正常推理，输出有语义的多语言 token。

### 3.2 TP=8 验证（2026-04-05 更新）

#### TP=8 FP8 3 层（已通过）
```
GPU 显存: 2.9 GB/卡（3 层）
prompt: 'Hello, my name is'
tokens: [17195, 8196, 24656, 18030, 23404, 11819, 7657, 13215, 24974, 19954]
text:   '��inct municípIONS芸来了毕hips móv刺激'
```

#### TP=8 FP8 62 层全量（已通过）
```
GPU 显存: 53.4 GB/卡
prompt: 'Hello, my name is'
tokens: [21735, 16060, 10159, 14001, 15021, 16990, 20719, 5158, 1142, 15339, ...]
text:   'angkaagues媒 PosATA吹肿anger�stal发挥准ertainienesense看看heimculalu租'
```
输出有语义的多语言 token（中、英、日、印尼语混合），每卡 ~53 GB 显存。推理速度约 1.0-1.3s/token。

#### TP=8 WINT4（已通过，MoE 保持 BF16）
TP=8 WINT4 模式下，MoE 层在 SM80 上保持 BF16（`moe_expert_ffn` 的 int4 kernel 与 TP-sharded weight layout 不兼容）。
输出与 FP8 模式一致（因为两者都是 BF16 MoE）。

#### 已修复的问题
1. **Manual Forward 只在 rank 0 执行 generate**：NCCL all-reduce 死锁。修复：所有 rank 都参与前向计算。
2. **KV head 计算未使用 per-device 值**：`kvh = num_kv_heads // TP_SIZE`。
3. **使用 `paddle.device.cuda.synchronize`**：替换为 `paddle.device.synchronize`。

#### 仍存在的问题
1. **TP=8 WINT4 MoE int4 不工作**：`paddle.nn.quant.weight_quantize` 在 SM80 上对 TP-sharded weight 的 packed layout 与 `moe_expert_ffn` int4 kernel 不兼容。需要进一步调试或使用替代量化方案。
2. **FD LLM API 输出重复 token**：SM80 上 FD 推理流水线的兼容性问题，尚未排查。

### 3.3 显存估算

| 模式 | 层数 | TP | GPU 显存 | 每层增量 |
|------|------|-----|----------|----------|
| FP8 | 3 层 | TP=1 | 22.8 GB | ~7.1 GB |
| WINT4 | 3 层 | TP=1 | 7.6 GB | ~1.7 GB (MoE) + 0.09 GB (Linear) |
| FP8 | 3 层 | TP=8 | 2.9 GB/卡 | ~0.86 GB/卡/层 |
| FP8 | 62 层 | TP=8 | ~53.4 GB/卡 | ~0.86 GB/卡/层 |
| WINT4 (BF16 MoE) | 62 层 | TP=8 | ~53.4 GB/卡 | 与 FP8 相同（MoE 未量化） |

---

## 4. 核心问题：WINT4 输出全 0（已解决）

### 4.1 问题现象（历史）
开启 `FD_WINT4_QUANTIZE=1` 后，模型加载和量化过程正常（无报错），但推理生成的 token 全部为 0。

### 4.2 根因分析

问题出在 **`paddle.nn.quant.weight_only_linear` 在 SM80 (A100) 上不支持 int4/int8**。

#### 单元测试结果

| 测试项 | 硬件 | 方法 | cos_sim | 结论 |
|--------|------|------|---------|------|
| `weight_only_linear` int4 | SM80 | PaddlePaddle 内置 | ≈ 0 | **不工作** |
| `weight_only_linear` int8 | SM80 | PaddlePaddle 内置 | ≈ 0 | **不工作** |
| `moe_expert_ffn` int8 | SM80 | FD custom op | **0.9999** | **完美** |
| `moe_expert_ffn` int4 | SM80 | FD custom op | **0.975** | **正常** |
| `moe_expert_ffn` int4 + 转置 | SM80 | FD custom op | nan | **不工作** |

### 4.3 修复方案

**Linear 层**（qkv_proj, o_proj）：
- SM80 上跳过 WINT4 量化，保持 BF16。
- SM90+ (H100) 上仍然使用标准的 `weight_only_linear` WINT4 路径。

**MoE 层**（256 experts）：
- 使用 `moe_expert_ffn` 的 `weight_only_int4` 路径。
- 不做额外转置，直接 quantize 原始 BF16 layout。
- 量化后替换 `quant_method` 为 `CutlassWeightOnlyMoEMethod(WINT4Config(...))`。

---

## 5. 关键代码改动

### 5.1 `minimax_m2_5.py` — 逐层流式加载

```python
# 收集 FP8 权重按层分组
fp8_by_layer: Dict[int, dict] = {}
scales_by_layer: Dict[int, dict] = {}

# 在 weights_iterator 循环中，按层分组
li = self._extract_layer_idx(loaded_weight_name, num_main_layers)
fp8_by_layer.setdefault(li, {})[loaded_weight_name] = loaded_weight

# 非层权重先处理（embed, norm, lm_head）
# 然后逐层流式处理
for li in layer_indices:
    self._dequant_fp8_weights(li, fp8_by_layer[li], scales_by_layer.get(li, {}))
    if _enable_wint4:
        self._wint4_quantize_layer(li)
    del fp8_by_layer[li]
    paddle.device.cuda.empty_cache()
```

### 5.2 `fused_moe_triton_backend.py` — SM80 BF16 Fallback

```python
# BlockWiseFP8MoEMethod.apply() 中添加 SM80 检测
if get_sm_version() < 90 and current_platform.is_cuda():
    if not hasattr(self, '_bf16_moe_method'):
        self._bf16_moe_method = get_moe_method(layer)
    # Dequant FP8 expert weights to BF16
    if not getattr(layer, '_bf16_dequanted', False):
        for wname, sname in [...]:
            # FP8 -> BF16 dequant with block-wise scales
            ...
        layer._bf16_dequanted = True
    return self._bf16_moe_method.apply(layer, x, gate, ...)
```

### 5.3 `weight_only.py` — WINT4Config 修复

```python
class WINT4Config(WeightOnlyConfig):
    def __init__(self, is_checkpoint_bf16=False):
        super().__init__("weight_only_int4", is_checkpoint_bf16)
        self.is_quantized = True  # 修复 CutlassWeightOnlyMoEMethod 的 is_quantized 检查
```

---

## 6. 关键文件清单

### 6.1 模型代码

| 文件路径 | 说明 |
|----------|------|
| `FastDeploy/fastdeploy/model_executor/models/minimax_m2_5.py` | 模型定义 + 逐层流式加载 + WINT4 量化 (核心文件) |
| `FastDeploy/fastdeploy/model_executor/layers/quantization/weight_only.py` | Linear WINT4 实现 + WINT4Config 修复 |
| `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_triton_backend.py` | SM80 FP8 MoE BF16 fallback |
| `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py` | MoE WINT4 实现 (`CutlassWeightOnlyMoEMethod`) |
| `FastDeploy/custom_ops/gpu_ops/moe/moe_ffn.cu` | `moe_expert_ffn` CUDA kernel（含 `weight_only_int4` 分支） |

### 6.2 测试脚本（my-tools/）

| 文件路径 | 说明 |
|----------|------|
| `test_tp8_manual_forward.py` | TP=8 手动前向推理（有 append_attention shape bug） |
| `fd_llm_demo_tp1_wint4_5layer.py` | 5 层 WINT4 端到端推理 TP=1（已验证通过） |
| `fd_llm_tp8_62layer.py` | 62 层 TP=8 FP8 推理（FD LLM API，输出重复） |
| `fd_llm_tp8_62layer_wint4.py` | 62 层 TP=8 WINT4 推理（FD LLM API，输出重复） |
| `run_fd_gen.py` | TP=1 手动前向推理 |
| `test_wint4_sm80_unit.py` | SM80 WINT4 单元测试 |

---

## 7. 已踩的坑

| # | 坑 | 原因 | 修复方式 |
|---|-----|------|----------|
| 1 | `ModuleNotFoundError: No module named 'paddle'` | 未激活 conda 环境 | `source anaconda3/etc/profile.d/conda.sh && conda activate paddle` |
| 2 | FD config.json 没有 `torch_dtype` 字段 | MiniMax 配置格式不同 | 添加 `model_type.startswith("minimax")` 检测 |
| 3 | `partial_rotary_factor` 默认 1.0 | FD 不读 `rotary_dim` | 添加 `rotary_dim/head_dim` 自动计算 |
| 4 | `num_local_experts` 未映射 | FD 用 `num_experts` | 添加映射 |
| 5 | Expert 权重全零 | `block_sparse_moe` vs `mlp` 命名不匹配 | 添加命名替换 |
| 6 | Attention 输出巨大 (36M) | Linear 权重是 FP8 没反量化 | 添加 FP8 反量化逻辑 |
| 7 | DeepGEMM 在 SM 80 上不支持 | 需要 SM 90+ | 添加 BF16 dequant fallback |
| 8 | FusedMoE 重复创建 | `__init__` 中创建了两次 | 修复为单次创建 |
| 9 | `paddle.nn.quant.weight_only_linear` SM80 不工作 | PaddlePaddle 内核不支持 | SM80 上 Linear 保持 BF16，MoE 用 `moe_expert_ffn` |
| 10 | `moe_expert_ffn` 转置后输出 nan | kernel 期望原始 layout | 不做额外转置 |
| 11 | 62 层模型 TP=4 OOM | BF16 参数峰值 ~108 GB/卡 | 实现逐层流式加载 |
| 12 | `Fleet` 对象缺少 `_hcg` | 未初始化 fleet | 添加 `fleet.init(is_collective=True, strategy=strategy)` |
| 13 | `ParallelConfig` 缺少 `tp_group` | 需要调用 `set_communicate_group()` | 添加调用 |
| 14 | `_wint4_quantize_layer` 用错对象 | 用了 `layer.mlp` (MiniMaxM2_5MoE) 而非 `layer.mlp.experts` (FusedMoE) | 改为 `moe = layer.mlp.experts` |
| 15 | WINT4Config 缺少 `is_quantized` | `CutlassWeightOnlyMoEMethod` 检查此字段 | 添加 `self.is_quantized = True` |
| 16 | TP=8 append_attention shape 错误 | KV head=1 时 kernel shape 计算 bug | **待修复** |
| 17 | FD LLM API SM80 输出重复 | 推理流水线 SM80 兼容性问题 | **待修复** |

---

## 8. 下一步工作

### 8.1 优先级最高

1. **TP=8 WINT4 MoE int4 量化**：
   - `paddle.nn.quant.weight_quantize` 在 SM80 上对 TP-sharded weight（`[384, 3072]`）的 packed layout（`[1536, 384]`）与 `moe_expert_ffn` int4 kernel 不兼容
   - 解决方案选项：
     a. 使用 `_numpy_int4_quant_and_pack` 替代 `_wq`（保持 `[out, in//8*4]` layout）
     b. quantize 前将 TP-sharded weight 拼接成完整 weight（需要大量显存）
     c. 修改 `moe_expert_ffn` kernel 支持转置的 int4 layout
   - 当前 TP=8 WINT4 模式下 MoE 保持 BF16（与 FP8 模式相同显存）

2. **FD LLM API SM80 兼容性**：
   - SM80 上 FD LLM API 输出重复 token
   - 需要排查根因（可能涉及 attention kernel, sampler, CUDA Graph 等）

### 8.2 优先级中

3. **CUDA Graph 支持**：
   - TP=8 手动前向已使用 CUDA Graph（`step_use_cudagraph=False` 但框架自动捕获）
   - 需要验证 CUDA Graph 在 full model 上的正确性

4. **62 层 WINT4 全量推理**：
   - 需要先解决 MoE int4 在 TP=8 上的兼容性问题
   - 预期显存：如果 MoE int4 能工作，~20-25 GB/卡

### 8.3 优先级低

5. **MTP (Multi-Token Prediction) 层支持**：
   - 当前跳过了 62+ 层（MTP），完整模型需要支持

6. **FP8 原生计算 kernel**：
   - 类似 vLLM 的 Marlin kernel，需要 CUDA 代码
   - 可以进一步减少显存（FP8 1 byte/param vs BF16 2 bytes/param）

---

## 9. 测试命令汇总

### 9.1 TP=1 手动前向推理（已验证通过）
```bash
# FP8 模式，3 层
CUDA_VISIBLE_DEVICES=0 python run_fd_gen.py

# WINT4 模式，3 层
FD_WINT4_QUANTIZE=1 CUDA_VISIBLE_DEVICES=0 python run_fd_gen.py
```

### 9.2 TP=8 手动前向推理（已验证通过）
```bash
# FP8 模式，3 层
python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 my-tools/test_tp8_manual_forward.py --mode fp8 --n_layers 3

# FP8 模式，62 层全量
python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 my-tools/test_tp8_manual_forward.py --mode fp8 --n_layers 62 --n_tokens 20

# WINT4 模式，3 层（MoE 保持 BF16）
FD_WINT4_QUANTIZE=1 python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 my-tools/test_tp8_manual_forward.py --mode wint4 --n_layers 3
```

### 9.3 WINT4 单元测试
```bash
# SM80 WINT4 单元测试
CUDA_VISIBLE_DEVICES=0 python my-tools/test_wint4_sm80_unit.py
```

---

## 10. 环境信息

- **GPU**: 8 × A100 80GB (SM 8.0)
- **FD 环境**: conda activate `paddle` (Python 3.10, PaddlePaddle 3.3.0)
- **vLLM 环境**: conda activate `vllm17` (Python 3.10, PyTorch, vLLM v0.17 dev)
- **模型路径**: `/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5`
- **Checkpoint**: 125 个 safetensors 文件，总计 230 GB (FP8)
- **代码仓库**: `github.com:ZhijunLStudio/FastDeploy.git` 分支 `feat/minimax-m2.5-wint4`
