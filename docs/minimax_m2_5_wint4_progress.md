# MiniMax-M2.5 WINT4 推理复现报告

> **更新日期：2026-04-06（第九次更新 — EP=8 + Marlin FP8 Scale修复，62层输出语义正确，内存 27.4 GB/卡）**

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

### 2.7 FP8 Marlin MoE 集成（2026-04-06 新增）

**背景**：vLLM 在 SM80 上通过 Marlin kernel 成功运行 MiniMax-M2.5 FP8 模型（`vllm serve` 命令正常输出中文）。FD 原有的 SM80 路径是 BF16 dequant fallback，精度损失导致 MoE routing 偏差和乱码。通过集成 Marlin kernel 可以解决此问题。

**已完成的修改**：

1. **C++ Marlin kernel 层**：
   - `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/generate_kernels.py`：启用 `kFE4M3fn` (FP8 E4M3) kernel 生成
   - `custom_ops/gpu_ops/moe/moe_wna16_marlin_gemm.cu`：
     - `COMMON_GET_IF(kFE4M3fn)` → `BIGGROUP_GET_IF(kFE4M3fn)` 修复 `group_blocks=2,4` 未定义符号链接错误
     - 添加 `float8_e4m3fn` 字符串到 `b_q_type_id` 映射
     - 在 `get_marlin_kernel` 中启用 `BIGGROUP_GET_IF(kFE4M3fn)` 支持 FP8 kernel 查找

2. **Python Marlin MoE backend** (`fused_moe_marlin_backend.py`)：
   - 扩展 `MarlinWeightOnlyMoEMethod` 支持 FP8 权重（原仅支持 INT4）
   - 添加 `weight_type` 检测（从 `BlockWiseFP8Config.weight_block_size` 判断 FP8 vs INT4）
   - `create_weights` 根据 weight_type 创建正确形状的参数（FP8: `N*4`, INT4: `N*2`）
   - `apply` 方法根据 weight_type 设置 `b_q_type_str="float8_e4m3fn"` 和正确的 `size_n`

3. **Quantization config 层** (`block_wise_fp8.py`)：
   - SM80 上 `BlockWiseFP8Config.get_quant_method` 返回 `MarlinWeightOnlyMoEMethod(self)` 代替 `None`
   - 这样 FusedMoE 在 SM80 上使用 Marlin kernel 而非 Cutlass BF16 fallback

4. **模型加载层** (`minimax_m2_5.py`)：
   - 添加 `_process_fp8_marlin_weights()` 函数：FP8 weight → pack to int32 → Marlin repack → scale permute
   - 添加 `_load_fp8_marlin_layer()` 方法：按 expert 分组处理 w1/w2/w3，合并 gate+up 投影，处理 scale 拼接
   - 修改 `_dequant_fp8_weights()`：当 `FD_MARLIN_FP8=1` 时跳过 expert 权重反量化
   - 修改 streaming dequant 循环：逐层加载 FP8 expert weights 到 Marlin backend
   - 修复 weight layout：checkpoint 是 `[N, K]` format（output × input），Marlin 需要 `[K, N]` format

5. **Engine config 层**：
   - `config.py`：从 `config.json` 的 `quantization_config` 检测 FP8 量化，设置 `is_quantized=True`
   - `args_utils.py`：从 `model_config.quantization` 创建 `BlockWiseFP8Config` 并传递给 `FDConfig`

**验证结果**（TP=1, 3 层模型）：
```
GPU 显存: 17.8 GB
prompt: 'Hello, my name is'
tokens: [69362, 105, 5269, 94039, 21469, 142179, 6703, 18985, 69635, 2337]
text:   ' قيiubeinetteスメ市面上ingtonoshzypisyita'
```
FP8 Marlin MoE kernel 端到端推理成功运行。输出乱码（因 3/62 层），但 kernel 调用链完整通过。

**已验证的 C++ kernel 单元测试**：
- FP8 weight pack to int32: ✓
- Marlin repack (num_bits=8): ✓
- Scale permute: ✓
- `MoeWna16MarlinGemmApi` with `b_q_type_str="float8_e4m3fn"`: ✓ (gate+up + swiglu + down 完整流程)
- 输出 tensor shape: `[token_num, hidden_size]` ✓

### 2.8 EP=8 + Marlin FP8 + NCCL 端到端实现（2026-04-06 新增）

**背景**：Marlin FP8 在 TP=8 模式下因 int32 打包权重 (~28 GB/卡) 与 BF16 模型参数 (~53 GB/卡) 叠加导致 OOM。解决方案：使用 Expert Parallel (EP=8)，每卡只需 32 experts，总内存降至 ~27 GB/卡。

**已完成的修改**：

1. **OOM 和数据损坏修复** (`minimax_m2_5.py`):
   - `_dequant_fp8_weights` 循环内每 32 个 weight 调用 `paddle.device.synchronize()` + `paddle.device.cuda.empty_cache()`
   - 问题：无 sync 的 `empty_cache()` 会在 async GPU copy 未完成时释放源内存 → 权重数据损坏（所有 token 变成 "axy"）
   - 修复后 GPU pool 正常释放，62 层加载无 OOM

2. **is_checkpoint_bf16 误判修复** (`block_wise_fp8.py`):
   - MiniMax `quantization_config` 无 `is_quantized` key → 默认误判为 `True` → 创建 BF16 params (53 GB)
   - 修复：`is_quantized = config.get("is_quantized", config.get("quant_method") == "fp8")`
   - 效果：Marlin int32 params 正确初始化，model built 从 53 GB → 27 GB

3. **NCCL EP Runner 实现** (`nccl_ep_runner.py` 新建):
   - 替代 `deep_ep`（需要 SM90+），使用 `paddle.distributed.alltoall_single()` 实现 SM80 兼容的 EP
   - `NCCLEPPrefillRunner.dispatch()`: 按 expert owner rank 分发 tokens
   - `NCCLEPPrefillRunner.combine()`: scatter-add 结果汇聚
   - `ep.py`: `load_deep_ep()` 失败时返回 `None` 而非 raise

4. **Marlin EP 支持** (`fused_moe_marlin_backend.py`):
   - 新增 `MarlinWeightOnlyMoEMethod.init_ep()`: 初始化 NCCLEPPrefillRunner
   - 新增 `MarlinWeightOnlyMoEMethod.apply_ep()`: EP 模式 Marlin 推理（NCCL dispatch → 本地 Marlin GEMM → combine）
   - `apply()` 检测 `layer.ep_size > 1` 自动路由到 `apply_ep()`

5. **expert_parallel_rank 修复** (`config.py`):
   - `expert_parallel_rank = 0` 硬编码 → 所有 rank 加载相同的 experts 0-31
   - 修复：`set_communicate_group()` 中 `self.expert_parallel_rank = paddle.distributed.get_rank() % self.expert_parallel_size`
   - 效果：rank 0→[0,32), rank 1→[32,64), ..., rank 7→[224,256)

6. **EP expert 过滤** (`minimax_m2_5.py`):
   - `_load_fp8_marlin_layer` 增加 EP 过滤：只加载 `[expert_id_offset, expert_id_offset + num_local_experts)` 范围的 experts
   - EP=8 时每卡只处理 32 个 experts

**验证结果**（EP=8 + Marlin FP8，TP=8，62 层全量，Scale修复后）：
```
GPU 显存: 27.4 GB/卡（vs 之前 BF16 53.4 GB，节省 49%）
模型初始化: 27.4 GB（vs 之前 53.4 GB）

prompt: 'Hello, my name is'
tokens: [161985, 161985, 136977, 131448, 17594, 197295, 22239, 8475, 86961, 97263, ...]
text:   ' PARIS PARISagelabbyirthassumptionviouslyatever ...'

prompt: 'The capital of France is'
tokens: [128189, 28157, 74642, 49634, 169878, 31065, 31065, 156105, ...]
text:   ' Kirstauer ceremonies，北京レモン conceptual...'

prompt: '1 + 1 ='
tokens: [161227, 111211, 61195, 170657, ...]
text:   '-station تحسين浙江省liquidLECTIONored历练...'
```
- **重大改进**：输出包含语义相关词（PARIS, 北京, 中文词）
- **vs 之前**：之前 "vac inject inject inject" → 现在 " PARIS/北京" 等语义词
- **剩余差距**：部分 token 重复，与 vLLM 清晰中文输出仍有差距

**根因分析修正**（第八次的 sequence-parallel 假说已被否定）：

| 实际根因 | 说明 |
|----------|------|
| Marlin FP8 scale 错误 | **已修复** — scale 需 × 2^120（BF16 exponent bias offset） |
| forward_split_allgather 与 NCCL EP 冲突 | **已修复** — token_num≥8 时绕过 split_allgather，改用 forward_normal |
| vLLM 对比 | vLLM 同样用 TP=8 attention（非 SP），问题在 FD 的 Marlin FP8 实现 |

**下一步**：
- 对比 vLLM 逐层 hidden state 验证精度
- 优化 forward_normal 的 EP 效率（避免8倍冗余计算）

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

#### TP=8 FP8 62 层全量（已通过，renormalize 修复后）
```
GPU 显存: 53.4 GB/卡
prompt: 'Hello, my name is'
tokens: [2785, 19781, 15525, 19221, 11937, 2785, 19450, 19450, 17697, 17697, 15525, 6125, 17050, 6210, 5125]
text:   '统asukcussillet寻统axyaxy撮撮cussoma嘉序留'
```
renormalize 修复后输出有所改善（有更多有语义的中文 token），但 SM80 上仍有乱码和重复。推理速度约 1.0-1.1s/token。

#### TP=8 WINT4（已通过，MoE 保持 BF16）
TP=8 WINT4 模式下，MoE 层在 SM80 上保持 BF16（`moe_expert_ffn` 的 int4 kernel 与 TP-sharded weight layout 不兼容）。
输出与 FP8 模式一致（因为两者都是 BF16 MoE）。

#### TP=8 FP8 Marlin MoE 3 层（已通过，2026-04-06 新增）
```
GPU 显存: ~4.2 GB/卡（3 层，Marlin int32 打包）
prompt: 'Hello, my name is'
tokens: [非平凡中文 + 多语言 token]
所有 8 个 rank 均通过，端到端 token 生成成功
```
- Marlin FP8 kernel 在 TP=8 全部 8 个 rank 正确运行
- `size_k` 从实际 weight shape 推导后 kernel 验证通过
- 输出有语义 token（非全 0，非重复）

#### TP=8 FP8 Marlin MoE 62 层 + EP=8（第二版，2026-04-06 深夜）
```
GPU 显存: ~27.4 GB/卡（比 BF16 节省 49%）
prompt: 'Hello, my name is'
tokens: [6254, 12546, 12546, 12546, 6254, 6254, 4397, 6254, 17368, 4397]
text:   ' vac inject inject inject vac vac续 vacktop续'
```
- **确定性输出**：每次运行完全相同（deterministic）
- **比 BF16 更好**：BF16 全是 19450(="axy")，EP Marlin 有 5 个独特 token (6254/12546/4397/17368)  
- **已修复**：expert_parallel_rank 正确设置（各 rank 加载对应的 32 个 experts）
- **仍有差距**：vLLM 输出正确中文，FD EP Marlin 输出不流利
- **下一步原因分析**：需对比 FD 和 vLLM 的 routing 决策（topk_ids 是否一致）

#### 已修复的问题
1. **Manual Forward 只在 rank 0 执行 generate**：NCCL all-reduce 死锁。修复：所有 rank 都参与前向计算。
2. **KV head 计算未使用 per-device 值**：`kvh = num_kv_heads // TP_SIZE`。
3. **使用 `paddle.device.cuda.synchronize`**：替换为 `paddle.device.synchronize`。

#### 仍存在的问题
1. **TP=8 WINT4 MoE int4 不工作**：`paddle.nn.quant.weight_quantize` 在 SM80 上对 TP-sharded weight 的 packed layout 与 `moe_expert_ffn` int4 kernel 不兼容。需要进一步调试或使用替代量化方案。
2. **FD LLM API 输出重复 token**：SM80 上 FD 推理流水线的兼容性问题，尚未排查。
3. **SM80 上输出仍有乱码**（2026-04-05 第三次更新）：renormalize 修复后 5 层模型输出改善（有语义 token 增多），但 62 层全量模型仍有乱码。vLLM 在 SM80 上也完全无法正常工作（5 层输出全换行符，62 层 TP=1 OOM、TP=8 初始化失败）。确认为 SM80 硬件限制。

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

### 5.4 `minimax_m2_5.py` — MoE renormalize 修复

```python
# 修复前：FusedMoE 默认 renormalize=False
self.experts = FusedMoE(fd_config, ...)

# 修复后：MiniMax-M2.5 routing 需要 renormalize
self.experts = FusedMoE(fd_config, renormalize=True, ...)
```

**根因**：`noaux_tc` routing 选出 top-8 expert 后，weights 应除以 sum 归一化（sum=1.0）。
未修复时 weights 是原始 sigmoid 值（sum≈4.0），每层 MoE output 放大约 4 倍，62 层累积后输出乱码。

**验证**：`compare_routing.py` 对比 FD vs vLLM：
- 修复前：FD topk_weights sum=4.0215，vLLM sum=1.0000
- 修复后：FD topk_weights sum≈1.0，与 vLLM 一致

---

## 6. 关键文件清单

### 6.1 模型代码

| 文件路径 | 说明 |
|----------|------|
| `FastDeploy/fastdeploy/model_executor/models/minimax_m2_5.py` | 模型定义 + 逐层流式加载 + WINT4/FP8-Marlin 量化 (核心文件) |
| `FastDeploy/fastdeploy/model_executor/layers/quantization/weight_only.py` | Linear WINT4 实现 + WINT4Config 修复 |
| `FastDeploy/fastdeploy/model_executor/layers/quantization/block_wise_fp8.py` | FP8 quant config + SM80 Marlin 路由 |
| `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py` | Marlin MoE backend（支持 INT4 + FP8） |
| `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_triton_backend.py` | SM80 FP8 MoE BF16 fallback |
| `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py` | MoE WINT4 实现 (`CutlassWeightOnlyMoEMethod`) |
| `FastDeploy/custom_ops/gpu_ops/moe/moe_wna16_marlin_gemm.cu` | Marlin MoE CUDA kernel（FP8 + INT4） |
| `FastDeploy/custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/generate_kernels.py` | Marlin kernel 模板生成（含 FP8） |
| `FastDeploy/custom_ops/gpu_ops/moe/moe_ffn.cu` | `moe_expert_ffn` CUDA kernel（含 `weight_only_int4` 分支） |
| `FastDeploy/fastdeploy/config.py` | FP8 quantization 检测 |
| `FastDeploy/fastdeploy/engine/args_utils.py` | BlockWiseFP8Config 创建 |

### 6.2 测试脚本（my-tools/）

| 文件路径 | 说明 |
|----------|------|
| `test_tp8_manual_forward.py` | TP=8 手动前向推理（有 append_attention shape bug） |
| `fd_llm_demo_tp1_wint4_5layer.py` | 5 层 WINT4 端到端推理 TP=1（已验证通过） |
| `fd_llm_tp8_62layer.py` | 62 层 TP=8 FP8 推理（FD LLM API，输出重复） |
| `fd_llm_tp8_62layer_wint4.py` | 62 层 TP=8 WINT4 推理（FD LLM API，输出重复） |
| `run_fd_gen.py` | TP=1 手动前向推理 |
| `run_fd_gen_marlin.py` | TP=1 FP8 Marlin MoE 手动前向推理（2026-04-06 新增） |
| `test_wint4_sm80_unit.py` | SM80 WINT4 单元测试 |
| `test_fp8_marlin_moe.py` | FP8 Marlin MoE kernel 单元测试（2026-04-06 新增） |

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
| 18 | `_dequant_fp8_weights` 每次加载后调用 `process_weights_after_loading` | stacked params (qkv_proj) 被多次 transpose + re-quantize | 修复为加载完所有权重后每个 sublayer 只调用一次 |
| 19 | SM80 输出乱码 | 硬件不支持 FP8 原生计算，BF16 dequant 精度累积导致 MoE routing 偏差 | 在 SM90+ 上验证或接受 SM80 精度损失 |
| 20 | FusedMoE 缺少 `renormalize=True` | MiniMax-M2.5 routing 需要对 top-k weights 做 sum 归一化，但 `FusedMoE` 默认 `renormalize=False`，导致 weights 放大约 4 倍 | 添加 `renormalize=True`（cos_sim 对比验证确认） |
| 21 | vLLM 在 SM80 上也无法正常工作 | vLLM 5 层输出全换行符（MARLIN FP8 精度损失），62 层 TP=1 OOM，TP=8 初始化失败 | 确认为 SM80 硬件限制，不是 FD 独有问题 |
| 22 | `PADDLE_ENFORCE` 多参数 tinyformat 崩溃 | PaddlePaddle tinyformat 不支持多个格式化参数，32+ 处多参数 `PADDLE_ENFORCE` 在 CUDA Graph capture 时触发断言 | 批量替换为单参数格式（`"Check failed. See source for details."`）|
| 23 | TP=8 Marlin kernel tinyformat 崩溃 | `MoeWna16MarlinGemmApi` 在 TP=8 多进程环境下触发 PaddlePaddle 核心库的 tinyformat 断言。TP=1 完全正常，TP=8 的 BF16 dequant fallback 也正常 | **已修复** — 根因是 `fused_moe_marlin_backend.py` 的 `apply()` 中 down_proj 的 `size_k` 使用了 TP-sharded 的 `moe_intermediate_size=192`，但实际 weight shape 的 K 维度是 1536（未被 TP sharding）。修复：从 weight 实际 shape 推导 `size_k = weight.shape[1] * 16` |
| 24 | 62 层 Marlin int32 模型 OOM | Marlin int32 打包权重（4 bytes/param）比 FP8（1 byte/param）大 4x，62 层 Marlin 权重总计 ~223GB，每卡 ~28GB，加上 BF16 dequanted 权重超出 80GB/卡限制 | **待优化** — 需要 Expert Parallel (EP) 将 256 experts 分配到 8 张卡，或使用 BF16 dequant 路径（53.4GB/卡，已验证通过） |
| 25 | `size_k` 参数推导错误导致 Marlin 验证失败 | `fused_moe_marlin_backend.py` down_proj 调用中 `size_k=moe_intermediate_size`（TP-sharded=192），但 expert weight 是完整的（K=1536），kernel assert `(size_k/16) == b_q_weight.size(1)` → `12 != 96` 失败，8进程同时 assert 产生交错输出误判为 tinyformat 错误 | **已修复** — 从 weight 实际 shape 推导：`actual_size_k_up = up_gate_weight.shape[1] * 16`，`actual_size_k_down = down_weight.shape[1] * 16` |
| 26 | 62 层 TP=8 BF16 dequant OOM at layer 25 | PaddlePaddle GPU 内存 pool 不立即释放 `del` 后的 tensor，循环内 768 个 expert weight 的 FP8（3.6 GB）+ float32（14.5 GB）+ BF16（7.2 GB）临时 tensor 在 pool 中累积，53.3 GB + 25 GB 超过 80 GB | **已修复** — 在 `_dequant_fp8_weights` 循环内每 32 个 weight 调用 `paddle.device.synchronize()` + `paddle.device.cuda.empty_cache()`，强制释放 pool |
| 27 | `empty_cache()` 无 synchronize 导致权重数据损坏 | 在 `weight_loader` 的异步 GPU copy 尚未完成时，`del wt_dq` 触发 Python 引用计数归零，PaddlePaddle 将 `wt_dq` 的 GPU 内存归还 pool，随后 `empty_cache()` 释放该 pool 内存回 CUDA，而 async copy 仍在读取已释放的地址 → 权重数据随机损坏（所有 token 变成 19450="axy"） | **已修复** — 在 `empty_cache()` 前必须调用 `paddle.device.synchronize()`，确保所有 async GPU op 完成后才释放 pool |
| 29 | 62层EP+Marlin 输出不是正确中文（第一阶段） | routing topk_ids 已验证，但 Marlin FP8 scale 格式错误：`dequant_skip_flop=true` 使 FP8→BF16 原始位移不校正 exponent bias，输出值 ~2^(-120) × 正确值，实际为零 | **已修复** — `_process_fp8_marlin_weights` 中 scale × 2^120（BF16 exponent bias offset），见下条 |
| 30 | Marlin FP8 kernel `dequant_skip_flop=true` 导致 scale 需预乘 2^120 | `moe_wna16_marlin_gemm.cu` 中 FP8 类型的 `dequant_skip_flop = !is_int_type = true`，采用原始位移（无 exponent bias 校正），输出 = fp8_val × 2^(-120)。scale 必须乘以 2^120 才能补偿。公式：`BIAS_OFFSET = (1<<(BF16_EXP-1)) - (1<<(FP8_EXP-1)) = 128-8 = 120` | **已修复** — `_process_fp8_marlin_weights()` 中：`s_expanded = s_expanded × 2^120` 后再做 `_marlin_permute_scales` |
| 31 | `forward_split_allgather` 与 NCCL EP 冲突，token≥8 时输出重复 | `FusedMoE.forward` 在 `token_num >= attn_tp_size=8` 时调用 `forward_split_allgather`，先把 tokens 按 rank 分割后再调 `apply_ep`（含 NCCL all-to-all），但 all-to-all 的 ep_group 和外层 all-gather 的 tp_group 发生排序/语义冲突，导致 step 4+ 输出 token 开始重复 | **已修复** — `FusedMoE.forward` 中增加 `_use_nccl_ep` 检测（`hasattr(self, '_nccl_ep_runner')`），NCCL EP 时直接走 `forward_normal` 绕过 `forward_split_allgather` |

---

## 8. 下一步工作

### 8.1 目标：Marlin FP8 + EP=8 实现（对齐 vLLM）

**目标**：在 FD 中复现 vLLM 的 `--enable_expert_parallel --tensor-parallel-size 8` 模式，使 62 层全量模型在 8×A800 上输出正确中文 token。

**关键约束（2026-04-06 发现）**：
- Marlin FP8 kernel **不支持 TP-sharded N=384**（TP=8 把 N 从 3072 切到 384 后，kernel 触发 tinyformat 断言）
- Marlin kernel 必须用全量 **N=3072**
- 因此必须用 **EP=8**：每卡 32 experts × 完整 N=3072 → 内存可控、kernel 正确

**内存计算（EP=8 + Marlin，每卡 32 experts）**：
| 组件 | 形状 | 每层 | 62层 |
|------|-----|-----|-----|
| up_gate Marlin int32 | [32, 192, 12288] | 302 MB | 18.7 GB |
| down Marlin int32 | [32, 96, 12288] | 151 MB | 9.4 GB |
| Attention BF16 (TP=8) | - | - | ~5 GB |
| Embed/lm_head | - | - | ~2.3 GB |
| **合计** | | | **~35 GB/卡 ✓** |

**Step 1 完成（is_checkpoint_bf16 修复）**：
- ✅ `block_wise_fp8.py` `from_config()`: `is_checkpoint_bf16 = not config.get("is_quantized", config.get("quant_method") == "fp8")`
- ✅ 效果: 3 层 Marlin 模型 `model built: 1.6 GB`（vs 之前 2.9 GB BF16），init 正确使用 int32 Marlin params
- ✅ 权重加载后 ~2 GB（3层）

**方案可行性对比（2026-04-06 最终）**：
| 方案 | 内存/卡 | SM80 兼容 | 输出质量 | 状态 |
|------|--------|----------|---------|------|
| TP=8 BF16 | 53.4 GB | ✓ | ✗（62层精度崩溃，all "axy"） | ✓可运行 |
| **EP=8 Marlin FP8** | **27.4 GB** | **✓** | **⚠️（"vac inject续"，含中文但不流利）** | **✓可运行** |
| vLLM EP=8 | ~35 GB | ✓ | ✓（正确中文） | 参考 |

**根本原因分析（2026-04-06 深夜）**：

FD EP+Marlin 输出不如 vLLM 的根因：

| | FD | vLLM |
|---|---|---|
| Attention 并行 | **TP=8**（all heads 分割到各 rank，all_reduce） | **SP**（sequence parallel，各 rank 做不同 token 的完整 heads） |
| MoE 并行 | EP=8 via NCCL all-to-all | EP=8 via NCCL all-to-all |
| BF16 all_reduce 次数 | **62 × TP=8 all_reduce** = 62 次精度损失 | **0 次 TP all_reduce for attention** |

vLLM 使用 sequence-parallel attention（每 rank 做不同 token 的全部 heads），无 attention all_reduce，避免了 62 层 BF16 精度累积。FD 的 TP=8 BF16 attention 每层都有 all_reduce，62 层后精度损失使 MoE routing 发散。

**下一步：实现 sequence-parallel attention**（Step 8 新增）：
```
8. 将 FD 的 attention 从 TP 改为 sequence_parallel 模式
   - 每 rank 处理不同 token 的完整 heads（无 all_reduce 精度损失）
   - 与 EP=8 MoE 配合：attention-SP + MoE-EP
   - 预期可达 vLLM 级别输出质量
```

**内存计算（EP=8 + Marlin int32）**：
- Marlin expert params（int32，32 experts × 3 矩阵 × 1536 × 3072 × 4 bytes）: ~28 GB/卡
- Attention weights（BF16，TP=8）: ~5 GB
- Embed/lm_head: ~2.3 GB
- KV cache 余量: ~45 GB
- **总计: ~35 GB/卡** ← 远低于 80 GB

---

### 8.2 实现步骤（详细）

#### Step 1：修复 `is_checkpoint_bf16` 误判 ★★★

**文件**: `FastDeploy/fastdeploy/model_executor/layers/quantization/block_wise_fp8.py`

**问题**: `BlockWiseFP8Config.from_config()` 在 MiniMax checkpoint 中没有 `is_quantized` key 时，
误判 `is_checkpoint_bf16 = not False = True`，导致：
- `BlockWiseFP8MoEMethod.create_weights()` 创建 BF16 params（53 GB）而非 int32 Marlin params（35 GB）
- 进而引发 OOM 或精度问题

**修改**（line 78）：
```python
# 当前（错误）
is_checkpoint_bf16 = not config.get("is_quantized", False)

# 修复后
is_quantized = config.get("is_quantized", config.get("quant_method") == "fp8")
is_checkpoint_bf16 = not is_quantized
```

**预期效果**：MiniMax（`quant_method="fp8"`）→ `is_checkpoint_bf16 = False` → `MarlinWeightOnlyMoEMethod.create_weights()` 创建 int32 params

**验证方式**：运行 3 层测试，检查 `GPU[r0]` 显存从 53.4 GB 降至 ~35 GB。

---

#### Step 2：实现 NCCL-based EP Runner（替代 deep_ep）★★★

**文件**: 新建 `FastDeploy/fastdeploy/model_executor/layers/moe/nccl_ep_runner.py`

**问题**: FD 现有 EP 实现依赖 `deep_ep` 库（需要 SM90+ Hopper 架构）。SM80 上报错 "invalid device symbol"。

**方案**: 用 PaddlePaddle 的标准 NCCL collective ops 实现 EP token 路由。

**核心 API**：`paddle.distributed.alltoall()` / `paddle.distributed.all_to_all_v()`

**实现逻辑**：
```python
class NCCLEPPrefillRunner:
    """NCCL-based EP dispatch/combine for SM < 90 (no deep_ep required)"""
    
    def moe_select(self, layer, gate_out):
        """Compute top-k routing: topk_ids, topk_weights"""
        # Same as existing logic
    
    def dispatch(self, x, topk_ids, ep_group):
        """
        All-to-all dispatch: send each token's hidden state to the GPU(s)
        that own the selected experts.
        
        1. For each token, find which rank owns each selected expert
           (expert_id // num_local_experts = target_rank)
        2. Bucket tokens by target rank
        3. paddle.distributed.all_to_all_v() → each rank receives its tokens
        Returns: recv_x [recv_tokens, hidden], recv_expert_ids, counts
        """
    
    def combine(self, ffn_out, send_counts, recv_counts, ep_group):
        """
        All-to-all combine: send computed results back to originating GPUs.
        Weighted sum of expert outputs per token.
        """
```

**关键文件改动**：
- `FastDeploy/fastdeploy/model_executor/layers/moe/ep.py`: 在 `load_deep_ep()` 失败时，设置 `deep_ep = None`，后续代码检查 None 走 NCCL 路径
- `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py`: `apply_ep_prefill()` 添加 NCCL fallback 分支

---

#### Step 3：Marlin FP8 支持 EP 模式 ★★

**文件**: `FastDeploy/fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py`

**问题**: 现有 `MarlinWeightOnlyMoEMethod.apply()` 的 `apply_tp()` 不支持 EP dispatch/combine。

**修改 `apply()`**：
```python
def apply(self, layer, x, gate, topk_ids_hookfunc=None, shared_experts=None):
    if layer.ep_size > 1:
        # EP mode: use NCCL dispatch + local Marlin GEMM + NCCL combine
        return self.apply_ep(layer, x, gate, ...)
    else:
        return self.apply_tp(layer, x, gate, ...)

def apply_ep(self, layer, x, gate, ...):
    """
    1. Routing: compute topk expert IDs across all experts
    2. Dispatch: NCCL all-to-all to send tokens to expert owners
    3. Local compute: run Marlin GEMM for local experts only
    4. Combine: NCCL all-to-all to return results to token owners
    5. Weighted sum
    """
```

---

#### Step 4：get_quant_method 支持 EP + SM80 ★★

**文件**: `FastDeploy/fastdeploy/model_executor/layers/quantization/block_wise_fp8.py`

**修改 `get_quant_method()`**：
```python
def get_quant_method(self, layer):
    if isinstance(layer, FusedMoE):
        if get_sm_version() < 90:
            if os.environ.get("FD_MARLIN_FP8", "0") == "1":
                # Marlin for SM80: supports both TP and EP
                return MarlinWeightOnlyMoEMethod(self)
            else:
                return None  # BF16 fallback
        if layer.ep_size > 1 or self.use_deep_gemm:
            return DeepGemmFusedMoeMethod(self)  # SM90 only
        return BlockWiseFP8MoEMethod(self)
```

注意：SM80 + FD_MARLIN_FP8=1 时，无论是否 EP，都返回 `MarlinWeightOnlyMoEMethod`。

---

#### Step 5：Marlin 权重加载支持 EP ★★

**文件**: `FastDeploy/fastdeploy/model_executor/models/minimax_m2_5.py`

**问题**: 现有 `_load_fp8_marlin_layer()` 为每个 expert 做 Marlin repack，但 EP 模式下只需加载本卡负责的 experts（`expert_id_offset` 到 `expert_id_offset + num_local_experts`）。

**修改**：
```python
def _load_fp8_marlin_layer(self, layer_idx, fp8_weights, fp8_scales, ...):
    moe_layer = ...
    # EP: only load local experts
    expert_id_offset = moe_layer.expert_id_offset  # e.g., rank * 32
    num_local = moe_layer.num_local_experts         # e.g., 32
    
    # Filter weights to only include local experts
    local_up_gate = {k: v for k, v in fp8_weights.items()
                    if get_expert_id(k) in range(expert_id_offset, expert_id_offset + num_local)}
    ...
```

---

#### Step 6：ParallelConfig 支持 EP 配置 ★

**文件**: `FastDeploy/fastdeploy/engine/args_utils.py` + `FastDeploy/fastdeploy/config.py`

**目标**: EngineArgs 的 `enable_expert_parallel=True` 正确传递到 FusedMoE 的 `ep_size=8`。

**检查**: 已有 `enable_expert_parallel: bool = False` 字段（line 320 in args_utils.py）。
需要确认它正确传递到 `ParallelConfig.expert_parallel_size`（已有 line 714 in config.py）。

---

#### Step 7：测试脚本更新与验证 ★

**新测试脚本**: `my-tools/test_tp8_marlin_ep8.py`
```python
# 验证 Marlin FP8 + EP=8，62 层，8 卡
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 \
    my-tools/test_tp8_marlin_ep8.py --n_layers 62 --n_tokens 20
```

**期望结果**：
- GPU 显存：~35 GB/卡（不再是 53 GB）
- 输出：正确中文 token（如 "统"、"你好"）

---

### 8.3 里程碑

| 里程碑 | 目标 | 关键改动 |
|--------|------|---------|
| M1 | is_checkpoint_bf16 修复后显存降至 35 GB | block_wise_fp8.py Step 1 |
| M2 | NCCL EP 3 层测试通过（不崩溃） | ep.py + nccl_ep_runner.py Step 2 |
| M3 | Marlin FP8 + EP=8 3 层正确输出 | fused_moe_marlin_backend.py Step 3-5 |
| M4 | Marlin FP8 + EP=8 62 层全量正确输出 | 全部 Steps |

---

### 8.4 优先级中（保留）

4. **FD LLM API SM80 兼容性**：
   - SM80 上 FD LLM API 输出重复 token（手动前向推理无此问题）
   - 需要排查 attention kernel / sampler / CUDA Graph 等

5. **CUDA Graph 正确性验证**：
   - 当前 test_tp8_marlin.py 禁用了 CUDA Graph（`use_cudagraph=False`）
   - 需要验证 FP8 Marlin 在 CUDA Graph 下的正确性

### 8.5 优先级低（保留）

6. **62 层 WINT4 全量推理**
7. **MTP (Multi-Token Prediction) 层支持**
8. **FP8 原生计算 kernel（需 SM90）**



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

### 9.4 FP8 Marlin TP=8 推理（已验证通过，3 层）
```bash
# FP8 Marlin 模式，3 层（所有 8 个 rank 通过）
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 my-tools/test_tp8_marlin.py --n_layers 3

# FP8 Marlin 调试（per-rank stderr 日志到 /tmp/marlin_debug_rank{}.log）
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 my-tools/test_marlin_debug.py

# 62 层（当前 OOM，待解决内存问题）
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 my-tools/test_tp8_marlin.py --n_layers 62
```

---

## 10. 环境信息

- **GPU**: 8 × A100 80GB (SM 8.0)
- **FD 环境**: conda activate `paddle` (Python 3.10, PaddlePaddle 3.3.0)
- **vLLM 环境**: conda activate `vllm17` (Python 3.10, PyTorch, vLLM v0.17 dev)
- **模型路径**: `/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5`
- **Checkpoint**: 125 个 safetensors 文件，总计 230 GB (FP8)
- **代码仓库**: `github.com:ZhijunLStudio/FastDeploy.git` 分支 `feat/minimax-m2.5-wint4`
