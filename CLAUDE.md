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

## 2. 完成的工作（有意义的部分）

### 2.1 模型实现文件
**文件**: `FastDeploy/fastdeploy/model_executor/models/minimax_m2_5.py`

实现了完整的 MiniMax-M2.5 模型架构：
- `MiniMaxRMSNorm` — 匹配 vLLM 的 `MiniMaxText01RMSNormTP`，per-token full-vector RMSNorm
- `MiniMaxM2_5Attention` — GQA + QK-Norm + partial RoPE
- `MiniMaxM2_5MoE` — 256 experts, top-8, sigmoid scoring, `e_score_correction_bias`
- `MiniMaxM2_5DecoderLayer` / `MiniMaxM2_5Model` / `MiniMaxM2ForCausalLM`
- 注册为 `@ModelRegistry.register_model_class(architecture="MiniMaxM2ForCausalLM")`

### 2.2 配置兼容性修改
**文件**: `FastDeploy/fastdeploy/config.py`

- 添加 `partial_rotary_factor = rotary_dim / head_dim` 自动计算（MiniMax: 64/128=0.5）
- 添加 `num_local_experts` → `num_experts` 映射
- 添加 `model_format="torch"` 检测（MiniMax config.json 没有 `torch_dtype` 字段）

### 2.3 FP8 权重反量化
在 `load_weights` 中实现了 FP8 → BF16 反量化：
- CPU numpy 反量化（避免 GPU OOM）
- 支持所有权重类型（Linear + Expert）
- block_size=128 的 scale 展开
- 命名映射：`block_sparse_moe` → `mlp`，`w1/w2/w3` → `up_gate_proj/down_proj`，`e_score_correction_bias` 映射

### 2.4 权重加载修复
- `pad_token_id=None` 处理（`engine.py` + `base_processor.py`）
- `fp8_quant_blockwise()` 的 `using_ue8m0_scale` 参数兼容性修复
- SM 80 (A100) 上 DeepGEMM 不支持的 BF16 dequant fallback

### 2.5 验证通过的输出
**FD 5层模型（手动方式）**：
```
prompt: 'Hello, my name is'
tokens: [23541, 102417, 14666, 124783, 145609, 110712, 27022, 74541, 187744, 186943]
text:   ' св屏障诊reno自己的身体FCFFFaques Поједина多加 Alamos'

prompt: 'The capital of France is'
tokens: [125965, 133654, 27170, 154402, 171232, 74541, 38442, 110712, 59797, 167783]
text:   ' аутохтониdust有个ITIuyama ПојединаruzFCFFFhartavon'

prompt: '1 + 1 ='
tokens: [14534, 12318, 35629, 1115, 153519, 18318, 176369, 134998, 74541, 15824]
text:   '委员imest Metaax安排的acent mempromADV Поједина署'
```
（乱码是因为只有 5/62 层，但输出非平凡且可复现）

**精度对齐验证**（FD vs vLLM 手动前向）：
```
Embed:  cos=1.000000  diff=0.00000000
Layer 0 norm: cos=1.0000  (per-token cos_sim: [0.999997, 0.999997, 0.999992, 0.999996, 0.999997])
```

---

## 3. 失败的部分

### 3.1 FD LLM API 无法在 A100 上启动
**命令**: `CUDA_VISIBLE_DEVICES=4 python fd_llm_demo.py`

**失败原因**：
1. FD 的 `block_wise_fp8` quantization 在 SM 80 上调用 DeepGEMM，但 DeepGEMM 需要 SM 90+
2. `weight_scale_inv` 参数未正确加载（q/k/v 合并后 scale 没有合并）
3. FP8 权重反量化后的 shape 和 BF16 模式不兼容

**根本原因**：FD 的 LLM API 路径强制使用 `block_wise_fp8` quantization（根据 config.json 的 `quantization_config`），而这个 quantization 在 SM 80 上不完整。

### 3.2 vLLM LLM API 在 A100 上输出全是 `\n`
**命令**: `CUDA_VISIBLE_DEVICES=5 python run_vllm_gen.py`

**原因**：vLLM 在 SM 80 上使用 Marlin weight-only FP8 kernel，精度损失大，5层模型输出基本是噪声。

### 3.3 全参模型无法在 4 张 A100 上运行
- BF16 模型需要 ~460 GB 显存
- 4 × A100 = 320 GB，不够
- 需要 8 张卡 (TP=8) 或保持 FP8 量化

### 3.4 TP=4 无法工作
FD 的 TP=4 路径有 attention kernel 兼容问题（`group_size=0` 不支持）。

---

## 4. 运行命令汇总

### 4.1 FD 手动方式（能工作）
```bash
# 5层模型，TP=1，GPU 6
CUDA_VISIBLE_DEVICES=6 python run_fd_gen.py

# 输出见上文"验证通过的输出"
```

### 4.2 FD 1层精度对齐测试
```bash
# FD 端（GPU 6）
CUDA_VISIBLE_DEVICES=6 python test_step_align.py --backend fd

# vLLM 端（GPU 5）
CUDA_VISIBLE_DEVICES=5 python test_step_align.py --backend vllm

# 对比
python test_step_align.py --backend compare
```

### 4.3 vLLM 手动方式（能工作但输出全\n）
```bash
# 5层模型，TP=1，GPU 5
CUDA_VISIBLE_DEVICES=5 python run_vllm_gen.py
```

### 4.4 FD LLM API（失败）
```bash
CUDA_VISIBLE_DEVICES=4 python fd_llm_demo.py
# 失败原因：block_wise_fp8 在 SM 80 上不兼容
```

### 4.5 vLLM LLM API（能启动但输出全\n）
```bash
# 需要先修改 config.json 的 num_hidden_layers=5
CUDA_VISIBLE_DEVICES=5 python vllm_llm_demo.py
```

---

## 5. 已踩的坑

| # | 坑 | 原因 | 修复方式 |
|---|-----|------|---------|
| 1 | `ModuleNotFoundError: No module named 'paddle'` | 未激活 conda 环境 | `source anaconda3/etc/profile.d/conda.sh && conda activate paddle` |
| 2 | FD config.json 没有 `torch_dtype` 字段 | MiniMax 配置格式不同 | 添加 `model_type.startswith("minimax")` 检测 |
| 3 | `partial_rotary_factor` 默认 1.0 | FD 不读 `rotary_dim` | 添加 `rotary_dim/head_dim` 自动计算 |
| 4 | `num_local_experts` 未映射 | FD 用 `num_experts` | 添加映射 |
| 5 | Expert 权重全零 | `block_sparse_moe` vs `mlp` 命名不匹配 | 添加 `.replace(".block_sparse_moe.", ".mlp.")` |
| 6 | Attention 输出巨大 (36M) | Linear 权重是 FP8 没反量化 | 添加所有 FP8 权重的反量化逻辑 |
| 7 | FD profiling 阶段 OOM | KV cache 分配过多 | 使用 `num_gpu_blocks_override=80` |
| 8 | `max_num_batched_tokens` 超限 | 默认 8192 > max_num_seqs × max_model_len | 设置 `max_model_len=4096` |
| 9 | vLLM `SamplingParams` 循环导入 | spawn 模式下 monkey-patch 不生效 | 修改源码：`from vllm import SamplingParams` → `from vllm.sampling_params import SamplingParams` |
| 10 | vLLM `quantization_config: None` 报错 | `to_dict()` 调用 `None.to_dict()` | 不传 `quantization_config: None` override |
| 11 | `pad_token_id=None` 传给 worker | MiniMax 没有 pad_token | `engine.py` 中 None → -1 |
| 12 | `fp8_quant_blockwise()` 参数不兼容 | paddle 版本不支持 `using_ue8m0_scale` | 删除该参数 |
| 13 | DeepGEMM 在 SM 80 上不支持 | 需要 SM 90+ | 添加 BF16 dequant fallback |
| 14 | `weight_scale_inv` 未加载 | q/k/v 合并后 scale 没有合并 | 在 `load_weights` 中加载 scale（部分修复） |

---

## 6. 下一步需要做的

### 6.1 优先级高
1. **修复 FD LLM API 在 SM 80 上的兼容性**：
   - 在 `block_wise_fp8.py` 的 `apply` 方法中，SM 80 需要完整的 BF16 dequant fallback（包括 scale 的正确处理）
   - 或者让 FD 在 SM 80 上跳过 `block_wise_fp8` quantization，直接用 BF16 GEMM
   - 需要处理 q/k/v 合并后 `weight_scale_inv` 的合并逻辑

2. **全参模型验证**：
   - 需要 8 张 A100 (TP=8) 或在 H100 上验证
   - 或者实现 FP8 原生计算 kernel（类似 vLLM 的 Marlin kernel），减少显存占用

3. **vLLM BF16 模式对齐**：
   - 修改 vLLM 的 `fp8.py` 让它在 SM 80 上也走 BF16 dequant 路径（`self.use_marlin = False`）
   - 这样 FD 和 vLLM 用相同的精度路径，token 就能对齐

### 6.2 优先级中
4. **FD LLM API 的 tokenizer fork 问题**：
   - `huggingface/tokenizers` 在 fork 后报错
   - 需要设置 `TOKENIZERS_PARALLELISM=false` 或在 fork 前不使用 tokenizer

5. **MTP (Multi-Token Prediction) 支持**：
   - 当前跳过了 MTP 层（62+），完整模型需要支持

6. **CUDA Graph 支持**：
   - 模型有 `@support_graph_optimization` 装饰器但未测试

### 6.3 优先级低
7. **FP8 原生计算 kernel**：
   - 类似 vLLM 的 Marlin kernel，需要 CUDA 代码
   - 可以大幅减少显存占用（FP8 1 byte/param vs BF16 2 bytes/param）
   - 但实现复杂度高（~3000 行 CUDA 代码）

8. **TP > 1 支持**：
   - 当前 TP=4 有 attention kernel 兼容问题
   - 需要验证 TP=2, TP=4, TP=8 的正确性

---

## 7. 关键文件列表

| 文件 | 操作 | 说明 |
|------|------|------|
| `FastDeploy/fastdeploy/model_executor/models/minimax_m2_5.py` | **新建** | MiniMax-M2.5 模型实现 |
| `FastDeploy/fastdeploy/config.py` | **修改** | partial_rotary_factor, num_local_experts, model_format 检测 |
| `FastDeploy/fastdeploy/engine/engine.py` | **修改** | pad_token_id=None 处理 |
| `FastDeploy/fastdeploy/input/base_processor.py` | **修改** | get_pad_id() 处理 pad_token_id=None |
| `FastDeploy/fastdeploy/model_executor/layers/quantization/block_wise_fp8.py` | **修改** | SM 80 BF16 dequant fallback, 删除不兼容参数 |
| `vllm/vllm/model_executor/models/minimax_m2.py` | **修改** | 添加 layer skip, FP8 scale 处理 |
| `vllm/vllm/v1/sample/logits_processor/builtin.py` | **修改** | 修复循环导入 |
| `vllm/vllm/v1/sample/logits_processor/interface.py` | **修改** | 修复循环导入 |
| `run_fd_gen.py` | **新建** | FD 手动方式 5 层推理 |
| `run_vllm_gen.py` | **新建** | vLLM LLM API 5 层推理 |
| `test_step_align.py` | **新建** | FD vs vLLM 精度对齐测试 |
| `compare_fd_vllm.py` | **新建** | FD vs vLLM token 对比 |
| `fd_llm_demo.py` | **新建** | FD LLM API demo（A100 上失败） |
| `vllm_llm_demo.py` | **新建** | vLLM LLM API demo |

---

## 8. 环境信息

- **GPU**: 8 × A100 80GB (SM 8.0)
- **FD 环境**: conda activate `paddle` (Python 3.10, PaddlePaddle 3.3.0)
- **vLLM 环境**: conda activate `vllm17` (Python 3.10, PyTorch, vLLM v0.17 dev)
- **模型路径**: `/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5`
- **Checkpoint**: 125 个 safetensors 文件，总计 230 GB (FP8)
