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

## 3. 当前状态（2026-04-11）

### 3.1 EP=4 已验证正确的组件
- ✅ Marlin packed weights: byte-identical with vLLM
- ✅ Scales: 100% 一致（max_diff=0）
- ✅ Gate routing: cos=1.0
- ✅ Expert mapping: 正确
- ✅ sorted_token_ids: 正确的 flat index encoding
- ✅ AllReduce: 正确求和
- ✅ 62 层完整加载和推理：端到端运行，无崩溃

### 3.2 EP=4 待修复的问题
- ❌ **输出质量**：n≤15 层正确，n≥20 层开始退化
  - n=15: top-1=`'\n\n'` ✓
  - n=20: top-1=`'\xa0'` ✗
  - n=62: top-1=`'斯基'` ✗

### 3.3 根因定位
**FD 的 Marlin kernel 与 vLLM 的版本有 2216 行差异。**

- FD: `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/marlin_template.h`
- vLLM: `vllm/csrc/moe/marlin_moe_wna16/marlin_template.h`

差异导致浮点计算结果有微小差异，在 15-20 层后累积到足以改变 top-1 token。

**最有效的修复方案：将 FD 的 Marlin kernel 同步到 vLLM 的最新版本。**

### 3.4 vLLM 成功运行 MiniMax-M2.5
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
| `fastdeploy/model_executor/models/minimax_m2_5.py` | **新建** | MiniMax-M2.5 模型实现 |
| `fastdeploy/config.py` | **修改** | partial_rotary_factor, num_local_experts, model_format 检测 |
| `fastdeploy/engine/engine.py` | **修改** | pad_token_id=None 处理 |
| `fastdeploy/input/base_processor.py` | **修改** | get_pad_id() 处理 pad_token_id=None |
| `fastdeploy/model_executor/layers/quantization/block_wise_fp8.py` | **修改** | SM80 BF16 dequant fallback |
| `fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py` | **修改** | Scale 修复，EP=4 dump hooks |
| `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/marlin_template.h` | **待更新** | 需同步到 vLLM 版本 |

---

## 5. 运行命令

### 5.1 FD EP=4 Marlin FP8 推理
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate paddle
FD_MARLIN_FP8=1 python -m paddle.distributed.launch --devices=0,1,2,3 /tmp/ep4_bisect.py --n-layers 62
```

### 5.2 vLLM 8 卡部署（已验证成功）
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
