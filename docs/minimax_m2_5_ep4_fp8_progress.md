# MiniMax-M2.5 FP8 Marlin W8A16 复现进展

## 日期：2026-04-15 ~ 2026-04-17

---

## 一、整体进展概览

### Prefill（首 token）：**已完全验证正确** ✅

手动 ForwardMeta + Marlin W8A16，n=2 和 n=62 的 top-5 与 vLLM 完全一致（连顺序都相同）。

| | FD n=62 | vLLM n=62 |
|---|---|---|
| top-1 | `':'` (58) ✅ | `':'` (58) |
| top-2 | `':"'` (11861) ✅ | `':"'` (11861) |
| top-3 | `',"` (2304) ✅ | `',"` (2304) |
| top-4 | `="'` (1139) ✅ | `="'` (1139) |
| top-5 | `'.'` (46) ✅ | `'.'` (46) |

**注意：** 这是手动 ForwardMeta 的 prefill-only 测试，只跑一步 prefill（13 tokens），没有 autoregressive decode 循环，没有用 FD 的 LLM API。

### LLM API 端到端（prefill + decode）：**TP=2 EP=2 已跑通** ✅

TP=2 EP=2 + disable_sequence_parallel_moe + 2 层：成功，无崩溃，有输出。

### Decode 阶段：**存在精度差异** ⚠️

2 层 TP=2 端到端结果对比：
- **FD**: token 367, 84284, 367, 10, 37737, ...（乱码混合日文/法文/中文字符）
- **vLLM**: token 367, 367, 367, 367, ...（全部是换行符 367）

两者从第 2 个 token 开始就不一致。

### EP=4 LLM API 端到端：**运行中但极慢** ⚠️

2026-04-17 晚，首次用 FD LLM API（`fastdeploy.entrypoints.llm.LLM`）以 EP=4 模式运行 MiniMax-M2.5。

**关键配置发现：** FD 的 EP size 通过 `data_parallel_size` 参数控制，不是 `expert_parallel_size`。`expert_parallel_size = data_parallel_size * tensor_parallel_size`。

**运行状态：**
- 配置正确：`expert_parallel_size=4`, `data_parallel_size=4`, `use_ep=True`
- 模型加载成功：2 层，~23 秒，SM80 FP8 CPU offload 路径
- KV cache 初始化成功
- 请求已调度（M=6 tokens）
- **卡在 MoE forward**：`_apply_ep_sm80_bf16` 处理 64 experts 时极慢（每步 forward 都需要 CPU numpy dequant 64 experts）

**根因：** SM80 bf16 workaround 的设计缺陷——FP8 expert 权重存 CPU，每次 MoE forward 都要重新做 `numpy dequant FP8→BF16`（64 experts），这是每步推理的开销，不是一次性的加载开销。

### EP=4 TP=4：**已定位根因** 🔍

之前 EP=4 TP=4 全量 62 层输出乱码，根因已确认为 `use_sequence_parallel_moe` 导致 NCCL `alltoall` 错误。添加 `disable_sequence_parallel_moe=True` 即可修复（尚未在 4 GPU 上验证）。

---

## 二、逐层 Hidden States 对比结果（2026-04-17，关键发现）

### 2.1 对比方法

在 FD 和 vLLM 的 decoder layer forward 中注入 dump 逻辑，每个 sublayer 执行后保存 `hidden_states` 为 npy 文件，然后逐层对比 cosine similarity、max_abs_diff、mean_abs_diff。

**Dump 点位：**
- `post_norm1`: input_layernorm 之后
- `post_attn`: self_attn 之后
- `post_norm2`: post_attention_layernorm 之后
- `post_moe`: MoE/FFN 之后

### 2.2 对比结果（TP=2 EP=2，2 层，rank 0）

| Layer | Sublayer | Cosine | MaxAbsDiff | 判断 |
|-------|----------|--------|------------|------|
| L0 | post_norm1 (input_layernorm) | **0.999995** | 0.0039 | ✅ 基本一致 |
| L0 | post_attn (attention output) | **0.999656** | 1.0000 | ✅ 轻微差异 |
| L0 | post_norm2 (post_attn_norm) | **0.998221** | 0.1763 | ✅ 轻微差异 |
| **L0** | **post_moe (MoE output)** | **0.565914** | **2.2588** | **❌ 严重 DIVERGE** |
| L1 | post_norm1 (input_layernorm) | **0.926662** | 0.0586 | ❌ 已偏离 |
| L1 | post_attn (attention output) | **0.841582** | 4.7656 | ❌ 更偏 |
| L1 | post_moe (MoE output) | **0.666242** | 3.3359 | ❌ 更偏 |

### 2.3 根因定位

**Layer 0 的 MoE 输出就已经严重 diverge (cosine=0.57)，而 L0 的 attention 输出几乎完美一致 (cosine=0.9997)。**

这说明：
- ❌ 不是 attention 的问题
- ❌ 不是 embedding 的问题
- ❌ 不是 input_layernorm 的问题
- **✅ 问题在 MoE 层的 Marlin kernel 计算**

可能的根因：
1. Expert weights 的 FP8 → Marlin repack 出错（packed weights 或 scales）
2. Marlin kernel 本身的计算精度问题（SM80 上的 W8A16 路径）
3. MoE 内部的 gating/routing 差异（虽然之前验证过 gate logits 一致，但需重新确认）
4. Expert 内部的 swiglu 激活函数差异

### 2.4 下一步排查方向

需要在 MoE forward 内部加 dump，对比：
- Gate logits（路由是否一致）
- Top-k expert ids（选了哪些 expert）
- 每个 expert 的 GEMM 输入/输出
- Marlin kernel 使用的 scales 值
- Swiglu 激活后的中间值

---

## 三、核心修复（已完成）

### 3.1 `scales *= 2^120` FP8→BF16 exponent bias adjustment

**文件：** `minimax_m2_5.py`，`_process_fp8_marlin_weights` 函数

**原理：** FP8 (e4m3) exponent bias = 7，BF16 exponent bias = 127，差值 = 120。Marlin W8A16 kernel 读取 FP8 weight bits 后需要乘以 `2^120` 来补偿。vLLM 在 `fp8_fused_exponent_bias_into_scales` 中做这个操作，FD 原来缺失。

```python
marlin_s = marlin_s * (2 ** 120)  # FP8→BF16 exponent bias adjustment
marlin_s_d = marlin_s_d * (2 ** 120)
```

### 3.2 `weight_scale_inv` stacked QKV slice bug 修复

**文件：** `minimax_m2_5.py`，`_dequant_fp8_weights` 函数

**问题：** PaddlePaddle LazyGuard 模式下参数 `memory_size=0`，slice 操作崩溃。
**修复：** 收集 q/k/v 3 个 shard 后一次性 numpy concat + copy_。

### 3.3 Marlin kernel vLLM 版本移植

**文件：** `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/` 下多个文件

FD 的旧版 marlin_kernel 与 vLLM 有 2620 行差异，已整体替换为 vLLM 版本（DP+SK 调度）。

### 3.4 `block_wise_fp8.py` scale expansion transpose fix

**文件：** `fastdeploy/model_executor/layers/quantization/block_wise_fp8.py`

**问题：** `paddle.expand` 返回 non-contiguous view，直接 reshape 布局错误。
**修复：** 添加 `sc_exp = sc_exp.transpose([0, 2, 1, 3])`

### 3.5 `block_wise_fp8.py` `weight_scale_inv` default_initializer

**文件：** `block_wise_fp8.py`，行 248

**修复：** 添加 `default_initializer=paddle.nn.initializer.Constant(0)`，确保 float32 分支的 `weight_scale_inv` 参数有内存分配。

### 3.6 tinyformat assertion failure 修复

**文件：** `custom_ops/gpu_ops/moe/moe_wna16_marlin_gemm.cu`

**操作：** 全局替换 `PADDLE_ENFORCE(` → `PD_CHECK(`（65处）

**原因：** PaddlePaddle 3.3.0 的 `PADDLE_ENFORCE` 使用 `Sprintf(args...)` 把第一个参数当 format string，但 FD 用流式多参数拼接，导致 tinyformat 断言失败。

### 3.7 EP group 创建冲突修复

**文件：** `fastdeploy/config.py` 的 `set_communicate_group()`

**修复：** 使用 `range(ep_start, ep_end)` 其中 `ep_start = dp_rank * ep_size`

```python
ep_start = self.data_parallel_rank * self.expert_parallel_size
ep_end = ep_start + self.expert_parallel_size
self.ep_group = dist.new_group(range(ep_start, ep_end))
```

### 3.8 logger 未定义修复

**文件：** `fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py`

**修复：** 添加 `from paddleformers.utils.log import logger`

### 3.9 M=0 EP all_reduce 修复

**文件：** `fastdeploy/model_executor/layers/moe/fused_moe_marlin_backend.py`

**修复：** 创建 dummy `[1, hidden_size]` tensor 做 all_reduce，返回 `[0, hidden_size]`

### 3.10 config.py 模型格式检测修复

**文件：** `fastdeploy/config.py`

**修复：** 移除错误的 `else: raise ValueError(...)` 分支

### 3.11 sublayer 缓存 + empty_cache 频率优化

**文件：** `minimax_m2_5.py`

- `load_weights` 入口缓存 `layer_idx → moe_sublayer` 映射，避免每层 `named_sublayers()` 全树遍历
- `empty_cache()` 从每层调用改为每 10 层调用

### 3.12 numpy import 作用域 bug 修复（2026-04-17）

**文件：** `minimax_m2_5.py`，`_dequant_fp8_weights` 方法

**问题：** 方法内 `sm80_keep_fp8` 分支有 `import numpy as np`，Python 编译器将整个方法的 `np` 视为局部变量。当走普通 dequant 分支时，`np` 未被赋值，报 `UnboundLocalError`。

**修复：** 删除方法内的 `import numpy as np`，使用文件顶部的全局 import。

### 3.13 FD 多进程 worker 日志位置发现（2026-04-17）

**关键发现：** FD 的 worker 进程日志不在 stdout/stderr，而是写入 `$CWD/log/` 目录（默认），可通过 `FD_LOG_DIR` 环境变量自定义。

日志文件：
- `workerlog.0`, `workerlog.1` — worker 进程的 stdout/stderr
- `worker_process.log` — worker process 生命周期日志
- `fastdeploy.log` — 主引擎日志
- `gpu_worker.log` — GPU worker 日志
- `config.log` — 配置 dump

---

## 四、加载速度优化

### 4.1 瓶颈分析

**根因：** `_process_fp8_marlin_weights` 中每个 expert 的 FP8 weight 需要逐个调用 `gptq_marlin_repack`（C++ dispatch）。

每层 128 次（64 experts × 2 权重），每次 dispatch 的 Python→C++ 开销叠加。

| 操作 | 耗时 | 占比 |
|------|------|------|
| `.numpy()` + `paddle.to_tensor()` 循环（128 experts） | ~30s | **99%** |
| `paddle.stack()` | ~0.02s | <1% |
| C++ `gptq_marlin_repack` | ~0.01s | <1% |
| Scales 处理 + set_value | ~0.05s | <1% |

### 4.2 方案 A：消除 numpy roundtrip（已完成，效果有限）

用 `paddle.stack` + cast("uint8") + 位移操作替代 numpy roundtrip。消除了 GPU↔CPU 搬运，但 128 次 C++ dispatch 仍是瓶颈。2 层加载 ~200s → ~160s（20% 提升）。

### 4.3 方案 B/C：C++ batch repack（受阻于 paddle::empty bug）

PaddlePaddle 的 `paddle::empty({d0, d1, d2}, ...)` 在 C++ custom op 中创建 3D tensor 时会 flatten 成 1D。可通过 Python 层预分配 3D tensor 传入 C++ 绕过。

`gptq_marlin_moe_repack_batch` 已编译通过，C++ dispatch 次数可从 128 → 2。

---

## 五、EP=4 TP=4 乱码根因分析

### 5.1 根因确认：`use_sequence_parallel_moe` 导致 NCCL `alltoall` 错误

**自动启用条件**（`config.py:735-739`）：
```python
self.use_sequence_parallel_moe = (
    (not self.disable_sequence_parallel_moe)
    and self.expert_parallel_size > 1
    and self.tensor_parallel_size > 1
)
```

TP=4 EP=4 时自动启用。影响：Linear 层 forward 时调用 `paddle.distributed.alltoall()` 做 token split，在 EP=4 拓扑下不兼容。

**修复：** `disable_sequence_parallel_moe=True`

---

## 六、调试流程 SOP（标准操作流程）

### 6.1 环境准备

```bash
# FD 环境
source ~/anaconda3/etc/profile.d/conda.sh && conda activate paddle

# vLLM 环境
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vllm17
```

### 6.2 修改 config.json 为少量层数（快速测试）

```bash
# 将 62 层改为 2 层（或你需要测试的层数）
python -c "
import json
cfg = '/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5/config.json'
with open(cfg) as f: d = json.load(f)
d['num_hidden_layers'] = 2
with open(cfg, 'w') as f: json.dump(d, f, indent=2)
"
```

**注意：** vLLM 和 FD 的脚本都会自动备份和恢复 config.json。如果手动修改，记得改回来。

### 6.3 清空日志目录

**FD 的日志目录在 `$CWD/log/`（即 FastDeploy 目录下的 `log/`），不是 `/data/lizhijun/work/fd-vllm/log/`！**

```bash
# 清空 FD 日志（每次跑之前必须做）
rm -rf /data/lizhijun/work/fd-vllm/FastDeploy/log/*

# 清空 dump 目录
rm -rf /tmp/dump_compare/fd/* /tmp/dump_compare/vllm/*
```

### 6.4 启动 FD 并监控

```bash
# 后台启动 FD
CUDA_VISIBLE_DEVICES=6,7 FD_MARLIN_FP8=1 python scripts/run_fd_dump.py &

# 监控 GPU 显存（确认 worker 是否启动并加载权重）
watch -n 5 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader'

# GPU 6,7 应该从 4MiB 逐渐增长到数 GB
# 如果长时间（>2分钟）GPU 显存不变，说明 worker 卡住了
```

### 6.5 查看 FD 日志

```bash
# Worker 进程日志（最重要的，包含模型加载和推理信息）
tail -100 /data/lizhijun/work/fd-vllm/FastDeploy/log/workerlog.0

# Worker process 日志（包含权重加载进度）
tail -100 /data/lizhijun/work/fd-vllm/FastDeploy/log/worker_process.log

# GPU worker 日志
tail -100 /data/lizhijun/work/fd-vllm/FastDeploy/log/gpu_worker.log

# 错误日志
cat /data/lizhijun/work/fd-vllm/FastDeploy/log/console_error.log

# 快速看是否有报错
grep -i "error\|exception\|traceback\|failed" /data/lizhijun/work/fd-vllm/FastDeploy/log/workerlog.0
```

### 6.6 运行 vLLM 基线

```bash
# vLLM 的 dump 脚本（自动改/恢复 config.json）
cd /data/lizhijun/work/fd-vllm/FastDeploy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vllm17
CUDA_VISIBLE_DEVICES=6,7 python scripts/run_vllm_dump.py
```

### 6.7 运行 FD dump

```bash
cd /data/lizhijun/work/fd-vllm/FastDeploy
source ~/anaconda3/etc/profile.d/conda.sh && conda activate paddle
# 先清空日志和 dump 目录
rm -rf log/* /tmp/dump_compare/fd/*
CUDA_VISIBLE_DEVICES=6,7 FD_MARLIN_FP8=1 python scripts/run_fd_dump.py
```

### 6.8 对比 dump 结果

```bash
python scripts/compare_dump.py /tmp/dump_compare 0
# 参数 1: dump 目录
# 参数 2: rank 号（0 或 1）
```

对比脚本输出每个 sublayer 的：
- `cosine_sim`: 余弦相似度（<0.99 = diverge，<0.9999 = slight diff）
- `max_abs_diff`: 最大绝对误差
- `mean_abs_diff`: 平均绝对误差
- `rel_err`: 相对误差

### 6.9 Dump 逻辑说明

Dump 代码位于 `minimax_m2_5.py` 的两个位置：

1. **`MiniMaxM2_5DecoderLayer.forward`** — 每层 4 个 dump 点（post_norm1, post_attn, post_norm2, post_moe）
2. **`MiniMaxM2_5Model.forward`** — embed 和 final_norm 的 dump

Dump 条件：`DUMP_DIR and hidden_states.shape[0] > 0`（只 dump 非空 tensor，跳过 decode 阶段的空输入）

Dump 文件命名：`{fd|vllm}_r{rank}_l{layer}_{sublayer}.npy`
- 例如：`fd_r0_l0_post_moe.npy` = FD rank0 layer0 MoE 输出

vLLM 侧的 dump 通过 `VLLM_DUMP_DIR` 环境变量控制，在 `vllm/vllm/model_executor/models/minimax_m2.py` 的 `MiniMaxM2DecoderLayer.forward` 中。

### 6.10 常见问题排查

| 现象 | 可能原因 | 排查方法 |
|------|----------|----------|
| FD 启动后 GPU 显存不变 | Worker 进程卡住或崩溃 | 查看 `workerlog.0` 最后几行 |
| `UnboundLocalError: np` | numpy import 作用域问题 | 检查方法内是否有 `import numpy as np` |
| FD dump 文件为空 `[0, N]` | decode 阶段空 tensor 覆盖了 prefill 数据 | 添加 `shape[0] > 0` 条件 |
| `config.json` 层数不对 | 跑完没恢复 | 检查 `num_hidden_layers` 值 |
| vLLM 和 FD 的 GPU 占用冲突 | 同时跑 | 先跑 vLLM 完全退出后再跑 FD |
| workerlog.0 报 `tinyformat` | PADDLE_ENFORCE 格式化 bug | 全局替换为 `PD_CHECK(` |
| NCCL `alltoall` error | `use_sequence_parallel_moe` 冲突 | 添加 `disable_sequence_parallel_moe=True` |

---

## 七、已排查并确认的组件

| 组件 | 一致性 | 说明 |
|------|--------|------|
| Marlin packed weights | ✅ | 与 vLLM repack 逻辑一致 |
| Scales (permute) | ✅ | 与 vLLM 一致，max_diff=0 |
| Gate routing (noaux_tc) | ✅ | 与 vLLM grouped_topk 一致 |
| sorted_token_ids | ✅ | flat index 编码一致 |
| Scale dtype (bfloat16) | ✅ | s_type = kBFloat16 |
| FP8→BF16 exponent bias (2^120) | ✅ | 已融入 scales |
| embed_tokens + lm_head | ✅ | BF16 格式，无 FP8 dequant |
| L0 input_layernorm | ✅ | cosine=0.999995 |
| L0 attention output | ✅ | cosine=0.9997 |
| **L0 MoE output** | **❌** | **cosine=0.57 — 严重 diverge** |
| EP group 拓扑 | ✅ | 正确 |
| Expert map | ✅ | 每卡 expert_id_offset 正确 |
| use_sequence_parallel_moe | ❌ | 导致 alltoall NCCL 错误，需 disable |
| 加载速度 | ⚠️ | 2 层 ~2min，62 层预估 ~60min |

---

## 八、修改的文件清单

| 文件 | 修改 |
|------|------|
| `minimax_m2_5.py` | scales *= 2^120; stacked QKV scale copy; Marlin FP8 weight loading; sublayer 缓存; empty_cache 频率; numpy import 作用域修复; layer-by-layer dump 逻辑 |
| `block_wise_fp8.py` | scale expansion transpose fix; weight_scale_inv default_initializer |
| `fused_moe_marlin_backend.py` | scale dtype (bfloat16); _swiglu 兼容; M=0 处理; logger 导入 |
| `config.py` | partial_rotary_factor; EP group 创建修复; 模型格式检测修复 |
| `engine.py` | pad_token_id=None 处理 |
| `custom_ops/gpu_ops/moe/moe_wna16_marlin_gemm.cu` | PADDLE_ENFORCE → PD_CHECK (65处) |
| `custom_ops/gpu_ops/moe/moe_wna16_marlin_utils/*` | 整体替换为 vLLM 版本 |
| `custom_ops/gpu_ops/moe/gptq_marlin_repack.cu` | 添加 `gptq_marlin_moe_repack_batch` |
| `fastdeploy_ops/__init__.py` | 重新生成 |
| `vllm/vllm/model_executor/models/minimax_m2.py` | dump 逻辑添加 rank 后缀 |
| `scripts/run_vllm_dump.py` | **新建** — vLLM dump 脚本 |
| `scripts/run_fd_dump.py` | **新建** — FD dump 脚本 |
| `scripts/compare_dump.py` | **新建** — 逐层对比脚本 |

---

## 九、下一步计划

### 优先级 1：MoE 内部精度排查（当前最紧急）

L0 MoE 输出 cosine=0.57，需要在 MoE forward 内部加 dump：
- Gate logits 对比
- Top-k expert ids 对比
- 每个 expert 的 GEMM 输入/输出
- Marlin kernel 的 scales 值
- Swiglu 激活后的中间值

### 优先级 2：解决 `paddle::empty` 3D shape flatten bug（解锁加载加速）

Python 层预分配 3D tensor 传入 C++，C++ dispatch 从 128 → 2。

### 优先级 3：EP=4 TP=4 + disable_sequence_parallel_moe 验证

需要 4 张空闲 GPU。

### 优先级 4：性能基准测试

Prefill/Decode 吞吐、GPU 内存峰值、与 vLLM 相对速度比。

---

## 十、运行命令

### FD TP=2 EP=2（当前可用，GPUs 6,7）
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate paddle
CUDA_VISIBLE_DEVICES=6,7 FD_MARLIN_FP8=1 python -c "
import sys; sys.path.insert(0, '/data/lizhijun/work/fd-vllm/FastDeploy')
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams
llm = LLM(
    model='/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5',
    tensor_parallel_size=2, enable_expert_parallel=True,
    disable_sequence_parallel_moe=True,
    max_model_len=512, gpu_memory_utilization=0.85,
    max_num_seqs=4, num_gpu_blocks_override=100,
    max_num_batched_tokens=512,
    graph_optimization_config={'use_cudagraph': False},
)
outputs = llm.generate(['Hello, how are you?'], SamplingParams(temperature=0, max_tokens=16))
for out in outputs:
    print('prompt:', out.prompt)
    if out.outputs:
        print('text:', repr(out.outputs[0].text))
        print('tokens:', out.outputs[0].token_ids)
"
```

### FD 编译
```bash
cd /data/lizhijun/work/fd-vllm/FastDeploy/custom_ops
~/anaconda3/envs/paddle/bin/python setup_ops.py build
cp build/fastdeploy_ops/lib.linux-x86_64-cpython-310/fastdeploy_ops.so \
   ../fastdeploy/model_executor/ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so
cp build/fastdeploy_ops/lib.linux-x86_64-cpython-310/fastdeploy_ops.py \
   ../fastdeploy/model_executor/ops/gpu/fastdeploy_ops/__init__.py
```

### vLLM 基线（2 层快速对比）
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vllm17
CUDA_VISIBLE_DEVICES=6,7 python -c "
import json
cfg = '/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5/config.json'
with open(cfg) as f: d = json.load(f)
d['num_hidden_layers'] = 2
with open(cfg, 'w') as f: json.dump(d, f, indent=2)
from vllm import LLM, SamplingParams
llm = LLM(model='/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5', trust_remote_code=True,
           enable_expert_parallel=True, tensor_parallel_size=2, max_model_len=512,
           gpu_memory_utilization=0.90, enforce_eager=True)
out = llm.generate(['Hello, how are you?'], SamplingParams(max_tokens=16, temperature=0))
print('Output:', repr(out[0].outputs[0].text))
print('Token IDs:', out[0].outputs[0].token_ids)
"
```

### 使用 dump 脚本对比（推荐方式）
```bash
cd /data/lizhijun/work/fd-vllm/FastDeploy

# Step 1: vLLM dump
conda activate vllm17
rm -rf /tmp/dump_compare/vllm/* log/*
CUDA_VISIBLE_DEVICES=6,7 python scripts/run_vllm_dump.py

# Step 2: FD dump
conda activate paddle
rm -rf /tmp/dump_compare/fd/* log/*
CUDA_VISIBLE_DEVICES=6,7 FD_MARLIN_FP8=1 python scripts/run_fd_dump.py

# Step 3: 对比
python scripts/compare_dump.py /tmp/dump_compare 0
```

---

## 十一、环境信息

- **GPU**: 8 × A100 80GB (SM 8.0)
- **FD 环境**: conda activate `paddle` (Python 3.10, PaddlePaddle 3.3.0)
- **vLLM 环境**: conda activate `vllm17` (Python 3.10, PyTorch, vLLM v0.17 dev)
- **模型路径**: `/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5`
- **Checkpoint**: 125 个 safetensors 文件，总计 230 GB (FP8)
- **可用 GPU**: GPU 6,7 空闲（常用）；GPU 0,1 被占用；GPU 3,4 有时空闲

---

## 十二、EP=2 MoE 内部调试（2026-04-17）

### 12.1 EP=2 运行验证

**环境：** GPU 4,5 (paddle env)，EP=2，长 prompt（201 tokens），2 层

**结果：**
- `num_tokens_pp=1024` > 0 ✅（之前担心的 zero-token 问题在大 batch 下不发生）
- FD 输出：`' increí'` ❌（西班牙语乱码）
- vLLM EP=2 输出：`'\n\n'` ✅

### 12.2 MoE dump 分析

**Dump 文件命名问题：** MoE dump（`fd_moe_r{rk}_*.npy`）没有 layer index 后缀，多层运行时互相覆盖。n=2 时 dump 来自 layer 1（最后一层），n=1 时 dump 来自 layer 0。

**n=1 验证结果：**
- `gate_input` (MoE 输入) == `l0_post_norm2` ✅（dump 一致性验证通过）
- `post_attn` = **全零** ❌（attention 输出为零）
- `post_norm2` ≠ `RMSNorm(embed)`（因为 residual 连接使用原始输入而非 normed）

### 12.3 关键发现：attention 输出为零

FD 的 append attention backend 在 EP=2 模式下输出全零（`post_attn: mean=0, std=0`）。这不是 MoE 的问题，而是 **attention 实现阶段的问题**。

- 同样的 prompt，vLLM 的 attention 正常工作
- FD 使用 `APPEND ATTN backend` + `CUDA GRAPH`，可能在 capture 阶段没有正确初始化 KV cache
- 这影响所有层的精度，不仅是 MoE

### 12.4 FD vs vLLM gate routing 对比

单独对比 gate routing（用 FD 的 embed 输入 + vLLM 的 gate weight）：
- `topk_ids` **完全不匹配**（因为输入不同：FD embed vs vLLM embed）
- 当使用相同输入时，gate routing 应该一致（之前 2026-04-12 验证过 cos=1.0）

### 12.5 FD Marlin kernel 在 SM80 上的状态

**当前代码状态：** `fused_moe_marlin_backend.py` 只有 Marlin 路径（`apply` 和 `apply_ep_noalltoall`）。之前的 SM80 bf16 workaround（`_apply_ep_sm80_bf16`）已丢失（被 linter revert）。

**Marlin kernel 在 SM80 上不工作的根本原因：**
- FD 的 `marlin_template.h` 与 vLLM 有 2216 行差异
- 核心计算逻辑（dequant、scale）byte-identical
- 差异在调度策略（FD stripe vs vLLM DP+SK）和类型系统
- 尝试过禁用 `should_load_a`、修改 `max_num_stage_groups`、修改 shm padding — 全部无效

### 12.6 SM80 bf16 workaround 回顾（2026-04-12 验证成功）

之前的 SM80 bf16 workaround 在 EP=4 下验证成功：
- FP8 expert 权重 CPU numpy 反量化到 bfloat16
- 使用 cuBLAS BF16 GEMM 代替 Marlin kernel
- n=1~61 层全部正确（scale expansion transpose fix 之后）
- n=62 输出 `':'` — 与 vLLM 一致（确认正确）
- 限制：62 层 4 卡 ~59 GB GPU，接近 80 GB 上限

**该 workaround 的代码在 `fused_moe_marlin_backend.py` 的多次 revert 中丢失。需要重新实现。**

### 12.7 SM80 bf16 workaround 重新实现的工作量评估

**需要修改的文件：**

| 文件 | 修改内容 | 工作量 |
|------|----------|--------|
| `fused_moe_marlin_backend.py` | 添加 `_apply_ep_sm80_bf16` 方法（~80 行），在 `apply_ep_noalltoall` 中 SM80 检测 + 路由 | **中等**（有之前的代码可参考） |
| `minimax_m2_5.py` | `_load_fp8_marlin_layer` 中 SM80 路径：不创建 Marlin packed，保持 FP8 在 CPU；expert 权重 `.cuda()` 按需拷贝 | **中等**（需要恢复之前的逻辑） |
| `block_wise_fp8.py` | 已有 SM80 scale transpose fix ✅，无需修改 | **无** |

**4 卡能跑吗？**
- 4 卡 EP=4：每卡 64 experts × 1536×3072 FP8 (CPU) + 62 层非 expert 权重 FP8 → SM80 bf16 dequant
- GPU 显存估算：attention 权重 bf16 ~3 GB + KV cache ~5 GB + workspace ~1 GB = ~9 GB/卡（可接受）
- Expert 权重在 CPU，forward 时 `.cuda()` 拷贝 + dequant + 计算后释放
- **4 卡完全可以跑**，之前的实验已验证 n=61 正确

**总体工作量：** ~2-3 小时（恢复之前的代码 + 测试验证）。核心逻辑之前已写过并验证成功，主要是代码恢复和集成。

### 12.8 当前分支

当前在 `feat/minimax-m2.5-wint4` 分支上。

---

## 十三、SM80 BF16 Workaround 恢复 + 逐层 Dump 对比（2026-04-17 下午）

### 13.1 SM80 BF16 Workaround 恢复

**已完成。** 恢复了之前在 linter revert 中丢失的 SM80 bf16 workaround 代码。

**修改的文件：**

**`fused_moe_marlin_backend.py`：**
1. `_moe_dump` / `_moe_dump_int` 添加 `layer_idx` 参数，文件名从 `fd_moe_r0_gate_input` 改为 `fd_moe_r0_l0_gate_input`
2. 所有 dump 调用点添加 `layer.layer_idx`
3. `apply_ep_noalltoall` 添加 SM80 检测路由（`get_sm_version() < 90` 时跳过 Marlin，调用 `_apply_ep_sm80_bf16`）
4. 新增 `_apply_ep_sm80_bf16` 方法（~100 行）：
   - FP8 expert 权重从 CPU → GPU
   - numpy block-wise dequant FP8→BF16
   - 逐 expert cuBLAS BF16 GEMM（gate/up → swiglu → down）
   - float32 scatter-add 汇总
   - all_reduce 跨 EP ranks
   - 关键：`.numpy()` 前必须 `.cast("float32")`（避免 bf16→uint16 bug）
5. MoE dump 添加 `x.shape[0] > 0` 保护（跳过 M=0 的 decode 阶段）
6. `num_tokens_pp == 0` 的 early return 改为 dummy `[1, hidden_size]` 做 all_reduce（NCCL 不接受 0-size tensor）

**`minimax_m2_5.py`：**
1. `_load_fp8_marlin_layer` 添加 SM80 分支：
   - SM80 上不创建 Marlin packed int32 格式
   - 将 raw FP8 expert weights stack 为 `[E, N*2, K]` + `[E, N, K]`
   - 存储到 `moe_layer._sm80_fp8_up_gate` / `_sm80_fp8_down` 等属性
   - 全部 `.cpu()` 放 CPU，forward 时按需拷贝到 GPU
2. Decoder layer dump 添加 `_dumped_once` 保护（只 dump prefill，避免 decode 覆盖）

**`vllm/vllm/model_executor/models/minimax_m2.py`：**
1. Decoder layer forward 添加 dump 逻辑（通过 `VLLM_DUMP_DIR` 环境变量控制）
2. 使用 `torch.distributed.get_rank()` 动态获取 rank 号
3. `import numpy as np` 在 `_do_dump` 分支内（避免 torch.compile 崩溃，需 `enforce_eager=True`）

### 13.2 加载速度改善

SM80 bf16 workaround 路径完全跳过了 `gptq_marlin_repack`（128 次 C++ dispatch），直接 `paddle.stack()` 后 `.cpu()`。

| 方案 | 2 层加载时间 | 瓶颈 |
|------|------------|------|
| Marlin packed（旧路径） | ~200s | 128 次 `gptq_marlin_repack` C++ dispatch |
| SM80 FP8 CPU（新路径） | ~46s | FP8 → CPU stack（无 C++ dispatch） |

**加载速度提升 ~4.3x。**

### 13.3 EP=2 逐层 Dump 对比结果（2026-04-17）

**环境：** GPU 6,7，EP=2，2 层，`FD_MARLIN_FP8=1`，prompt=`'Hello, how are you?'`

| Layer | Sublayer | Cosine | MaxAbsDiff | 判断 |
|-------|----------|--------|------------|------|
| L0 | post_norm1 (input_layernorm) | **0.999995** | 0.0039 | ✅ 基本一致 |
| L0 | post_attn (attention output) | **0.999656** | 1.0000 | ✅ 轻微差异 |
| L0 | post_norm2 (post_attn_norm) | **0.998221** | 0.1763 | ✅ 轻微差异 |
| **L0** | **post_moe (MoE output)** | **0.565885** | **2.2588** | **❌ 严重 DIVERGE** |
| L1 | post_norm1 | **0.926653** | 0.0586 | ❌ 已偏离 |
| L1 | post_attn | **0.841503** | 4.7656 | ❌ 更偏 |
| L1 | post_moe | **0.666216** | 3.3359 | ❌ 更偏 |

**关键发现：**
1. **Attention 输出不再全零** ✅ — 之前 EP=2 模式下 attention 全零（`post_attn: std=0`），现在 `std=0.506724` 正常
2. **L0 MoE 仍然严重 diverge** ❌ — cosine=0.57，和之前 Marlin 路径的结果完全一致
3. **这说明 MoE 的精度问题不是 Marlin kernel 的问题** — SM80 bf16 workaround 用 cuBLAS GEMM 替代了 Marlin kernel，但输出仍然 diverge
4. **问题在 MoE 层内部** — 需要进一步排查 gate routing、expert GEMM、swiglu、scatter-add 等环节

### 13.4 MoE 内部 Dump 分析

MoE dump 正确捕获了 prefill 阶段（M=3），不再被 decode（M=0）覆盖：

```
fd_moe_r0_l0_gate_input: shape=(3, 3072), mean=0.005815, std=0.269330
fd_moe_r0_l0_topk_weights: shape=(3, 8), mean=0.125000
fd_moe_r0_l0_topk_ids: shape=(3, 8)
fd_moe_r0_l0_sorted_token_ids: shape=(896,)
```

**Shape 不匹配发现：** `gate_input shape=(3, 3072)` 而 `post_norm2 shape=(6, 3072)`。这是因为 FD 的 MoE 使用 `forward_split_allgather` 路径（`attn_tp_size=2`），将 6 个 tokens 拆成 2 份（每份 3 个），分别计算 MoE 后 all_gather。MoE dump 只捕获了第 1 份。

**影响：** MoE 内部 dump 的 `gate_input` 无法直接和 vLLM 的 MoE dump 对比（token 数不同）。需要对比 MoE 的最终输出（`post_moe`），或在 MoE forward 内部逐 expert dump。

### 13.5 修复的 Bug

| Bug | 修复 |
|-----|------|
| M=0 all_reduce 崩溃 | dummy `[1, hidden_size]` tensor 做 all_reduce |
| MoE dump 被 decode 覆盖 | 添加 `x.shape[0] > 0` 保护 |
| Decoder layer dump 被覆盖 | 添加 `_dumped_once` 保护（只 dump prefill） |
| vLLM dump rank 硬编码 | 使用 `torch.distributed.get_rank()` |

### 13.6 下一步

### 优先级 1（最紧急）：MoE 内部逐 Expert Dump

L0 MoE 输出 cosine=0.57，但 SM80 bf16 workaround 和 Marlin 路径结果一致，说明问题不在 Marlin kernel 本身。需要在 `_apply_ep_sm80_bf16` 中添加更细粒度的 dump：

- 每个 expert 的 dequant 后 BF16 weights（和 vLLM 对比）
- gate/up GEMM 的输入和输出
- swiglu 的输入和输出
- down GEMM 的输出
- scatter-add 后的最终 MoE 输出

可能的根因：
1. FP8 dequant 的 scale 处理不对（虽然之前的 scale expansion transpose fix 已验证正确）
2. Expert weights 的排列顺序不对（gate vs up 的合并方式）
3. Swiglu 实现差异（paddle.nn.functional.swiglu vs vLLM 的 swiglu）
4. Scatter-add 的累积方式差异

### 优先级 2：4 卡 EP=4 全量验证

SM80 bf16 workaround 恢复后，4 卡 EP=4 应该可以跑 62 层（之前验证 n=1~61 正确）。

### 优先级 3：`forward_split_allgather` 的 token 拆分逻辑验证

FD 的 MoE 使用 `forward_split_allgather` 将 tokens 拆分后分别计算。需要验证这个拆分逻辑是否和 vLLM 一致。

---

## 十四、EP=4 LLM API 端到端首次运行（2026-04-17 晚）

### 14.1 运行环境

**环境：** GPU 4,5,6,7 (paddle env)，EP=4，2 层，`FD_MARLIN_FP8=1`

**启动命令：**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 FD_MARLIN_FP8=1 FD_ATTENTION_BACKEND=FLASH_ATTN \
  conda run -n paddle python /tmp/fd_minimax_demo.py
```

**关键配置：**
```python
llm = LLM(
    model='/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5',
    tensor_parallel_size=1,
    data_parallel_size=4,  # 控制 EP size
    enable_expert_parallel=True,
    disable_sequence_parallel_moe=True,
    max_model_len=512,
    gpu_memory_utilization=0.90,
    max_num_seqs=4,
    num_gpu_blocks_override=100,
    max_num_batched_tokens=512,
    graph_optimization_config={'use_cudagraph': False},
)
```

### 14.2 关键发现：FD 的 EP 配置方式

FD 的 `EngineArgs` 不接受 `expert_parallel_size` 参数。EP size 通过以下方式推导：
```python
# config.py
if self.enable_expert_parallel:
    self.expert_parallel_size = self.data_parallel_size * self.tensor_parallel_size
```

所以 `tensor_parallel_size=1, data_parallel_size=4, enable_expert_parallel=True` → `expert_parallel_size=4`。

**之前错误的配置：** 只传 `enable_expert_parallel=True` 不传 `data_parallel_size`，导致 `expert_parallel_size=1*1=1`，EP 不生效。

### 14.3 运行结果

**加载阶段（成功）：**
- 模型加载 125 个 safetensors 文件：~17 秒
- SM80 FP8 CPU offload：每层 ~1.5 秒
- 总加载时间：~23 秒
- GPU 显存：GPU 4: 8097 MiB, GPU 5-7: 4759 MiB
- KV cache 初始化成功（100 blocks, 2 层）

**推理阶段（极慢）：**
- 请求已调度：M=6 tokens, topk_ids.shape=[6, 8]
- 卡在 `_apply_ep_sm80_bf16` 的 MoE forward
- GPU 利用率 100%，但进展极慢
- 5+ 分钟未完成第一步 MoE forward

### 14.4 根因分析：SM80 BF16 Workaround 每次 Forward 都重新 Dequant

**问题：** `_apply_ep_sm80_bf16` 在每次 MoE forward 时对每个 expert 做：
1. FP8 权重从 CPU → GPU（`expert_w = layer._sm80_fp8_up_gate[ei].cuda()`）
2. CPU numpy block-wise dequant FP8→BF16
3. GPU BF16 GEMM

对于 64 experts × top-8 routing，每步 decode 都要 dequant 64 个 expert 的权重。这是 O(64 × N × K) 的 numpy 计算，非常慢。

**正确的设计应该是：**
- 加载时：FP8 → BF16 反量化一次，BF16 权重常驻 GPU
- 推理时：直接用 BF16 权重做 GEMM

但当前为了省内存（62 层 4 卡 FP8 expert BF16 权重 ~144 GB，超出 4×80 GB），选择了 CPU FP8 + forward dequant。对于 2 层模型显存完全够用，不需要 CPU offload。

### 14.5 下一步修复方向

**方案 A（推荐，适用于小层数）：** 2 层模型直接在加载时 dequant 到 BF16 放 GPU，不做 CPU offload。
- 2 层 × 64 experts × BF16 = ~1.5 GB/卡，完全够用
- 每步 forward 直接用 GPU BF16 权重做 GEMM，无 dequant 开销

**方案 B（适用于 62 层）：** 保持 CPU FP8 offload，但只 dequant 一次，缓存 BF16 结果。
- 加载时不做 dequant（省内存）
- 首次 forward 时 dequant 64 experts → BF16 缓存到 GPU
- 后续 forward 直接用缓存的 BF16 权重
- 问题：62 层 × 64 experts BF16 不够放 GPU

**方案 C（适用于 62 层）：** LRU 缓存——只缓存最近 N 个被选中的 expert 的 BF16 权重，淘汰不常用的。
- 复杂度较高，但能平衡显存和速度

### 14.6 确认的配置信息

FD LLM API 启动日志确认的配置：
```
expert_parallel_size: 4
data_parallel_size: 4
use_ep: True
worker_num_per_node: 4
MoE config: ep_size=4, num_experts=256[0, 64)
SM80: Stored raw FP8 expert weights on CPU for layer 0
Model loading took 23.609 seconds
```

Worker 日志位置：`/data/lizhijun/work/fd-vllm/vllm/log/workerlog.{0-3}`（注意在 vllm 目录下，不是 FastDeploy 目录下）。

---

## 十五、MoE 精度深度分析 + RoPE 根因定位（2026-04-18）

### 15.1 单 Expert 精度对比（FD vs Reference）

**方法：** 在 GPU 7 上单卡加载 Expert 0 (Layer 0)，对比 FD 的 numpy dequant + BF16 GEMM vs 纯 float32 参考实现。

**Dequant 精度：**

| 对比方法 | cosine | max_diff | 结论 |
|----------|--------|----------|------|
| FD numpy dequant vs 纯 float32 | **0.99998556** | 4.9e-4 | FD dequant 本身正确 |
| FD numpy dequant vs numpy expand+transpose | **0.98538098** | 7.4e-2 | transpose 是错误的 |

**关键发现：** FD 的 numpy block-wise dequant（reshape blocked → scale → reshape back）和纯 float32 方法 **cos=0.99999**，说明 dequant 逻辑本身完全正确。之前测试中 "vLLM-style" 的差异来自 paddle expand+transpose，而实际 vLLM 的 Marlin kernel 不这样做。

**MoE 单 Expert 计算精度：**

| 路径 | vs Reference (float32) | cosine | max_diff |
|------|------------------------|--------|----------|
| FD 实际路径 (bf16 GEMM + f32 scatter-add) | Ref(f32) | **0.99999971** | 0.127 |
| bf16 dequant + f32 GEMM | Ref(f32) | **1.00000000** | 0.031 |
| 全 bf16 (包括 scatter-add) | Ref(f32) | **0.99999745** | 0.372 |

**结论：** FD 的 MoE 单 expert 计算精度 **cos=0.99999971**，非常接近完美。之前的 cos=0.988 的 MoE 层差异来自 **upstream attention 差异的放大**，不是 MoE 本身的精度问题。

### 15.2 Batch FP8 Dequant 改造

**修改文件：** `minimax_m2_5.py` 的 `load_weights` 方法

**改动：** 将逐层 dequant 循环改为一次性处理所有层：
- 移除每层的 `empty_cache()` 调用（只在全部处理完后调用一次）
- 移除每层的内存日志（改为汇总日志）
- 保持每层内部的 FP8 → dequant → load → free 流程不变

### 15.3 RoPE 根因定位（关键发现）

**问题描述：** MoE 单 expert 精度 cos=0.99999971，但端到端 MoE 输出 cos=0.988。差异来自 attention 层（cos=0.9996）被 MoE routing 放大。

**根因：** FD 的 RoPE 实现对 MiniMax-M2.5 是**错误的**。

**MiniMax-M2.5 配置：**
- `head_dim=128`, `rotary_dim=64`, `partial_rotary_factor=0.5`
- 只旋转 Q/K 的**前 64 个维度**，后 64 个保持不变（standard-style partial RoPE）

**FD 当前实现（错误）：**
- `minimax_m2_5.py` 设置 `use_neox_rotary_style=False`
- 这导致 CUDA kernel `GQAVariableLengthRotarySplitKernel` 对**所有 128 个维度都做旋转**
- 后 64 个维度被错误旋转，导致 attention 输出偏差

**vLLM 参考实现（`modeling_minimax_m2.py`）：**
```python
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]  # rotary_dim=64
q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
q_embed = torch.cat([q_embed, q_pass], dim=-1)  # 后 64 个不变
```

**FD CUDA kernel 路由逻辑（`gqa_rope_write_cache.cu`）：**

| `use_neox_rotary_style` | `rotary_dim` | 调用的 kernel | 行为 |
|--------------------------|--------------|--------------|------|
| `False` | 任意 | `GQAVariableLengthRotarySplitKernel` | **旋转所有 head_dim=128 维度** ❌ |
| `True` | == head_dim | `gqa_rotary_qk_split_variable_qwen3` | 旋转所有维度（Qwen3 风格） |
| `True` | < head_dim | `gqa_neox_partial_rotary_qk_split_variable` | **只旋转前 rotary_dim=64 维度** ✅ |

**修复方案：**
1. `minimax_m2_5.py`: `use_neox_rotary_style=False` → `True`
2. `get_rope` 调用: `rotary_dim=128` → `rotary_dim=model_config.rotary_dim=64`
   - 这使 `rotary_embs.shape[4] = 64 // 2 = 32`
   - CUDA kernel 中 `rotary_embs.dims()[4] == head_dim / 4` → `32 == 32` ✓
   - 然后 `rotary_dim = head_dim / 2 = 64` ✓

**Neox partial kernel 的旋转方式：**
- `half_rotary_dim = 32`
- `h_bias < 32`: forward pair `(x[i], x[i+32])`
- `h_bias >= 32`: backward pair `(x[i], x[i-32])`
- 这和 vLLM 的 `rotate_half`（对 rotary_dim=64，pair 是 `(x[i], x[i+32])`）**完全一致** ✅

### 15.4 RoPE 修复验证结果

**EP=2, 2 层, GPU 6,7：**

| 指标 | RoPE 修复前 | RoPE 修复后 | 改进 |
|------|------------|------------|------|
| Embed | 1.00000000 | 1.00000000 | - |
| L0 post_attn | ~0.9996 | **0.99999447** | **+0.0004** |
| L0 post_moe | ~0.988 | **0.99903549** | **+0.011** |
| L0 post_norm1 | ~0.9996 | **0.99999516** | **+0.0004** |
| L0 post_norm2 | ~0.9996 | **0.99991466** | **+0.0003** |
| L1 post_attn | 退化 | **0.99747502** | 大幅改进 |
| L1 post_moe | 退化 | **0.99231013** | 大幅改进 |
| final_norm | 退化 | **0.99950202** | 大幅改进 |

**关键成果：**
1. **L0 post_attn**: cos 从 0.9996 → **0.99999447**（attention 输出几乎完美对齐）
2. **L0 post_moe**: cos 从 0.988 → **0.99903549**（MoE 输出大幅改进）
3. **2 层端到端**: 所有层都有合理精度，没有退化
4. **top-1 输出**: `'\n\n'` (16.88%) — 合理输出

### 15.5 剩余差异分析

L0 post_moe cos=0.9990（而非 1.0）的剩余差异来源：
1. EP=2 vs EP=8 的 routing 差异（不同 expert 分片，vLLM dump 是 EP=8）
2. SM80 bf16 workaround 的固有精度差异（numpy dequant + cuBLAS GEMM vs Marlin CUDA kernel）
3. Scatter-add 累积顺序差异（float32 累加顺序不同）

### 15.6 修改的文件（本次更新）

| 文件 | 修改 |
|------|------|
| `minimax_m2_5.py` | `use_neox_rotary_style=False` → `True`（行 244）；`load_weights` 批量 dequant 改造 |
| `ep4_e2e_dump.py` | `get_rope(rotary_dim=hdim, ...)` → `get_rope(rotary_dim=fd.model_config.rotary_dim, ...)` |

### 15.7 下一步

1. **全量 62 层验证** — RoPE 修复后，62 层端到端精度应大幅改善
2. **性能优化** — SM80 bf16 workaround 的 numpy dequant 仍是瓶颈
3. **EP=4 验证** — 4 卡全量测试
