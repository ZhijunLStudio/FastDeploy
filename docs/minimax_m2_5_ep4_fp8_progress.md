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

### LLM API 端到端（prefill + decode）：**TP=2 EP=2 已跑通** ✅

TP=2 EP=2 + disable_sequence_parallel_moe + 2 层：成功，无崩溃，有输出。

### Decode 阶段：**存在精度差异** ⚠️

2 层 TP=2 端到端结果对比：
- **FD**: token 367, 84284, 367, 10, 37737, ...（乱码混合日文/法文/中文字符）
- **vLLM**: token 367, 367, 367, 367, ...（全部是换行符 367）

两者从第 2 个 token 开始就不一致。

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
