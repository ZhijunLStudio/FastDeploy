"""
vLLM MiniMax-M2.5 2-layer dump script (GPUs 6,7).

Usage (in vllm17 env):
    CUDA_VISIBLE_DEVICES=6,7 python scripts/run_vllm_dump.py
"""
import os, json

MODEL_DIR = '/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5'
cfg_path = os.path.join(MODEL_DIR, 'config.json')

# Backup & set 2 layers
with open(cfg_path) as f:
    orig_cfg = f.read()
d = json.loads(orig_cfg)
d['num_hidden_layers'] = 2
with open(cfg_path, 'w') as f:
    json.dump(d, f, indent=2)

try:
    DUMP_DIR = "/tmp/dump_compare/vllm"
    os.makedirs(DUMP_DIR, exist_ok=True)
    os.environ["VLLM_DUMP_DIR"] = DUMP_DIR

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        max_model_len=512,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )

    out = llm.generate(['Hello, how are you?'], SamplingParams(max_tokens=1, temperature=0))
    print('Output:', repr(out[0].outputs[0].text))
    print('Token IDs:', out[0].outputs[0].token_ids)
    print(f'Dump saved to {DUMP_DIR}')
finally:
    # Restore original config
    with open(cfg_path, 'w') as f:
        f.write(orig_cfg)
    print('config.json restored to 62 layers')
