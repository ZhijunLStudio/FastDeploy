"""Quick benchmark: SM80 WINT4 MoE single layer forward speed."""
import os, sys, time, argparse, numpy as np
sys.path.insert(0, '/data/lizhijun/work/fd-vllm/FastDeploy')
os.environ["FD_WINT4_QUANTIZE"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--n_layers", type=int, default=0, help="Override num_hidden_layers (0=use full model)")
parser.add_argument("--prompt", type=str, default="Hello")
parser.add_argument("--gpus", type=str, default="6,7", help="CUDA_VISIBLE_DEVICES")
parser.add_argument("--dump_dir", type=str, default="", help="Set FD_DUMP_DIR and FD_MOE_DUMP_DIR")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
if args.dump_dir:
    os.makedirs(args.dump_dir, exist_ok=True)
    os.environ["FD_DUMP_DIR"] = args.dump_dir
    os.environ["FD_MOE_DUMP_DIR"] = args.dump_dir

import paddle

from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams

tp_size = len(args.gpus.split(","))

llm_kwargs = dict(
    model='/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5',
    tensor_parallel_size=tp_size,
    disable_sequence_parallel_moe=True,
    max_model_len=256,
    gpu_memory_utilization=0.90,
    max_num_seqs=1,
    num_gpu_blocks_override=100,
    max_num_batched_tokens=256,
    graph_optimization_config={'use_cudagraph': False},
)

if args.n_layers > 0:
    # Override num_hidden_layers via model config
    # FD reads config.json, so we patch it temporarily
    import json
    config_path = '/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5/config.json'
    with open(config_path, "r") as f:
        orig_config = json.load(f)
    orig_n = orig_config.get("num_hidden_layers", 62)
    patched_config = dict(orig_config)
    patched_config["num_hidden_layers"] = args.n_layers
    with open(config_path, "w") as f:
        json.dump(patched_config, f, indent=2)
    print(f"Patched config.json: num_hidden_layers={orig_n} -> {args.n_layers}")
else:
    orig_config = None
    orig_n = None

try:
    llm = LLM(**llm_kwargs)

    t0 = time.time()
    outputs = llm.generate([args.prompt], SamplingParams(temperature=0, max_tokens=1))
    dt = time.time() - t0
    print(f"Prefill+1decode: {dt:.1f}s", flush=True)

    for out in outputs:
        try:
            if isinstance(out.outputs, list):
                print(f'text: {repr(out.outputs[0].text)}')
                print(f'tokens: {out.outputs[0].token_ids}')
            else:
                print(f'text: {repr(out.outputs.text)}')
                print(f'tokens: {out.outputs.token_ids}')
        except Exception as e:
            print(f'output error: {e}')
            print(f'outputs type: {type(out.outputs)}')

    if args.dump_dir:
        import glob
        files = sorted(glob.glob(f"{args.dump_dir}/*.npy"))
        print(f"\nDumped {len(files)} files to {args.dump_dir}:")
        for f in files:
            arr = np.load(f)
            print(f"  {os.path.basename(f)}: shape={arr.shape}")

finally:
    # Restore config
    if orig_config is not None:
        with open('/data-ssd/lizhijun/models/MiniMax/MiniMax-M2.5/config.json', "w") as f:
            json.dump(orig_config, f, indent=2)
        print(f"Restored config.json: num_hidden_layers={orig_n}")
