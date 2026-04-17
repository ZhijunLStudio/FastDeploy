"""
Dump comparison workflow for FD vs vLLM MiniMax-M2.5.

Step 1: python scripts/run_vllm_dump.py   (in vllm17 env, GPUs 6,7)
Step 2: python scripts/run_fd_dump.py      (in paddle env, GPUs 6,7)
Step 3: python scripts/compare_dump.py     (any env, compare npy files)

Both scripts set num_hidden_layers=2 in config.json before loading.
Remember to restore config.json to 62 after testing.
"""
