#!/bin/bash

# --- 配置参数 ---
MODEL_PATH="/home/aistudio/config_folder"
PORT=7890
METRICS_PORT=8021
WORKER_QUEUE_PORT=8022
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=4
LOAD_CHOICES="default_v1"
TP_SIZE=8

# --- 启动命令 ---
echo "Starting FastDeploy OpenAI API server..."
echo "Model Path: $MODEL_PATH"
echo "API Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"

python -m fastdeploy.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --metrics-port $METRICS_PORT \
    --engine-worker-queue-port $WORKER_QUEUE_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --load_choices "$LOAD_CHOICES" \
    --tensor-parallel-size $TP_SIZE 

echo "Server stopped."