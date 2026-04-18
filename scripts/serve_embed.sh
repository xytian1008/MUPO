vllm serve Qwen/Qwen3-Embedding-0.6B \
    --gpu_memory_utilization 0.0125 \
    --port 8000 \
    --max-model-len 32768 \
    --task embed \
    --tensor-parallel-size 8