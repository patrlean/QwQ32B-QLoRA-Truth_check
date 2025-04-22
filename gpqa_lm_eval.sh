
lm_eval --model vllm \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
    --tasks gpqa_diamond_zeroshot \
    --batch_size auto \
    
    --confirm_run_unsafe_code \
    --output_path results/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond \

