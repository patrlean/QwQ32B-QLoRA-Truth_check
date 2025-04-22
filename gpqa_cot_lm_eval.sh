
lm_eval --model vllm \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
    --tasks gpqa_diamond_cot_zeroshot \
    --batch_size auto \
    --log_samples \
    --limit 100 \
    --confirm_run_unsafe_code \
    --gen_kwargs '{"temperature": 0.6, "top_p": 0.95}' \
    --output_path results/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond_cot \

