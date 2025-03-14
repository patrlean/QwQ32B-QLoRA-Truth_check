from mii import pipeline
import deepspeed
import os

pipe = pipeline("Qwen/QwQ-32B")
for i in range(10):
    output = pipe(["<think>\nMII features include blocked KV-caching, continuous batching, Dynamic SplitFuse, tensor parallelism, and high-performance CUDA kernels to support fast high throughput text-generation for LLMs such as Llama-2-70B, Mixtral (MoE) 8x7B, and Phi-2. The latest updates in v0.2 add new model families, performance optimizations, and feature enhancements. MII now delivers up to 2.5 times higher effective throughput compared to leading systems such as vLLM. For detailed performance results please see our latest DeepSpeed-FastGen blog and DeepSpeed-FastGen release blog."],
        min_new_tokens=100,
        max_new_tokens=10000,
        temperature=0.6, 
        top_p=0.95, 
        top_k=30, 
        do_sample=True
    )
    output2 = pipe(["<think>\n是否存在无穷多个四胞胎素数"],
        min_new_tokens=100,
        max_new_tokens=10000,
        temperature=0.6, 
        top_p=0.95, 
        top_k=30, 
        do_sample=True
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 节点内进程 ID
    
    if local_rank == 0:
        text = output[0].generated_text
        text2 = output2[0].generated_text
        print("------------------------------------------------------")
        print(f"\nHello my name is {text}\n")
        print("------------------------------------------------------")
        print(f"\noutput 2 is {text2}\n")