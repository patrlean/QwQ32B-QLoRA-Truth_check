import os
import time
import logging
from vllm import LLM, SamplingParams

# 设置详细日志环境变量（在程序启动前设置）
os.environ["VLLM_LOG_LEVEL"] = "DEBUG"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
os.environ["VLLM_ENABLE_CUDA_GRAPHS"] = "0"

# 记录程序各个阶段的时间
def timestamp(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

timestamp("开始程序")

# 模型设置
model_name = "Qwen/Qwen2.5-7B-Instruct"
tensor_parallel_size = 4
max_model_len = 1024  # 可调低一点加快初始化

# 记录 LLM 初始化开始时间
timestamp("开始初始化 LLM 对象")
t0 = time.time()
llm = LLM(
    model=model_name,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=max_model_len,
)
t1 = time.time()
timestamp(f"LLM 初始化完成，用时 {t1 - t0:.2f} 秒")

# 构造 prompts 和 sampling 参数
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 开始生成
timestamp("开始调用 llm.generate()")
t2 = time.time()
outputs = llm.generate(prompts, sampling_params)
t3 = time.time()
timestamp(f"生成完成，用时 {t3 - t2:.2f} 秒")

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

timestamp("程序结束")
