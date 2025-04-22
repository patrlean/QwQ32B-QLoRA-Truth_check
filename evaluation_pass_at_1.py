import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
from datetime import datetime

def extract_answer(text):
    match = re.search(r"\(([A-D])\)", text)
    if match:
        return f"({match.group(1)})"
    match = re.search(r"Answer: ([A-D])", text)
    if match:
        return f"({match.group(1)})"
    return text.strip()[-3:]  # fallback

def main():

    starttime_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = datetime.now()

    with open("evaluation/gpqa_diamond/prompts.json", "r") as f:
        prompts = json.load(f)
    with open("evaluation/gpqa_diamond/answers.json", "r") as f:
        answers = json.load(f)

    model = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        tensor_parallel_size=4
        )

    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95, 
        n=16, 
        max_tokens=32768
        )

    correct = 0
    total = len(prompts)
    batch_size = 16

    for i in range(0, len(prompts), batch_size):
        print(f"⏳ Prompt {i+1}-{min(i+batch_size, len(prompts))}")
        batch_prompts = prompts[i:i+batch_size]
        outputs = model.generate(batch_prompts, sampling_params)
        for j, out in enumerate(outputs):
            preds = [extract_answer(o.text) for o in out.outputs]
            correctness_per_response = 0
            assert len(preds) == sampling_params.n
            for pred in preds:
                if answers[i+j] == pred:
                    correctness_per_response += 1 / sampling_params.n
            correct += correctness_per_response


    print(f"\n🎯 Sampling-based pass@1: {correct}/{total} = {correct/total:.2%}")

    result = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "task": "gpqa_diamond_cot_zeroshot",
    "pass@1": correct / total,
    "correct": correct,
    "total": total,
    "temperature": sampling_params.temperature,
    "top_p": sampling_params.top_p,
    "n_samples_per_prompt": sampling_params.n,
    "max_tokens": sampling_params.max_tokens
}

    # 保存结果到文件夹中
    import os
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result["timestamp"] = timestamp
    result["start_time"] = starttime_stamp
    result["total_time"] = (datetime.now() - start_time).total_seconds() 

    os.makedirs("results/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond_cot", exist_ok=True)
    with open(f"results/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond_cot/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond_cot_{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ 结果已保存至 results/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond_cot/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond_cot_{timestamp}.json")


if __name__ == "__main__":
    main()
