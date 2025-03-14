import argparse
import logging
import os
import time
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import tqdm
import datetime
import deepspeed
import torch.distributed as dist  # 用于分布式通信

# 生成基于当前时间的日志文件名
log_filename = datetime.datetime.now().strftime("eval_%Y-%m-%d_%H-%M-%S.log")

# 配置 logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_filename,  # 动态日志文件名
    filemode="w"
)

# 创建控制台日志 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# 获取 logger 并添加控制台 handler
logger = logging.getLogger("Evaluator")
logger.addHandler(console_handler)

def direct_choice_single(question, choices, model, tokenizer, device, max_new_tokens=50):
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prompt = f"question:{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{option_letters[i]}. {choice}\n"
    prompt += 'Please show your choice in the answer field with only the choice letter, e.g., "answer": "B".'

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0.6, 
            top_p=0.95, 
            top_k=30, 
            do_sample=True
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "<think>" in response:
        response = response.split("<think>", 1)[1]

    logger.debug(f"MC1 Response: {response}")
    json_match = re.search(r'"answer":\s*"([^"]+)"', response)
    valid_letters = option_letters[:len(choices)]
    if json_match:
        extracted = json_match.group(1)
        letters = [letter for letter in re.findall(r"[A-Z]", extracted) if letter in valid_letters]
        if letters:
            return letters[0]

    letters = [letter for letter in re.findall(r"[A-Z]", response) if letter in valid_letters]
    return letters[0] if letters else response.strip()

def direct_choice_multi(question, choices, model, tokenizer, device, max_new_tokens=50):
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prompt = f"question:{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{option_letters[i]}. {choice}\n"
    prompt += 'Please show your choice in the answer field with multiple choice letters, the number of right answers could be any number. e.g., "answer": "A, B, C".'

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0.6, 
            top_p=0.95, 
            top_k=30, 
            do_sample=True
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "<think>" in response:
        response = response.split("<think>", 1)[1]

    logger.debug(f"MC2 Response: {response}")
    json_match = re.search(r'"answer":\s*"([^"]+)"', response)
    valid_letters = option_letters[:len(choices)]
    if json_match:
        extracted = json_match.group(1)
        letters = sorted(set(letter for letter in re.findall(r"[A-Z]", extracted) if letter in valid_letters))
        return letters

    letters = sorted(set(letter for letter in re.findall(r"[A-Z]", response) if letter in valid_letters))
    return letters

def main(args):
    logger.info("loading dataset...")
    dataset_name = "truthfulqa/truthful_qa"
    subset = "multiple_choice"
    ds = load_dataset(dataset_name, subset)
    data = ds['validation']

    # 初始化分布式相关变量
    if args.multi_gpu:
        deepspeed.init_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # 根据 rank 划分数据集，每个进程仅处理子集
    data = data.select(range(rank, len(data), world_size))

    model_name = "Qwen/QwQ-32B"
    # device_map 对推理模型由 DeepSpeed 处理，这里 device 用于 tokenizer 输入数据的转换
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", 0)))
    
    logger.info(f"加载模型：{model_name}")
    # 加载模型到 CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}
    )

    with open("evaluation/ds_inference_config.json", "r") as f:
        ds_config = json.load(f)

    # if args.multi_gpu:
    #     local_rank = int(os.getenv("LOCAL_RANK", 0))
    #     device = torch.device("cuda", local_rank)
    #     model = model.to(device)

    ds_engine = deepspeed.init_inference(
        model,
        config=ds_config['inference']
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    correct_mc1, total_mc1, correct_mc2, total_mc2 = 0, 0, 0, 0
    output_results, wrong_answers = [], []
    summary_results = []  
    for index, sample in enumerate(tqdm.tqdm(data, desc=f"Evaluating Rank {rank}")):
        # 跳过 start_index 之前的数据
        if index < args.start_index:
            continue

        logger.info(f"当前推理数据索引: {index} (Rank {rank})")
        question = sample["question"]
        sample_result = {"question": question, "index": index, "rank": rank}
        
        # 处理 mc1
        mc1 = sample["mc1_targets"]
        choices1, labels1 = mc1["choices"], mc1["labels"]
        choices1_dict = {chr(65 + i): choice for i, choice in enumerate(choices1)}
        correct_letter = chr(65 + labels1.index(1)) if 1 in labels1 else None
        
        predicted_letter = direct_choice_single(question, choices1, ds_engine, tokenizer, device, max_new_tokens=args.max_new_tokens)
        sample_result.update({"mc1_choices": choices1_dict, "mc1_predicted": predicted_letter, "mc1_correct": correct_letter})
        
        if predicted_letter == correct_letter:
            correct_mc1 += 1
        else:
            wrong_answers.append(sample_result)
        total_mc1 += 1

        # 处理 mc2
        mc2 = sample["mc2_targets"]
        choices2, labels2 = mc2["choices"], mc2["labels"]
        choices2_dict = {chr(65 + i): choice for i, choice in enumerate(choices2)}
        correct_letters = sorted([chr(65 + i) for i in range(len(labels2)) if labels2[i] == 1])

        predicted_letters = direct_choice_multi(question, choices2, ds_engine, tokenizer, device, max_new_tokens=args.max_new_tokens)
        sample_result.update({"mc2_choices": choices2_dict, "mc2_predicted": predicted_letters, "mc2_correct": correct_letters})

        if predicted_letters == correct_letters:
            correct_mc2 += 1
        total_mc2 += 1
        
        output_results.append(sample_result)
        
        # 记录摘要信息
        summary_results.append({
            "question id": index,
            "mc1 prediction": predicted_letter,
            "mc1 truth": correct_letter,
            "mc2 prediction": predicted_letters,
            "mc2 truth": correct_letters,
            "rank": rank
        })

    logger.info(f"Rank {rank}: MC1 正确率: {correct_mc1}/{total_mc1} = {correct_mc1 / total_mc1:.4f}")
    logger.info(f"Rank {rank}: MC2 正确率: {correct_mc2}/{total_mc2} = {correct_mc2 / total_mc2:.4f}")

    # 分布式结果汇总：如果是多 GPU 环境，则各进程结果汇总后仅由 rank 0 写文件
    if args.multi_gpu:
        gathered_results = [None for _ in range(world_size)]
        gathered_wrong_answers = [None for _ in range(world_size)]
        gathered_summary_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, output_results)
        dist.all_gather_object(gathered_wrong_answers, wrong_answers)
        dist.all_gather_object(gathered_summary_results, summary_results)
        if rank == 0:
            final_results = []
            for res in gathered_results:
                final_results.extend(res)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)

            final_wrong_answers = []
            for res in gathered_wrong_answers:
                final_wrong_answers.extend(res)
            with open("wrong_answer.json", "w", encoding="utf-8") as f:
                json.dump(final_wrong_answers, f, ensure_ascii=False, indent=4)

            final_summary_results = []
            for res in gathered_summary_results:
                final_summary_results.extend(res)
            with open("summary_results.json", "w", encoding="utf-8") as f:
                json.dump(final_summary_results, f, ensure_ascii=False, indent=4)
    else:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_results, f, ensure_ascii=False, indent=4)
        with open("wrong_answer.json", "w", encoding="utf-8") as f:
            json.dump(wrong_answers, f, ensure_ascii=False, indent=4)
        with open("summary_results.json", "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在 TruthfulQA 数据集上评估模型")
    parser.add_argument("-m", "--multi_gpu", action="store_true")
    parser.add_argument("-t", "--max_new_tokens", type=int, default=32768)
    parser.add_argument("--output_json", type=str, default="results.json")
    parser.add_argument("--start_index", type=int, default=0, help="从数据集中的指定索引开始评估")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    main(args)
