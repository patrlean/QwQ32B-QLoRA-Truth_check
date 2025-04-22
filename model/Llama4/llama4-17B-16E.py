# inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E"

print(">>> Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(">>> Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",                  # 自动分配层到8张卡
    use_safetensors=True               # 加快加载速度
)

print(">>> Building pipeline...")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(">>> Running inference...")
prompt = "Roses are red,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
