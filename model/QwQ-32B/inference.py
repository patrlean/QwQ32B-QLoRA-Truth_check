from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\nQuestion: There are three cards with letters $\\texttt{a}$, $\\texttt{b}$, $\\texttt{c}$ placed in a row in some order. You can do the following operation at most once: \n \n-  Pick two cards, and swap them.  Is it possible that the row becomes $\\texttt{abc}$ after the operation? Output \"YES\" if it is possible, and \"NO\" otherwise.\n\nInput\n\nThe first line contains a single integer $t$ ($1 \\leq t \\leq 6$) — the number of test cases.\n\nThe only line of each test case contains a single string consisting of each of the three characters $\\texttt{a}$, $\\texttt{b}$, and $\\texttt{c}$ exactly once, representing the cards.\n\nOutput\n\nFor each test case, output \"YES\" if you can make the row $\\texttt{abc}$ with at most one operation, or \"NO\" otherwise.\n\nYou can output the answer in any case (for example, the strings \"yEs\", \"yes\", \"Yes\" and \"YES\" will be recognized as a positive answer).Sample Input 1:\n6\n\nabc\n\nacb\n\nbac\n\nbca\n\ncab\n\ncba\n\n\nSample Output 1:\n\nYES\nYES\nYES\nNO\nNO\nYES\n\n\nNote\n\nIn the first test case, we don't need to do any operations, since the row is already $\\texttt{abc}$.\n\nIn the second test case, we can swap $\\texttt{c}$ and $\\texttt{b}$: $\\texttt{acb} \\to \\texttt{abc}$.\n\nIn the third test case, we can swap $\\texttt{b}$ and $\\texttt{a}$: $\\texttt{bac} \\to \\texttt{abc}$.\n\nIn the fourth test case, it is impossible to make $\\texttt{abc}$ using at most one operation.\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT. Enclose your code within delimiters as follows. \n\n```python\n# YOUR CODE HERE\n```"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)