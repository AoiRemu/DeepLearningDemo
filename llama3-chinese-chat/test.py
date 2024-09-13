import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "shenzhi-wang/Llama3.1-8B-Chinese-Chat"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
)

chat = [
    {"role": "user", "content": "写一首关于机器学习的诗。"},
]
input_ids = tokenizer.apply_chat_template(
    chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=8192,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))