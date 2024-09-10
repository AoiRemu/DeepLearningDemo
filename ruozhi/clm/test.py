from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载预训练模型和分词器
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 准备输入文本
input_text = "现在的孩子太脆弱了,一砸就碎"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成文本
output_ids = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

print(generated_text)