import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('./ruozhi/seq2seq/model')

# 加载模型
model = torch.load('./ruozhi/seq2seq/model/model.pth')
model.eval()  # 切换到评估模式

# 输入句子
input_text = "这是一个测试句子。"

# 使用分词器进行编码
inputs = tokenizer(input_text, return_tensors='pt')

# 进行推理
with torch.no_grad():
    outputs = model.generate(**inputs)

# 解码输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("输出句子:", output_text)