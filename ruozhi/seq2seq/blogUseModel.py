from transformers import BertTokenizer
import json
import datasets
from transformers import EncoderDecoderModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
bert2bert = EncoderDecoderModel.from_pretrained('./ruozhi/seq2seq/test_results/checkpoint-1044').to(device)
tokenizer = BertTokenizer.from_pretrained("./ruozhi/seq2seq/test_results/checkpoint-1044")

input_text = '我告诉我家的狗它不是我亲生的'
inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)
outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Q:{input_text}\nA:{output_str}')