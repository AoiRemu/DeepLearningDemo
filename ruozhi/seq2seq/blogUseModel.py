from transformers import BertTokenizer
import json
import datasets
from transformers import EncoderDecoderModel
import torch

bert2bert = EncoderDecoderModel.from_pretrained('./ruozhi/seq2seq/test_results_0911/checkpoint-2097')
tokenizer = BertTokenizer.from_pretrained("./ruozhi/seq2seq/test_results_0911/checkpoint-2097")

# input_text = '如果我是DJ你会爱我吗'
# inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
# input_ids = inputs.input_ids.to('cpu')
# attention_mask = inputs.attention_mask.to('cpu')
# outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
# output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f'Q:{input_text}\nA:{output_str}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate(input_text):
    inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_str

if __name__ == '__main__':
    while True:
        input_text = input('请输入你的问题：')
        if input_text == 'exit':
            break
        output_str = generate(input_text)
        print(f'A:{output_str}')