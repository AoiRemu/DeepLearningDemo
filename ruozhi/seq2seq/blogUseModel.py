from transformers import BertTokenizer
import json
import datasets
from transformers import EncoderDecoderModel

bert2bert = EncoderDecoderModel.from_pretrained('./ruozhi/seq2seq/more_results/checkpoint-414')
tokenizer = BertTokenizer.from_pretrained("./ruozhi/seq2seq/more_results/checkpoint-414")

input_text = '鸡柳是鸡身上哪个部位啊？'
inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
input_ids = inputs.input_ids.to('cpu')
attention_mask = inputs.attention_mask.to('cpu')
outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Q:{input_text}\nA:{output_str}')