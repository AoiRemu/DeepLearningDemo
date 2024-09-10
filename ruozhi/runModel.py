from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('./ruozhi_model_self')
model = BertForQuestionAnswering.from_pretrained('./ruozhi_model_self')

# 假设用户的问题
question = "马上要上游泳课了，昨天洗的泳裤还没干，怎么办"

# 对问题进行编码
encoding = tokenizer.encode_plus(
    question,
    add_special_tokens=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# 将编码后的数据送入模型进行预测
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
outputs = model(input_ids, attention_mask=attention_mask)

# 获取模型预测的起始和结束位置
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# 将问题转换为token化的形式
input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

# 找到预测的起始和结束位置对应的token
start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits) + 1

# 将预测的token转换回文本
predicted_answer_tokens = input_tokens[start_index:end_index]
predicted_answer = tokenizer.convert_tokens_to_string(predicted_answer_tokens)

print(f"Question: {question}")
print(f"Answer: {predicted_answer}")