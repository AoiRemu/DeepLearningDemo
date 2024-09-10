from transformers import BertTokenizer, EncoderDecoderModel, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import AdamW, get_scheduler
from datetime import datetime
import os

# 假设我们已经有了一些对话数据
data = []
with open('./ruozhi/dataset/spider.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = 512
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        item = self.data[i]
        inputs = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            item['answers'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
        }

class ChatModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.fn = torch.nn.Linear(768, 512)
    
    def forward(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        outputs = self.fn(outputs.encoder_last_hidden_state[:, 0])
        # outputs['loss'] = outputs.loss
        return outputs

def train():
    encode_model_name = 'bert-base-chinese'
    decode_model_name = 'gpt2'
    encode_tokenizer = BertTokenizer.from_pretrained(encode_model_name)
    premodel = EncoderDecoderModel.from_encoder_decoder_pretrained(encode_model_name, decode_model_name)

    for param in premodel.encoder.parameters():
        param.requires_grad = False

    # 获取解码器的起始令牌ID
    decoder_start_token_id = encode_tokenizer.cls_token_id
    premodel.config.decoder_start_token_id = decoder_start_token_id
    premodel.config.pad_token_id = encode_tokenizer.pad_token_id

    dataset = ChatDataset(data, encode_tokenizer)
    model = ChatModel(premodel, encode_tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epoch_num = 3

    model.train()
    global_step = 0
    total_steps = len(loader) * epoch_num
    celoss = torch.nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        for i, batch in enumerate(loader):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            labels = batch['labels'] * 1e-4
            # labels = batch['labels'].float()

            loss = celoss(outputs, labels)
            # loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            # if global_step % 10 == 0:
            # 获取当前时间
            current_time = datetime.now()

            # 格式化时间
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            print(f'total_steps:{total_steps}, epoch: {epoch}, step: {global_step}, loss: {loss.item()}, time: {formatted_time}')

    torch.save(model, './ruozhi/seq2seq/model/model.pth')
    encode_tokenizer.save_pretrained('./ruozhi/seq2seq/model')

def useModel():
    tokenizer = BertTokenizer.from_pretrained('./ruozhi/seq2seq/model')
    model = torch.load('./ruozhi/seq2seq/model/model.pth')
    model.eval()
    # 输入句子
    input_text = "没有未来的未来不是我想要的未来"

    # 使用分词器进行编码
    inputs = tokenizer(input_text, return_tensors='pt')

    # 进行推理
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['input_ids'])

    # 解码输出
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("输出句子:", output_text)

if __name__ == '__main__':
    useModel()