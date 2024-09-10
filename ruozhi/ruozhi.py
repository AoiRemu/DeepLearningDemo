import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.nn.utils import clip_grad_norm_
import torch

# 定义数据集类
class ChineseQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']

        # 对问题和答案进行编码
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        token_type_ids = encoding['token_type_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # 将答案编码为BERT的输入格式
        start_positions = torch.tensor(self.tokenizer.encode(answer, add_special_tokens=False)[:-1]).to(torch.long)
        end_positions = start_positions + 1

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions
        }

# 加载分词器和预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=False)
model = BertForQuestionAnswering.from_pretrained('bert-base-chinese')

# 实例化数据集和数据加载器
dataset = ChineseQADataset('ruozhi/dataset/ruozhi.json', tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 将模型设置为训练模式
model.train()

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        # 清空之前梯度
        model.zero_grad()

        # 前向传播
        outputs = model(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions']
        )

        # 计算损失
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪
        clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

        if step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

    print(f"Epoch: {epoch}, Total Loss: {total_loss}")

# 保存模型
model.save_pretrained('path_to_save_model')