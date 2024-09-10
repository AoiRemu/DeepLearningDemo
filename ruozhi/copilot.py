import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
import json

# 自定义 collate_fn 函数
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    start_positions = [item['start_positions'] for item in batch]
    end_positions = [item['end_positions'] for item in batch]

    # 使用 pad_sequence 进行填充
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    # 将 start_positions 和 end_positions 转换为张量
    start_positions = torch.tensor(start_positions, dtype=torch.long)
    end_positions = torch.tensor(end_positions, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

# 定义数据集类
class ChineseQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['instruction']
        context = item['output']
        start_positions = 0
        end_positions = len(context) - 1
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
          # 确保 start_positions 和 end_positions 在截断后的范围内
        if start_positions >= self.max_len:
            start_positions = self.max_len - 1
        if end_positions >= self.max_len:
            end_positions = self.max_len - 1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': start_positions,
            'end_positions': end_positions
        }

# 加载分词器和预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', clean_up_tokenization_spaces=False)
model = BertForQuestionAnswering.from_pretrained('bert-base-chinese')

# 实例化数据集和数据加载器
dataset = ChineseQADataset('ruozhi/dataset/ruozhi.json', tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

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

        # 更新参数
        optimizer.step()
        print(f"batch {step + 1}/{len(dataloader)}, Loss: {loss.item()}")

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# 保存模型
model.save_pretrained('./ruozhi_model_self')
# 保存分词器
tokenizer.save_pretrained('./ruozhi_model_self')