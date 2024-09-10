from transformers import BertTokenizer, EncoderDecoderModel, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import torch
import json

# 假设我们已经有了一些对话数据
data = []
with open('./ruozhi/dataset/spider.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义一个简单的对话数据集
class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = 512
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
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
            'labels': targets['input_ids'].squeeze()  # 对于 seq2seq 模型，标签通常是输入的偏移
        }

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-chinese', 'bert-base-chinese')

# 设置 decoder_start_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# 创建数据集和数据加载器
dataset = DialogueDataset(data, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained('./ruozhi/seq2seq/model')
tokenizer.save_pretrained('./ruozhi/seq2seq/model')