from transformers import BertTokenizer, EncoderDecoderModel, TrainingArguments, Trainer
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            item['question'], 
            item['answers'],
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': inputs['input_ids'].flatten()
        }

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 编码数据
dataset = DialogueDataset(data, tokenizer)

# 初始化编码器-解码器模型
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-chinese', 
    'gpt2'
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./ruozhi/seq2seq/results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./ruozhi/seq2seq/logs',
    logging_steps=10,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # 可以添加eval_dataset参数进行评估
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained('./ruozhi/seq2seq/model')
tokenizer.save_pretrained('./ruozhi/seq2seq/model')