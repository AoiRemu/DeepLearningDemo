from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, AutoModelWithLMHead, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import json

def train():
    # 1. 加载数据
    def load_custom_dataset(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_dict({
            'input': [item['question'] for item in data],
            'output': [item['answers'] for item in data]
        })

    dataset = load_custom_dataset('./ruozhi/dataset/spider.json')

    # 2. 加载预训练模型和分词器
    model_name = 'gpt2'  # 你可以选择其他预训练的中文模型
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 设置 pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 数据预处理
    def preprocess_function(examples):
        inputs = tokenizer(examples['input'], max_length=128, truncation=True, padding='max_length')
        outputs = tokenizer(examples['output'], max_length=128, truncation=True, padding='max_length')
        inputs['labels'] = outputs['input_ids']
        return inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 4. 定义训练参数
    training_args = TrainingArguments(
        output_dir='./ruozhi/results',
        evaluation_strategy="no",
        # evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # 5. 初始化Trainer并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    trainer.train()

    # 6. 保存模型
    model.save_pretrained('./ruozhi/trained_model')
    tokenizer.save_pretrained('./ruozhi/trained_model')

# 7. 使用模型生成对话
def generate_response(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained('./ruozhi/trained_model')
    model = GPT2LMHeadModel.from_pretrained('./ruozhi/trained_model')
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    # train()
    answer = generate_response('一斤棉花和一斤铁，同时掉进水里你先救谁')
    print(answer)