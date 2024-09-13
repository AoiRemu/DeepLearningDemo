from transformers import BertTokenizerFast
import json
import datasets
from transformers import EncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# 准备数据
origin_data = datasets.Dataset.from_json('./ruozhi/dataset/spider.json')
# 按 90% 和 10% 的比例分割数据
train_test_split = origin_data.train_test_split(test_size=0.1)
train_data = train_test_split['train']
val_data = train_test_split['test']

# tokenizer = BertTokenizerFast.from_pretrained("./ruozhi/seq2seq/results/checkpoint-414")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# def map_to_length(x):
#     x['question_len'] = len(tokenizer(x['instruction'])['input_ids'])
#     x['answers_len'] = len(tokenizer(x['output'])['input_ids'])
#     return x

# simple_size = len(train_data)
# print(simple_size)
# def computed_and_print_stats(x):
#     print(f"question_len: {sum(x["question_len"]) / simple_size}, answers_len: {sum(x["answers_len"]) / simple_size}")

# data_stats = train_data.map(map_to_length)
# output = data_stats.map(computed_and_print_stats, batched=True, batch_size=-1)

encoder_max_length = 32
decoder_max_length = 128

def process_data_to_model_inputs(batch):
    inputs = tokenizer(batch['instruction'], padding='max_length', truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch['output'], padding='max_length', truncation=True, max_length=decoder_max_length)

    batch['input_ids'] = inputs.input_ids
    batch['attention_mask'] = inputs.attention_mask
    batch['labels'] = outputs.input_ids.copy()

    batch['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']]

    return batch

batch_size = 8
# batch_size = 4
train_data = train_data.map(process_data_to_model_inputs, batched=True, batch_size=batch_size, remove_columns=['instruction', 'output'])
# print(train_data)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

val_data = val_data.map(process_data_to_model_inputs, batched=True, batch_size=batch_size, remove_columns=['instruction', 'output'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 使用模型
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-chinese', 'bert-base-chinese')
# bert2bert = EncoderDecoderModel.from_pretrained('./ruozhi/seq2seq/results/checkpoint-414')
# 配置参数
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

# 设置训练参数
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    # output_dir='./ruozhi/seq2seq/results',
    output_dir='./ruozhi/seq2seq/test_results_0911',
    logging_steps=1000,
    save_steps=1000,
    eval_steps=500,
)

# 评估设置
rouge = datasets.load_metric('rouge', trust_remote_code=True)

def compute_metric(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pre_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pre_str, references=label_str, rouge_types=['rouge2'])['rouge2'].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge_fmeasure": round(rouge_output.fmeasure, 4),
    }

# 训练
trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metric,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()