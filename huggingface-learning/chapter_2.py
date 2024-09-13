from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "Can can need",
    "New bee",
    "You can you up",
    "No zuo no die",
    "Long time no see",
    "Good good study, day day up",
    "Give you some color to see see",
    "Me aand you, no three",
    "No money no talk",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# print(inputs)

outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

for i,item in enumerate(predictions):
    negative, positive = item
    # print(f'{negative},{positive}')
    result = '讲得好'
    if negative > 0.5:
        result = '你放屁'
    print(f'{raw_inputs[i]}\t{result}')

# print(model.config.id2label)