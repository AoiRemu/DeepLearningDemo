from transformers import BertTokenizer,BertModel,EncoderDecoderModel

encode_model_name = 'bert-base-chinese'
decode_model_name = 'gpt2'
tokenizer = BertTokenizer.from_pretrained(encode_model_name, clean_up_tokenization_spaces=True)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(encode_model_name, decode_model_name)

input_text = "你好"
encoded_input = tokenizer(input_text, return_tensors='pt')

print(encoded_input)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

generate_output = model(
    input_ids=encoded_input['input_ids'],
    attention_mask=encoded_input['attention_mask'],
    labels=encoded_input['input_ids']
    # decoder_start_token_id=tokenizer.cls_token_id
)
print(type(generate_output))
# generated_text = tokenizer.decode(generate_output[0], skip_special_tokens=True)
# print(generated_text)