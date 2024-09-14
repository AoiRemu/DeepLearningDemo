from transformers import AutoModelForCausalLM, AutoTokenizer

lingzhi_model_path = "Lingzhi-AI/Lingzhi-2.7B-chat"

# model = AutoModelForCausalLM.from_pretrained(
#     lingzhi_model_path,
# )
# model.save_pretrained("./lingzhi-2.7B-chat")
tokenizer = AutoTokenizer.from_pretrained(lingzhi_model_path)
tokenizer.save_pretrained("./lingzhi-2.7B-chat")
