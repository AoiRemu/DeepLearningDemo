from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

lingzhi_model_path = "./lingzhi-2.7B-chat"

# model = AutoModelForCausalLM.from_pretrained(
#     lingzhi_model_path,
#     torch_dtype="auto",
#     device_map="auto",
# )
tokenizer = AutoTokenizer.from_pretrained(lingzhi_model_path)
# result = tokenizer.convert_ids_to_tokens(151643)
token = tokenizer.convert_tokens_to_ids("<|im_start|>")
print(token)