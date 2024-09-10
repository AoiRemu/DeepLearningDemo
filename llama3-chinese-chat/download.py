from huggingface_hub import snapshot_download
snapshot_download(repo_id="shenzhi-wang/Llama3.1-8B-Chinese-Chat", ignore_patterns=["*.gguf"])  # Download our BF16 model without downloading GGUF models.
