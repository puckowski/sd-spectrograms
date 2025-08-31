from transformers import AutoTokenizer

# Replace this with the *full path* to the folder that *does* contain config.json
local_tokenizer_dir = r"D:/Projects/Stable Diffusion Spectrogram/onnx_model/tokenizer"

tok = AutoTokenizer.from_pretrained(local_tokenizer_dir, use_fast=True)
tok.save_pretrained(local_tokenizer_dir)
print("✅ Fast tokenizer JSON written to", local_tokenizer_dir)