import os
from huggingface_hub import hf_hub_download

path = hf_hub_download("Qwen/Qwen2.5-VL-3B-Instruct", "config.json")
snapshot_dir = os.path.dirname(path)          # snapshot folder
model_root = os.path.dirname(snapshot_dir)    # models--Qwen--Qwen2.5-VL-7B-Instruct-AWQ
print("Snapshot dir:", snapshot_dir)
print("Model root:", model_root)