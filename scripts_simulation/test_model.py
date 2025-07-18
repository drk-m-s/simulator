import torch
import transformers
import sys
import argparse

your_token="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"

# Qwen2.5-VL-3B-Instruct
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# Add your local src directory to Python path
sys.path.insert(0, '/home/han.jiang/simulator/src')

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",use_fast=False)


# for name, module in model.named_modules():
#     print(name, type(module))

# def print_layer_hook(module, input, output):
#     print(f"Layer: {module.__class__.__name__}")

# for name, module in model.named_modules():
#     module.register_forward_hook(print_layer_hook)