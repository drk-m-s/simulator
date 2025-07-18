import torch
import transformers
import sys
import argparse

your_token="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"

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
model = AutoModelForImageTextToText.from_pretrained("THUDM/GLM-4.1V-9B-Thinking", trust_remote_code=True,token=your_token)
processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking", use_fast=False)

print (model)
for name, module in model.named_modules():
    print(name, type(module))