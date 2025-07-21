import torch
import transformers
import sys
import argparse

your_token="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"

# Load model directly
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
model = Glm4vForConditionalGeneration.from_pretrained("THUDM/GLM-4.1V-9B-Thinking", torch_dtype=torch.bfloat16, device_map="auto", token=your_token)
processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")

print (model)
for name, module in model.named_modules():
    print(name, type(module))