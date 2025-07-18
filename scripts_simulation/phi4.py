import torch
import transformers
import sys
import argparse

your_token="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"

# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True,token=your_token)
tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/Phi-4-multimodal-instruct", token=your_token)
print (model)
for name, module in model.named_modules():
    print(name, type(module))