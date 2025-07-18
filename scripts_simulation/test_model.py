import torch
import transformers


your_token="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"

model     = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=your_token,device_map="auto", torch_dtype="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=your_token)
print (model)
