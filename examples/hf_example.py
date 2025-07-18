import argparse
import sys
import time
import torch

import transformers

import upmem_llm_framework.pytorch_upmem_layers as upmem_layers
upmem_layers.profiler_init()

hf_token = "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"

model     = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=hf_token)
tokenizer = transformers.AutoTokenizer.from_pretrained       ("meta-llama/Llama-2-7b-chat-hf", use_auth_token=hf_token)

# 't' indicates when the input of a layer
# comes from HOST. This produces 2 transfers:  
#     - last device -> HOST
#     - HOST        -> new device
# layer_mapping = {
#    "LlamaRMSNorm"    : "PIM-AI-1chip,t",
#    "q_proj"	       : "PIM-AI-4chip,t",
#    "k_proj"	       : "PIM-AI-4chip"  ,
#    "rotary_emb"      : "PIM-AI-4chip"  ,
#    "v_proj"          : "PIM-AI-4chip"  ,
#    "o_proj"          : "PIM-AI-4chip,t",
#    "output_layernorm": "PIM-AI-1chip,t",
#    "gate_proj"       : "PIM-AI-4chip,t",
#    "up_proj"         : "PIM-AI-4chip,t",
#    "down_proj"       : "PIM-AI-4chip,t",
#    "norm"            : "PIM-AI-1chip,t",
#    "lm_head"         : "PIM-AI-4chip,t"
# }

layer_mapping = {
   "input_layernorm" : "Dimensity9300",
   "q_proj"	         : "Dimensity9300",
   "k_proj"	         : "Dimensity9300",
   "rotary_emb"      : "Dimensity9300",
   "v_proj"          : "Dimensity9300",
   "o_proj"          : "Dimensity9300",
   "output_layernorm": "Dimensity9300",
   "gate_proj"       : "Dimensity9300",
   "up_proj"         : "Dimensity9300",
   "down_proj"       : "Dimensity9300",
   "norm"            : "Dimensity9300",
   "lm_head"         : "Dimensity9300"
}


prompt = "Can you give me a brief summmary of the history of artificial intelligence? Let's start with the early days of AI research and then move on to the more recent developments in the field. Please keep it concise and focus on the key milestones and breakthroughs that have shaped the evolution of AI over the years. Limit your response to a few paragraphs, highlighting the most significant events and contributions in the history of AI."

inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)

# print out how many tokens are in the input
# print (f"Input tokens are {inputs.data["input_ids"][0].shape}")
print(f"Input tokens are {inputs['input_ids'][0].shape}")

model.eval() # Put model in evaluation / inference mode

#print (model)

upmem_layers.profiler_start(layer_mapping)
#start = time.time_ns() # In case we want to time the original execution (comment out profiler_start)
gen_tokens = model.generate(inputs.input_ids,
                            do_sample=True,
                            temperature=0.9,
		                    min_length=64,
                            max_length=1000)
#print ( (time.time_ns() - start)/1e6)
upmem_layers.profiler_end()

gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print (gen_text)
