import argparse
import sys
import time
import torch
# Add your local src directory to Python path
sys.path.insert(0, '/home/han.jiang/simulator/src')

import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import upmem_llm_framework.pytorch_upmem_layers as upmem_layers
# upmem_layers.profiler_init()

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="unknown")
parser.add_argument("--in-tokens", default=64, type=int)
parser.add_argument("--out-tokens", default=128, type=int)
parser.add_argument("--bs", default=1, type=int)
options, _ = parser.parse_known_args()

print ("Simulating with device...", options.device)
print ("in:", options.in_tokens, "out:", options.out_tokens)

your_token="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"

model     = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", token=your_token,torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", token=your_token)

layer_mapping = {
        "LlamaRMSNorm"    : options.device,
        "q_proj"	  : options.device,
        "k_proj"	  : options.device,
        "rotatory_emb"    : options.device,
        "v_proj"          : options.device,
        "o_proj"          : options.device,
        "output_layernorm": options.device,
        "gate_proj"       : options.device,
        "up_proj"         : options.device,
        "down_proj"       : options.device,
        "norm"            : options.device,
        "lm_head"         : options.device
}
layer_attn_ctxt = "q_proj"

print (model)

# print ("Batch 1")
# prompt = "placeholder"
# prompt_batch = [prompt] * options.bs
# input_ids = tokenizer(prompt_batch, return_tensors="pt", return_token_type_ids=False)
# input_ids["input_ids"] = torch.randint(100, [options.bs, options.in_tokens])
# input_ids["attention_mask"] = torch.ones([options.bs, options.in_tokens], dtype=torch.int)
# print (input_ids.data["input_ids"][0].shape)


# model.eval()

# upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt = layer_attn_ctxt)
# #start = time.time_ns()
# gen_tokens = model.generate(**input_ids,
#                             do_sample=True,
#                             temperature=0.9,
# 							min_length=options.out_tokens,
#                             max_length=options.out_tokens+100)
# #print ( (time.time_ns() - start)/1e6)
# upmem_layers.profiler_end  ()

# gen_text = tokenizer.batch_decode(gen_tokens)
# print (gen_text)

# sys.exit()

#------------------------------------------------------------------------#

