import argparse
import sys
import time
import torch
# Add your local src directory to Python path
sys.path.insert(0, '/home/han.jiang/simulator/src')

import transformers

import upmem_llm_framework.pytorch_upmem_layers as upmem_layers
upmem_layers.profiler_init()

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="unknown")
parser.add_argument("--in-tokens", default=1000, type=int)
parser.add_argument("--out-tokens", default=1000, type=int)
parser.add_argument("--bs", default=1, type=int)
options, _ = parser.parse_known_args()


print ("Simulating with device...", options.device)
print ("in:", options.in_tokens, "out:", options.out_tokens)

your_token="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"

model     = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=your_token,device_map="auto", torch_dtype="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=your_token)
# print (model)


#layer_mapping = {
#	    "LlamaRMSNorm"    : "PIM-AI-1chip,t",
#		"q_proj"	      : "PIM-AI-4chip,t",
#		"k_proj"		  : "PIM-AI-4chip"  ,
#		"rotatory_emb"    : "PIM-AI-4chip"  ,
#		"v_proj"          : "PIM-AI-4chip"  ,
#		"o_proj"		  : "PIM-AI-4chip,t",
#		"output_layernorm": "PIM-AI-1chip,t",
#		"gate_proj"       : "PIM-AI-4chip,t",
#		"up_proj"         : "PIM-AI-4chip,t",
#		"down_proj"       : "PIM-AI-4chip,t",
#		"norm"            : "PIM-AI-1chip,t",
#		"lm_head"         : "PIM-AI-4chip,t"
#}

layer_mapping = {
        "LlamaRMSNorm"    : options.device,
        "q_proj"	      : options.device,
        "k_proj"		  : options.device,
        "rotatory_emb"    : options.device,
        "v_proj"          : options.device,
        "o_proj"		  : options.device,
        "output_layernorm": options.device,
        "gate_proj"       : options.device,
        "up_proj"         : options.device,
        "down_proj"       : options.device,
        "norm"            : options.device,
        "lm_head"         : options.device
}
layer_attn_ctxt = "q_proj"


print (f"Batch {options.bs}")
prompt = "placeholder"
prompt_batch = [prompt] * options.bs
input_ids = tokenizer(prompt_batch, return_tensors="pt", return_token_type_ids=False)
input_ids["input_ids"] = torch.randint(100, [options.bs, options.in_tokens])
input_ids["attention_mask"] = torch.ones([options.bs, options.in_tokens], dtype=torch.int)
print (input_ids.data["input_ids"][0].shape)


model.eval()
# print (model)
upmem_layers.profiler_start(layer_mapping, layer_attn_ctxt = layer_attn_ctxt, batch_size=options.bs)
#start = time.time_ns()
gen_tokens = model.generate(**input_ids,
                            do_sample=True,
                            temperature=0.9,
							min_new_tokens=options.out_tokens,
                            max_new_tokens=options.out_tokens)
#print ( (time.time_ns() - start)/1e6)
upmem_layers.profiler_end  ()

gen_text = tokenizer.batch_decode(gen_tokens)
print (gen_text)

sys.exit()


## torch profiler snippet
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("forward"):
#        gen_text = tokenizer.batch_decode(gen_tokens)[0]
#        #model(inputs)
# 
# 
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
# 
# print ("----- Group by input shape")
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
# 
# prof.export_chrome_trace("trace.json")
