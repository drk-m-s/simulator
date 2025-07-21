import torch
import transformers
import sys
import argparse
from PIL import Image
import requests
from io import BytesIO

your_token="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"

# Qwen2.5-VL-3B-Instruct
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# Add your local src directory to Python path
sys.path.insert(0, '/home/han.jiang/simulator/src')

import upmem_llm_framework.pytorch_upmem_layers as upmem_layers
upmem_layers.profiler_init()

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="unknown")
parser.add_argument("--in-tokens", default=64, type=int)
parser.add_argument("--out-tokens", default=30, type=int)
parser.add_argument("--bs", default=1, type=int)
options, _ = parser.parse_known_args()

layer_mapping = {
        "proj"            : options.device,
        "rotary_pos_emb"  : options.device,
        "norm1"           : options.device,
        "norm2"           : options.device,
        "qkv"             : options.device,
        "gate_proj"       : options.device,
        "up_proj"         : options.device,
        "down_proj"       : options.device,
        "act_fn"          : options.device,
        "q_proj"	      : options.device,
        "k_proj"		  : options.device,
        "rotatory_emb"    : options.device,
        "v_proj"          : options.device,
        "o_proj"		  : options.device,
        "input_layernorm" : options.device,
        "post_attention_layernorm": options.device,
        "norm"            : options.device,
        "lm_head"         : options.device
}


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",use_fast=False)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


# # Download and resize the image
# img_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
# response = requests.get(img_url)
# img = Image.open(BytesIO(response.content)).convert("RGB")
# img = img.resize((468, 364))  # width, height

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image like I am a 4 year old and don't know much about anything."},
        ],
    }
]

upmem_layers.profiler_start(layer_mapping, batch_size=options.bs)

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

print("Total input tokens:", inputs.input_ids.shape[1])

# print ("Simulating with device...", options.device)
print ("in:", inputs.input_ids.shape[1], "out:", options.out_tokens)


# Inference: Generation of the output
generated_ids = model.generate(**inputs, min_new_tokens=options.out_tokens, max_new_tokens=options.out_tokens+1)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

upmem_layers.profiler_end()

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
