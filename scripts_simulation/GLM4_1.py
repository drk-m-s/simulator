import torch
import transformers
import sys
import argparse
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch
from PIL import Image
import requests
from io import BytesIO

# Add your local src directory to Python path
sys.path.insert(0, '/home/han.jiang/simulator/src')

import upmem_llm_framework.pytorch_upmem_layers as upmem_layers
upmem_layers.profiler_init()

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="unknown")
parser.add_argument("--in-tokens", default=1000, type=int)
parser.add_argument("--out-tokens", default=100, type=int)
parser.add_argument("--bs", default=1, type=int)
options, _ = parser.parse_known_args()

your_token="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"

layer_mapping = {
        "position_embedding"        : options.device,
        "proj"                      : options.device,
        "rotary_pos_emb"            : options.device,
        "norm1"                     : options.device,
        "norm2"                     : options.device,
        "qkv"                       : options.device,
        "proj"                      : options.device,
        "gate_proj"                 : options.device,
        "up_proj"                   : options.device,
        "down_proj"                 : options.device,
        "act_fn"                    : options.device,
        "proj"                      : options.device,
        "post_projection_norm"      : options.device,
        "gate_proj"                 : options.device,
        "up_proj"                   : options.device,
        "down_proj"                 : options.device,
        "act1"                      : options.device,
        "act_fn"                    : options.device,
        "post_conv_layernorm"       : options.device,
        "downsample"                : options.device,
        "post_layernorm"            : options.device,
        "embed_tokens"              : options.device,
        "q_proj"                    : options.device,
        "k_proj"                    : options.device,
        "v_proj"                    : options.device,
        "o_proj"                    : options.device,
        "gate_up_proj"              : options.device,
        "down_proj"                 : options.device,
        "activation_fn"             : options.device,
        "input_layernorm"           : options.device,
        "post_attention_layernorm"  : options.device,
        "post_self_attn_layernorm"  : options.device,
        "post_mlp_layernorm"        : options.device,
        "norm"                      : options.device,
        "rotary_emb"                : options.device,
        "lm_head"                   : options.device
}

MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

def generate_exact_text(processor, options):
    dummy_token = "describe"
    repeat = options.in_tokens // len(processor.tokenizer(dummy_token, add_special_tokens=False)["input_ids"])
    while True:
        dummy_text = " ".join([dummy_token] * repeat)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        "resized_height": 480,
                        "resized_width": 640,
                    },
                    {"type": "text", "text": dummy_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_only_inputs = processor(
            text=[text],
            images=None,
            padding=True,
            return_tensors="pt",
        )
        num_tokens = text_only_inputs.input_ids.shape[1]
        if num_tokens < options.in_tokens:
            repeat += 1
        elif num_tokens > options.in_tokens:
            repeat -= 1
        else:
            print("Text tokens:", num_tokens)
            return dummy_text

# Usage:
dummy_text = generate_exact_text(processor, options)

def download_and_resize_image(url, height, width):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((width, height), Image.BICUBIC)
    return img

# Example usage:
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
resized_height = 364
resized_width = 448
resized_image = download_and_resize_image(image_url, resized_height, resized_width)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": resized_image,
                # "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                # "resized_height": 480,
                # "resized_width": 640,
            },
            {
                "type": "text",
                "text": dummy_text
            }
        ],
    }
]


model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Print total input tokens
print("Total input tokens:", inputs["input_ids"].shape[1])
print("in:", inputs["input_ids"].shape[1], "out:", options.out_tokens)

# Modified profiler_start call with visual_end parameter
upmem_layers.profiler_start(
    layer_mapping, 
    batch_size=options.bs,
    visual_end="post_layernorm",  # Last unique layer before language model starts
    last_layer="lm_head"  # Last layer of language model
)

print(f"Simulating with device: {options.device}")

generated_ids = model.generate(**inputs, min_new_tokens=options.out_tokens, max_new_tokens=options.out_tokens+1)

upmem_layers.profiler_end()

output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)

# print (model)
# for name, module in model.named_modules():
#     print(name, type(module))