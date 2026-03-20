import argparse
import sys
import time
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel, AutoTokenizer

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

layer_mapping = {
        # Vision encoder layers (SiglipVisionTransformer)
        "patch_embedding"  : options.device,
        "position_embedding": options.device,
        "layer_norm1"      : options.device,
        "layer_norm2"      : options.device,
        "post_layernorm"   : options.device,
        "fc1"              : options.device,
        "fc2"              : options.device,
        "out_proj"         : options.device,
        
        # Resampler layers
        "kv_proj"          : options.device,
        "ln_q"             : options.device,
        "ln_kv"            : options.device,
        "ln_post"          : options.device,
        
        # LLM layers (Qwen3ForCausalLM)
        "embed_tokens"     : options.device,
        "q_proj"           : options.device,
        "k_proj"           : options.device,
        "v_proj"           : options.device,
        "o_proj"           : options.device,
        "q_norm"           : options.device,
        "k_norm"           : options.device,
        "gate_proj"        : options.device,
        "up_proj"          : options.device,
        "down_proj"        : options.device,
        "act_fn"           : options.device,
        "input_layernorm"  : options.device,
        "post_attention_layernorm": options.device,
        "norm"             : options.device,
        "rotary_emb"       : options.device,
        "lm_head"          : options.device,
        
        # Additional common layer names
        "mlp"              : options.device,
}

torch.manual_seed(100)

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5', trust_remote_code=True)

print(model)

# Load image from URL
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert('RGB')

def generate_exact_text(tokenizer, options):
    dummy_token = "describe"
    # Start with a reasonable guess
    repeat = options.in_tokens // len(tokenizer(dummy_token, add_special_tokens=False)["input_ids"])
    while True:
        dummy_text = " ".join([dummy_token] * repeat)
        # Test tokenization
        tokens = tokenizer(dummy_text, return_tensors="pt")
        num_tokens = tokens.input_ids.shape[1]
        if num_tokens < options.in_tokens:
            repeat += 1
        elif num_tokens > options.in_tokens:
            repeat -= 1
        else:
            return dummy_text

# Generate dummy text sequence to match desired input token count
dummy_text = generate_exact_text(tokenizer, options)

# Prepare messages with dummy text
question = dummy_text
msgs = [{'role': 'user', 'content': [image, question]}]

# Count text-only tokens (without image) using chat template
text_only_msgs = [{'role': 'user', 'content': question}]
try:
    # Get tokenized ids directly
    text_only_ids = tokenizer.apply_chat_template(
        text_only_msgs, tokenize=True, return_tensors="pt", add_generation_prompt=True
    )
    text_token_count = text_only_ids.shape[-1]
except Exception:
    # Fallback: get rendered prompt then tokenize
    text_only_prompt = tokenizer.apply_chat_template(
        text_only_msgs, tokenize=False, add_generation_prompt=True
    )
    text_only_tokens = tokenizer(text_only_prompt, return_tensors="pt")
    text_token_count = text_only_tokens.input_ids.shape[1]

print(f"Text tokens: {text_token_count}")

# Get the actual processed inputs with image to count total tokens
captured_input_ids = None
total_input_tokens = None
image_tokens = None

try:
    original_generate = model.llm.generate

    def capture_generate(*args, **kwargs):
        global captured_input_ids
        if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
            captured_input_ids = kwargs['input_ids']
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            captured_input_ids = args[0]
        raise StopIteration()

    model.llm.generate = capture_generate

    try:
        model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            enable_thinking=False,
            stream=False,
            max_new_tokens=1,
        )
    except StopIteration:
        pass
    finally:
        model.llm.generate = original_generate

    if captured_input_ids is not None:
        total_input_tokens = captured_input_ids.shape[1]
        image_tokens = total_input_tokens - text_token_count
        print(f"Image tokens: {image_tokens}")
        print(f"Total input tokens: {total_input_tokens}")
    else:
        print("Warning: Could not capture exact token count.")
        total_input_tokens = text_token_count
        image_tokens = 0

except Exception as e:
    print(f"Warning: Could not capture exact token count ({e}).")
    total_input_tokens = text_token_count
    image_tokens = 0

enable_thinking=False
stream=False  # Set to False for simulation

# Modified profiler_start call with visual_end parameter
upmem_layers.profiler_start(
    layer_mapping, 
    batch_size=options.bs,
    visual_end="ln_post",  # Last layer of resampler before LLM starts
    last_layer="lm_head"
)

print(f"Simulating with device: {options.device}")
print(f"in: {total_input_tokens} (text: {text_token_count} + image: {image_tokens}), out: {options.out_tokens}")

# Generate output
answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    enable_thinking=enable_thinking,
    stream=stream,
    max_new_tokens=options.out_tokens,
    min_new_tokens=options.out_tokens
)

upmem_layers.profiler_end()

# Handle both streaming and non-streaming output
if stream:
    generated_text = ""
    for new_text in answer:
        generated_text += new_text
        print(new_text, flush=True, end='')
    print()
else:
    print(answer)