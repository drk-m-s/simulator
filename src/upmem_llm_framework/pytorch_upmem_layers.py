#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file wraps PyTorch classes and functions into new UPM classes and functions able to
# track the start, inputs, end and outputs of the corresponding function.
# Currently, forward from multiple modules and other minor functions
# (normalizations, softmax, activations, etc.) are tracked and profiled.

import argparse
from .profiler import UPM_Profiler
import torch
from torch import Tensor
from typing import Optional, Union  # , Tuple
from inspect import getframeinfo, stack
from torch.nn.common_types import _size_1_t

import transformers

options = None
profiler = None
profiling = 0


def get_arguments():
    parser = argparse.ArgumentParser()
    # Verbose related
    parser.add_argument(
        "--report-layers",
        action="store_true",
        help="Enable reporting metrics for all executed layers at the end of the forward pass.",
    )
    parser.add_argument(
        "--report-functions",
        action="store_true",
        help="Enable reporting metrics for all executed functions at the end of the forward pass.",
    )
    parser.add_argument(
        "--print-log",
        action="store_true",
        help="Print a trace of the execution of layers and functions.",
    )
    parser.add_argument(
        "--print-log-summary",
        action="store_true",
        help="Print a detailed summary of each layer and function executed. For summarization, generation, and both.",
    )
    # Simulation related
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Enable simulation according to the layer mapping defined",
    )
    parser.add_argument(
        "--sim-compute",
        action="store_true",
        help="Simulate compute intensive operations. Note that some operations are still performed due to constraints in inputs/outputs of other layer/functions. CAUTION: Output tokens will be affected",
    )
    parser.add_argument(
        "--sim-weights-data-type",
        choices=["int4", "int8", "float16", "bfloat16", "float32"],
        default="int4",
        help="Set the datatype for weights.",
    )
    parser.add_argument(
        "--sim-activation-data-type",
        choices=["int4", "int8", "float16", "bfloat16", "float32"],
        default="int4--sim",
        help="Set the datatype for activations.",
    )
    parser.add_argument(
        "--sim-num-key-value-heads",
        type=int,
        default=-1,
        help="When using GQA, this value is used to simulate fetching the correct KV caches.",
    )
    parser.add_argument(
        "--sim-sliding-window",
        type=int,
        default=-1,
        help="When set, a sliding window is simulated according to this value. Note that the real underlying execution will run according to the model parameter.",
    )
    parser.add_argument(
        "--sim-verbose",
        action="store_true",
        help="Set a verbose mode for simulation",
    )
    toReturn, _ = parser.parse_known_args()
    return toReturn


def get_context():
    # https://stackoverflow.com/questions/24438976/debugging-get-filename-and-line-number-from-which-a-function-is-called
    return getframeinfo(stack()[2][0]).code_context[0].split()[0].replace("self.", "")


class UPM_Module(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        return x


class UPM_Linear(torch.nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        if options.sim_compute:
            shape = list(x.shape)
            shape[-1] = self.out_features
            x = torch.zeros(shape)
        else:
            x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_NonDynamicallyQuantizableLinear(
    torch.nn.modules.linear.NonDynamicallyQuantizableLinear
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())
        print("HERE")

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        if options.sim_compute:
            shape = list(x.shape)
            shape[-1] = self.out_features
            x = torch.zeros(shape)
        else:
            x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_LayerNorm(torch.nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_Embedding(torch.nn.Embedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_LlamaRotaryEmbedding(
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x, position_ids):
        context = get_context()
        shape = x.shape
        profiler.forward_start(shape)
        x = super().forward(x, position_ids)
        profiler.forward_end(shape, context, layer_obj=self)  # TODO: x is a tuple
        return x


class UPM_LlamaRMSNorm(transformers.models.llama.modeling_llama.LlamaRMSNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


# class UPM_SiLUActivation(transformers.activations.SiLUActivation):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         profiler.add(self, get_context())

#     def forward(self, x):
#         context = get_context()
#         profiler.forward_start(x.shape)
#         x = super().forward(x)
#         profiler.forward_end(x.shape, context, layer_obj=self)
#         return x

class UPM_SiLUActivation(torch.nn.SiLU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x

class UPM_NewGELUActivation(transformers.activations.NewGELUActivation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x

# Not used in inference
class UPM_Dropout(torch.nn.Dropout):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context)
        return x


class UPM_Conv1d(torch.nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())


class UPM_Conv1D(transformers.pytorch_utils.Conv1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context)
        return x

class UPM_Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


class UPM_Softmax(torch.nn.Softmax):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler = profiler
        profiler.add(self, get_context())

    def forward(self, x):
        profiler.forward_start()
        x = super().forward(x)
        profiler.forward_end()
        return x


class UPM_Tensor(torch.Tensor):

    def transpose(self, input, dim0, dim1):
        print("UPMTranpose with input:", input, "dim0", dim0, "dim1", dim1)
        super(UPM_Tensor, self).transpose(input, dim0, dim1)


## For Qwen2.5-VL
class UPM_Conv3D(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x

class UPM_Qwen2_5_VisionPatchEmbed(
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x
    
class UPM_Qwen2_5_VisionRotaryEmbedding(
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, seqlen):
        context = get_context()
        profiler.forward_start(torch.Size([seqlen]))
        x = super().forward(seqlen)
        profiler.forward_end(torch.Size([seqlen]), context, layer_obj=self)
        return x
    
class UPM_Qwen2RMSNorm(
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, hidden_states):
        context = get_context()
        profiler.forward_start(hidden_states.shape)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * hidden_states.to(input_dtype)
        profiler.forward_end(out.shape, context, layer_obj=self)
        return out

class UPM_GELU(torch.nn.GELU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x
    
class UPM_Qwen2_5_VLRotaryEmbedding(
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x, position_ids):
        context = get_context()
        shape = x.shape
        profiler.forward_start(shape)
        x = super().forward(x, position_ids)
        profiler.forward_end(shape, context, layer_obj=self)  # TODO: x is a tuple
        return x

## For Qwen2-VL

class UPM_Qwen2VLRotaryEmbedding(transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x, position_ids):
        context = get_context()
        shape = x.shape
        profiler.forward_start(shape)
        x = super().forward(x, position_ids)
        profiler.forward_end(shape, context, layer_obj=self)  # TODO: x is a tuple
        return x
    
class UPM_Qwen2RMSNorm(transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x):
        context = get_context()
        profiler.forward_start(x.shape)
        x = super().forward(x)
        profiler.forward_end(x.shape, context, layer_obj=self)
        return x


# For GLM4.1v-9B-Thinking
class UPM_Glm4vRMSNorm(transformers.models.glm4v.modeling_glm4v.Glm4vRMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())
    
    def forward(self, hidden_states):
        context = get_context()
        profiler.forward_start(hidden_states.shape)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * hidden_states.to(input_dtype)
        profiler.forward_end(out.shape, context, layer_obj=self)
        return out
    
class UPM_Glm4vVisionRotaryEmbedding(transformers.models.glm4v.modeling_glm4v.Glm4vVisionRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, seqlen: int) -> torch.Tensor:
        context = get_context()
        profiler.forward_start(torch.Size([seqlen]))
        x = super().forward(seqlen)
        profiler.forward_end(torch.Size([seqlen]), context, layer_obj=self)
        return x

class UPM_Glm4vTextRotaryEmbedding(transformers.models.glm4v.modeling_glm4v.Glm4vTextRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x, position_ids):
        context = get_context()
        shape = x.shape
        profiler.forward_start(shape)
        cos, sin = super().forward(x, position_ids)
        profiler.forward_end(shape, context, layer_obj=self)
        return cos, sin

## For Mistral-7B-Instruct-v0.3

class UPM_MistralRMSNorm(transformers.models.mistral.modeling_mistral.MistralRMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, hidden_states):
        context = get_context()
        profiler.forward_start(hidden_states.shape)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * hidden_states.to(input_dtype)
        profiler.forward_end(out.shape, context, layer_obj=self)
        return out

class UPM_MistralRotaryEmbedding(transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiler.add(self, get_context())

    def forward(self, x, position_ids):
        context = get_context()
        shape = x.shape
        profiler.forward_start(shape)
        x = super().forward(x, position_ids)
        profiler.forward_end(shape, context, layer_obj=self)
        return x


__pytorch_nn_functional_softmax = torch.nn.functional.softmax

# TODO: change logic here to not use stringly types
def UPM_Softmax_functional(input, dim=None, dtype=None):
    context = get_context()
    profiler.forward_func_start("softmax", context, input.shape)
    x = __pytorch_nn_functional_softmax(input, dim=dim, dtype=dtype)
    profiler.forward_func_end("softmax", context, x.shape)
    return x

__pytorch_matmul = torch.matmul

# TODO: here too
def UPM_Matmul(input, other, *, out=None):
    context = get_context()
    profiler.forward_func_start("matmul", context, input.shape)
    x = __pytorch_matmul(input, other, out=out)
    profiler.forward_func_end("matmul", context, x.shape)
    return x


__pytorch_scaled_dot_product_attention = (
    torch.nn.functional.scaled_dot_product_attention
)

# TODO: here too
def UPM_scaled_dot_product_attention(query, key, value, *args, **kwargs):
    context = get_context()
    profiler.forward_func_start("scaled_dot_product_attention", context, key.shape)
    if options.sim_compute:
        q_shape = list(query.shape)
        v_shape = list(value.shape)
        q_shape[-1] = v_shape[-1]
        x = torch.zeros(q_shape)
    else:
        x = __pytorch_scaled_dot_product_attention(query, key, value, *args, **kwargs)
    profiler.forward_func_end("scaled_dot_product_attention", context, x.shape)
    return x


__pytorch_transpose = torch.transpose


def UPM_Transpose(input, dim0, dim1):
    context = get_context()
    print("UPM_Transpose with input", input.shape, "dim0:", dim0, "dim1", dim1)
    x = __pytorch_transpose(input, dim0, dim1)
    return x


def profiler_init():

    global options
    options = get_arguments()

    global profiling, profiler
    profiling = 1
    profiler = UPM_Profiler(options)

    # torch library
    # torch.nn.Module = UPM_Module #This is a problem
    torch.nn.Linear = UPM_Linear
    torch.nn.modules.linear.NonDynamicallyQuantizableLinear = (UPM_NonDynamicallyQuantizableLinear)
    torch.nn.LayerNorm = UPM_LayerNorm
    torch.nn.Embedding = UPM_Embedding
    torch.nn.Dropout = UPM_Dropout
    torch.nn.Conv1d = UPM_Conv1d
    torch.nn.Conv2d = UPM_Conv2d
    torch.nn.Softmax = UPM_Softmax
    torch.nn.functional.softmax = UPM_Softmax_functional
    torch.matmul = UPM_Matmul
    torch.transpose = UPM_Transpose
    torch.nn.functional.scaled_dot_product_attention = UPM_scaled_dot_product_attention
    torch.nn.SiLU = UPM_SiLUActivation

    # # transformers library
    transformers.pytorch_utils.Conv1D = UPM_Conv1D
    transformers.activations.NewGELUActivation = UPM_NewGELUActivation
    transformers.activations.ACT2FN["gelu_new"] = (UPM_NewGELUActivation)  # classes are hardcoded in ACT2FN     )
    transformers.activations.ACT2FN["silu"] = (UPM_SiLUActivation) # classes are hardcoded in ACT2FN
    transformers.models.llama.modeling_llama.LlamaRMSNorm = UPM_LlamaRMSNorm
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = (UPM_LlamaRotaryEmbedding)
    transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm = UPM_LlamaRMSNorm

    # #qwen2.5-vl
    torch.nn.Conv3d = UPM_Conv3D
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed = UPM_Qwen2_5_VisionPatchEmbed # this should not be included. It is a Conv3d inside.
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding = UPM_Qwen2_5_VisionRotaryEmbedding
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm = UPM_Qwen2RMSNorm
    torch.nn.GELU = UPM_GELU
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding = UPM_Qwen2_5_VLRotaryEmbedding

    # GLM4.1v-9B-Thinking
    transformers.models.glm4v.modeling_glm4v.Glm4vRMSNorm = UPM_Glm4vRMSNorm
    transformers.models.glm4v.modeling_glm4v.Glm4vVisionRotaryEmbedding = UPM_Glm4vVisionRotaryEmbedding
    transformers.models.glm4v.modeling_glm4v.Glm4vTextRotaryEmbedding = UPM_Glm4vTextRotaryEmbedding

    # Mistral-7B-Instruct-v0.3
    transformers.models.mistral.modeling_mistral.MistralRMSNorm = UPM_MistralRMSNorm
    transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding = UPM_MistralRotaryEmbedding

def profiler_start(
    layer_mapping={},
    layer_attn_ctxt="",
    last_layer="lm_head",
    batch_size=1,
    moe_end="",
    experts_per_token=2,
    visual_end="",  # Add visual_end parameter
):
    global options
    profiler.set_options(options)
    profiler.start(
        layer_mapping=layer_mapping,
        layer_attn_ctxt=layer_attn_ctxt,
        last_layer=last_layer,
        batch_size=batch_size,
        moe_end=moe_end,
        experts_per_token=experts_per_token,
        visual_end=visual_end,  # Pass visual_end to profiler.start()
    )


def profiler_end():
    profiler.end()
