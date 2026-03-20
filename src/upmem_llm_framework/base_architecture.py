#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file implements Base_architecture class
# This class contains a default implementation of the following functions:
#   - get_tflops(input_shape, weight_shape): returns the TFLOPs required in a MxM
#   - get_moved_data_bytes(input_shape, weight_shape, load_input = True, load_weight = True): returns the required bytes to move in order to do an operation
#   - host_transfer(input_shape, direction = "to_device"): simulates a data transfer with host in any direction
#   - compute_ns(input_shape, weight_shape, load_input = True, load_weight = True): simulates the computation of a MxM
#   - compute_matmul_ns(context, shapeA, shapeB, use_kv_cache = True, summarization = False, sum_size = 0): simulates the computation of a matmul for self-attention
#   - compute_activation_ns(data_shape, dimension, activation = "SiLU"): simulates an activation layer
#   - compute_RMSNorm_ns(data_shape, dimension): simulates a RMSNorm layer
#   - compute_softmax_ns(data_shape): simulates a softmax operation
#
# Note that all simulations returns compute_time_ns, performance_dict, energy_dict:
#   - compute_time_ns: the simulated time in ns,
#   - performance_dict: dictionary containing the simulated time in ns for each operation simulated,
#   - energy_dict: dictionary containing the simulated energy in pJ for each operation simulated.

import torch
import math
from .utils import add_dictionaries
import transformers

class Base_architecture:

    def __init__(
        self,
        active_chips=1,
        tflops=1,
        pj_per_tflop=1,
        host_to_device_bw_GBs=1,
        device_to_host_bw_GBs=1,
        # inter_bw               = 1,
        memory=1,
        mem_bw_GBs=1,
        mem_pj_per_bit=1,
        # data_type_bytes=2,  # float16
        weights_data_type_bytes=2,  # float16
        activation_data_type_bytes=2,  # float16
        # 3000 cycles per row of 2048 elements --> 1.4 cycles / element
        # assuming 1 GHz, 1.5 ns / element, parallelized accross 4 chips -> 0.37
        softmax_ns_per_element=0.4,  # ns, considering it cycles in 1GHz config
        SiLU_ns_per_element=0.6,  # ns, softmax * 1.5 (empiric number based on execution of Llama2-7b)
        RMSNorm_ns_per_element=1.1,  # ns, softmax * 2.6 (empiric number based on execution of Llama2-7b)
        # 3000 cycles per row of 2048 elements with 5 TFLOPs of computing power
        # assuming 1 GHz, 0,000003 s --> 3MOPS per row of 2048 --> 1.5kOPS per element
        misc_tflops_per_element=1500 / 1e12,
        sliding_window=-1,
        num_key_value_heads=-1,
        verbose=False,
    ):

        self.active_chips = active_chips
        # Compute capabilities
        self.tflops = tflops
        # self.pj_per_tflop = 0.4 #commmented out by Han 
        self.pj_per_tflop = pj_per_tflop

        # Interface with HOST
        self.host_to_device_bw_GBs = host_to_device_bw_GBs
        self.device_to_host_bw_GBs = device_to_host_bw_GBs
        self.host_to_device_pj_per_bit = 25
        self.device_to_host_pj_per_bit = 25
        # self.inter_bw                 = inter_bw

        # Device memory (shared memory like)
        self.memory = memory  # unused at the moment
        self.mem_bw_GBs = mem_bw_GBs
        self.mem_pj_per_bit = mem_pj_per_bit

        # self.data_type_bytes = data_type_bytes
        self.weights_data_type_bytes = weights_data_type_bytes
        self.activation_data_type_bytes = activation_data_type_bytes

        self.softmax_ns_per_element = softmax_ns_per_element
        self.RMSNorm_ns_per_element = RMSNorm_ns_per_element
        self.SiLU_ns_per_element = SiLU_ns_per_element

        self.misc_tflops_per_element = misc_tflops_per_element

        self.sliding_window = sliding_window
        self.num_key_value_heads = num_key_value_heads

        self.verbose = verbose

    # Defined TFLOPS are defined for float16,
    # assume that performance is doubled if data type is demoted
    def adjust_for_quantization(self):
        ratio = 2 / self.weights_data_type_bytes  # Assume pj_per_tflop corresponds to float16
        self.pj_per_tflop = self.pj_per_tflop / ratio
        # self.tflops = self.tflops * (2 / self.data_type_bytes)

        # self.softmax_ns_per_element = self.softmax_ns_per_element / ratio
        # self.RMSNorm_ns_per_element = self.RMSNorm_ns_per_element / ratio
        # self.SiLU_ns_per_element = self.SiLU_ns_per_element / ratio

    def get_tflops(self, input_shape, weight_shape):
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1

        tflops = (
            2 * batch_size * n_heads * n_rows * weight_shape[1] * weight_shape[0]
        ) / 1e12

        return tflops

    def get_tflops_Linear(self, input_shape, weight_shape):
        out_features = weight_shape[0]
        tflops = input_shape.numel() * out_features * 2 / 1e12
        return tflops

     # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    def get_tflops_Conv2d(self, input_shape, layer, weight_shape):
        # Extract dimensions from input_shape
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        in_channels = input_shape[-3] if (len(input_shape) > 2) else 1
        input_height = input_shape[-2] if (len(input_shape) > 1) else 1
        input_width = input_shape[-1]
        
        # Get output channels from weight shape
        out_channels = weight_shape[0]
        
        # Handle string padding values like 'valid' or 'same'
        padding = layer.padding
        if isinstance(padding, str):
            if padding.lower() == 'valid':
                padding = (0, 0)
            elif padding.lower() == 'same':
                # For 'same' padding, calculate based on kernel size
                padding = (layer.kernel_size[0] // 2, layer.kernel_size[1] // 2)
            else:
                padding = (0, 0)

        # Ensure padding is a tuple
        if isinstance(padding, int):
            padding = (padding, padding)

        # Ensure stride is a tuple
        stride = layer.stride
        if isinstance(stride, int):
            stride = (stride, stride)
    
        # Ensure kernel_size is a tuple
        kernel_size = layer.kernel_size
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h = kernel_size[0]
            kernel_w = kernel_size[1]

        # Calculate output dimensions
        output_height = (input_height + 2 * padding[0] - kernel_h) // stride[0] + 1
        output_width = (input_width + 2 * padding[1] - kernel_w) // stride[1] + 1

        # TFLOPS calculation for Conv2d:
        # For each output element, we perform kernel_h * kernel_w * in_channels multiply-accumulate operations
        # Each MAC = 2 FLOPs (1 multiply + 1 add)
        # Total = batch_size * out_channels * output_height * output_width * kernel_h * kernel_w * in_channels * 2
        tflops = (
            2 * batch_size * out_channels * output_height * output_width * 
            kernel_h * kernel_w * in_channels
        ) / 1e12

        return tflops

    def get_tflops_Conv3d(self, input_shape, layer, weight_shape):
        # Extract dimensions from input_shape
        batch_size = input_shape[-5] if (len(input_shape) > 4) else 1
        in_channels = input_shape[-4] if (len(input_shape) > 3) else 1
        input_depth = input_shape[-3] if (len(input_shape) > 2) else 1
        input_height = input_shape[-2] if (len(input_shape) > 1) else 1
        input_width = input_shape[-1]

        # Get output channels from weight shape
        out_channels = weight_shape[0]

        # Handle padding
        padding = layer.padding
        if isinstance(padding, str):
            if padding.lower() == 'valid':
                padding = (0, 0, 0)
            elif padding.lower() == 'same':
                kernel_size = layer.kernel_size
                if isinstance(kernel_size, int):
                    padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2)
                else:
                    padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
            else:
                padding = (0, 0, 0)
        elif isinstance(padding, int):
            padding = (padding, padding, padding)

        # Handle stride
        stride = layer.stride
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        # Get kernel dimensions
        kernel_size = layer.kernel_size
        if isinstance(kernel_size, int):
            kernel_d = kernel_h = kernel_w = kernel_size
        else:
            kernel_d = kernel_size[0]
            kernel_h = kernel_size[1]
            kernel_w = kernel_size[2]

        # Calculate output dimensions
        output_depth = (input_depth + 2 * padding[0] - kernel_d) // stride[0] + 1
        output_height = (input_height + 2 * padding[1] - kernel_h) // stride[1] + 1
        output_width = (input_width + 2 * padding[2] - kernel_w) // stride[2] + 1

        # TFLOPS calculation for Conv3d:
        # For each output element, we perform kernel_d * kernel_h * kernel_w * in_channels MAC operations
        # Each MAC = 2 FLOPs
        tflops = (
            2 * batch_size * out_channels * output_depth * output_height * output_width *
            kernel_d * kernel_h * kernel_w * in_channels
        ) / 1e12

        return tflops

    def get_tflops_LayerNorm(self, input_shape):
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        tflops = (
            batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element
        )

        return tflops

    def get_tflops_by_layer(self, input_shape, layer, weight_shape):
        if issubclass(torch.nn.Conv2d, type(layer)):
            tflops = self.get_tflops_Conv2d(input_shape, layer, weight_shape)
        elif issubclass(torch.nn.Conv3d, type(layer)):
            tflops = self.get_tflops_Conv3d(input_shape, layer, weight_shape)
        elif issubclass(torch.nn.LayerNorm, type(layer)):
            tflops = self.get_tflops_LayerNorm(input_shape)
        elif issubclass(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm, type(layer)):
            tflops = self.get_tflops_LayerNorm(input_shape)
        elif issubclass(torch.nn.Linear, type(layer)):
            tflops = self.get_tflops_Linear(input_shape, weight_shape)
        else:
            # Treat everything else as Linear
            tflops = self.get_tflops_Linear(input_shape, weight_shape)
            #print ("get_tflops not defined for layer: ", type(layer))
            # sys.exit(-1)
        return tflops

    def get_moved_data_bytes(
        self, input_shape, weight_shape, load_input=False, load_weight=True
    ):
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        weight_size = weight_shape[1] * weight_shape[0] if load_weight else 0
        input_size = batch_size * n_heads * n_rows * n_columns if load_input else 0
        output_size = batch_size * n_rows * weight_shape[1]

        weight_bytes = self.weights_data_type_bytes * weight_size
        input_bytes = self.activation_data_type_bytes * input_size
        output_bytes = self.activation_data_type_bytes * output_size

        return weight_bytes + input_bytes + output_bytes

    # KV cache load
    def load_data(self, input_shape: torch.Size) -> tuple[float, dict, dict]:
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        data_size_bytes = self.activation_data_type_bytes * (
            batch_size * n_rows * n_heads * n_columns
        )
        # B / (GB/s) --> s/G --> ns
        transfer_time_ns = data_size_bytes / self.mem_bw_GBs

        performance = {"kv_load": transfer_time_ns}
        # GB/s * time --> GB * pJ/bit --> J
        energy = {"main_mem": data_size_bytes * 8 * self.mem_pj_per_bit}

        if self.verbose:
            print(
                "Load time for input_shape:",
                input_shape,
                "=",
                transfer_time_ns,
                "(ns) with",
                energy,
                "pj",
                performance,
            )

        return transfer_time_ns, performance, energy

    def host_transfer(self, input_shape, direction="to_device", generated_tokens=1):
        '''
        Simulates a data transfer with host in any direction.
        The input_shape is a torch.Size object, and the direction can be "to_device"
        or "to_host". The generated_tokens parameter is used to calculate the total
        number of tokens to be transferred.
        '''
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        bandwidth = (
            self.host_to_device_bw_GBs
            if direction == "to_device"
            else self.device_to_host_bw_GBs
        )

        data_size_bytes = self.activation_data_type_bytes * (
            batch_size * n_heads * n_rows * n_columns * generated_tokens
        )
        # B / (GB/s) --> s / G --> ns
        transfer_time_ns = data_size_bytes / bandwidth

        name_op = "host_to_device" if direction == "to_device" else "device_to_host"

        performance = {name_op: transfer_time_ns}
        moved_data = {name_op: data_size_bytes}

        ddr_pj_per_bit = (
            self.host_to_device_pj_per_bit
            if direction == "to_device"
            else self.device_to_host_pj_per_bit
        )
        energy = {name_op: data_size_bytes * 8 * ddr_pj_per_bit}

        if self.verbose:
            print(
                "Transfer time for input_shape:",
                input_shape,
                "=",
                transfer_time_ns,
                "(ns) with",
                energy,
                "pj perf:",
                performance,
                "data in bytes:",
                moved_data,
            )

        return transfer_time_ns, performance, energy, moved_data

    ## This is for Linear Layers
    def compute_ns(self, input_shape, weight_shape, load_input=False, load_weight=True):
        tflops = self.get_tflops(input_shape, weight_shape)
        data_size_bytes = self.get_moved_data_bytes(
            input_shape, weight_shape, load_input=load_input, load_weight=load_weight
        )

        compute_time_ns = (tflops / self.tflops) * 1e9
        transfer_time_ns = data_size_bytes / self.mem_bw_GBs
        real_time_ns = max(compute_time_ns, transfer_time_ns)

        performance = {
            "compute": compute_time_ns,
            "mem_transfer": transfer_time_ns,
        }

        energy = {
            "compute": tflops * self.pj_per_tflop,
            "main_mem": data_size_bytes * 8 * self.mem_pj_per_bit,
        }

        if self.verbose:
            print(
                "Computing",
                input_shape,
                "x",
                weight_shape,
                "with TFLOPS:",
                tflops,
                "with",
                data_size_bytes,
                "bytes",
            )
            print(
                "takes",
                real_time_ns,
                "(ns) with",
                compute_time_ns,
                "(ns) in compute and ",
                transfer_time_ns,
                "(ns) in loading data",
            )
            print(
                "and consumes",
                energy["compute"],
                ",",
                energy["main_mem"],
                "pJ for compute and loading, respectively",
            )
            print("performance:", performance)

        return real_time_ns, performance, energy


    def compute_ns_by_layer(self, input_shape, layer_obj, weight_shape, load_input=False, load_weight=True):
        
        if self.verbose:
            print(f"Compute_ns_by_layer {layer_obj} with type {type(layer_obj)}")
        tflops = self.get_tflops_by_layer(input_shape, layer_obj, weight_shape)

        data_size_bytes = self.get_moved_data_bytes(
            input_shape, weight_shape, load_input=load_input, load_weight=load_weight
        )

        compute_time_ns = (tflops / self.tflops) * 1e9
        transfer_time_ns = data_size_bytes / self.mem_bw_GBs
        real_time_ns = max(compute_time_ns, transfer_time_ns)

        performance = {
            "compute": compute_time_ns,
            "mem_transfer": transfer_time_ns,
        }

        energy = {
            "compute": tflops * self.pj_per_tflop,
            "main_mem": data_size_bytes * 8 * self.mem_pj_per_bit,
        }

        if self.verbose:
            print(
                f"Computing {input_shape} x {weight_shape} with TFLOPS: {tflops} "
                f"with {data_size_bytes} bytes"
            )
            print(
                f"takes {real_time_ns} ns with {compute_time_ns} in compute and {transfer_time_ns} "
                f"in loading data"
            )
            print(
                f"and consumes {energy['compute']} pJ for compute and {energy['main_mem']} pJ "
                "for loading data"
            )
            print(f"performance: {performance}")

        return real_time_ns, performance, energy

    def compute_scaled_dot_product_ns(
        self,
        context,
        key_shape,  # same dimensions as value_shape
        output_shape,
        use_kv_cache=True,
        summarization=False,
        sum_size=0,
    ):
        batch_size = key_shape[-4] if (len(key_shape) > 3) else 1
        n_heads = key_shape[-3] if (len(key_shape) > 2) else 1
        n_rows = key_shape[-2] if (len(key_shape) > 1) else 1  # already concatenated!
        n_columns = key_shape[-1]

        q_rows = (
            output_shape[-2] if (len(output_shape) > 2) else 1
        )  # 1 when using kv cache in GEN.

        compute_time_ns = 0
        load_k_time = 0
        load_v_time = 0
        performance = {}
        energy = {}

        # Load KV cache if GENeration and kv cache is enabled
        # Only K is required for next step
        if not summarization and use_kv_cache:
            if self.sliding_window != -1:
                n_rows = self.sliding_window
            if self.num_key_value_heads != -1:
                n_heads = self.num_key_value_heads

            k_cache = torch.Size([batch_size, n_heads, n_rows, n_columns])
            load_k_time, load_k_perf, load_k_energy = self.load_data(k_cache)
            performance = add_dictionaries(performance, load_k_perf)
            energy = add_dictionaries(energy, load_k_energy)

        # Q x Kt
        # (q_rows, embedding) x (embedding, kv_cache_length) = (q_rows, kv_cache_length)
        query_shape = torch.Size([batch_size, n_heads, q_rows, n_columns])
        kt_shape = torch.Size([batch_size, n_heads, n_columns, n_rows])
        step_time, step_perf, step_energy = self.compute_ns(
            query_shape, kt_shape, load_input=False, load_weight=False
        )
        compute_time_ns += max(
            load_k_time, step_time
        )  # overlap loading K with Q x Kt computation
        performance = add_dictionaries(performance, step_perf)
        energy = add_dictionaries(energy, step_energy)

        # Load KV cache if GENeration and kv cache is enabled
        # Only V is required for next step
        if not summarization and use_kv_cache:
            if self.sliding_window != -1:
                n_rows = self.sliding_window
            if self.num_key_value_heads != -1:
                n_heads = self.num_key_value_heads

            v_cache = torch.Size([batch_size, n_heads, n_rows, n_columns])
            load_v_time, load_v_perf, load_v_energy = self.load_data(v_cache)
            performance = add_dictionaries(performance, load_v_perf)
            energy = add_dictionaries(energy, load_v_energy)

        # QxKT x V
        # (q_rows, kv_cache_length) x (kv_cache_length, embedding) = (q_rows, embedding)
        qxkt_shape = torch.Size([batch_size, n_heads, q_rows, n_rows])
        step_time, step_perf, step_energy = self.compute_ns(
            query_shape, key_shape, load_input=False, load_weight=False
        )
        compute_time_ns += max(
            load_v_time, step_time
        )  # overlap loading V with QxKt x V computation
        performance = add_dictionaries(performance, step_perf)
        energy = add_dictionaries(energy, step_energy)

        if self.verbose:
            print(
                "compute_scaled_dot_product_ns:",
                query_shape,
                "x",
                kt_shape,
                "x",
                key_shape,
                "in",
                compute_time_ns,
                "with",
                energy,
            )

        return compute_time_ns, performance, energy

    def compute_matmul_ns(
        self,
        context,
        shapeA,
        shapeB,
        use_kv_cache=True,
        summarization=False,
        sum_size=0,
    ):
        batch_size = shapeA[-4] if (len(shapeA) > 3) else 1
        n_heads = shapeA[-3] if (len(shapeA) > 2) else 1
        n_rows = shapeA[-2] if (len(shapeA) > 1) else 1
        n_columns = shapeA[-1]

        compute_time_ns = 0
        performance = {}
        energy = {}

        if context == "attn_weights":
            if not summarization:
                kv_cache = torch.Size([batch_size, n_heads, sum_size, n_columns * 2])
                step_time, step_perf, step_energy = self.load_data(kv_cache)
                compute_time_ns += step_time
                performance = add_dictionaries(performance, step_perf)
                energy = add_dictionaries(energy, step_energy)

        step_time, step_perf, step_energy = self.compute_ns(
            shapeA, shapeB, load_input=False, load_weight=False
        )

        compute_time_ns += step_time
        performance = add_dictionaries(performance, step_perf)
        energy = add_dictionaries(energy, step_energy)

        if self.verbose:
            print(
                "compute_matmul_ns:",
                shapeA,
                "x",
                shapeB,
                "in",
                compute_time_ns,
                "with",
                energy,
            )

        return compute_time_ns, performance, energy

    def compute_activation_ns(self, data_shape, dimension, activation="SiLU"):
        batch_size = data_shape[-4] if (len(data_shape) > 3) else 1
        n_heads = data_shape[-3] if (len(data_shape) > 2) else 1
        n_rows = data_shape[-2] if (len(data_shape) > 1) else 1
        n_columns = data_shape[-1]

        activation_ns_per_element = 0
        if activation == "SiLU":
            activation_ns_per_element = self.SiLU_ns_per_element

        tflops = (
            batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element
        )
        compute_time_ns = (
            batch_size * n_heads * (n_rows * (activation_ns_per_element * n_columns))
        )

        performance = {"compute": compute_time_ns}
        energy = {"compute": tflops * self.pj_per_tflop}

        if self.verbose:
            print(
                "compute_activation_ns:",
                activation,
                ":",
                data_shape,
                "in",
                compute_time_ns,
                "with",
                energy,
            )

        return compute_time_ns, performance, energy

    def compute_RMSNorm_ns(self, data_shape, dimension):
        batch_size = data_shape[-4] if (len(data_shape) > 3) else 1
        n_heads = data_shape[-3] if (len(data_shape) > 2) else 1
        n_rows = data_shape[-2] if (len(data_shape) > 1) else 1
        n_columns = dimension

        tflops = (
            batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element
        )
        compute_time_ns = (
            batch_size * n_heads * n_rows * (self.RMSNorm_ns_per_element * n_columns)
        )

        performance = {"compute": compute_time_ns}
        energy = {"compute": tflops * self.pj_per_tflop}

        if self.verbose:
            print(
                "compute_RMSNorm_ns:", data_shape, "in", compute_time_ns, "with", energy
            )

        return compute_time_ns, performance, energy

    def compute_softmax_ns(self, data_shape):
        batch_size = data_shape[-4] if (len(data_shape) > 3) else 1
        n_heads = data_shape[-3] if (len(data_shape) > 2) else 1
        n_rows = data_shape[-2] if (len(data_shape) > 1) else 1
        n_columns = data_shape[-1]

        tflops = (
            batch_size * n_heads * n_rows * n_columns * self.misc_tflops_per_element
        )
        compute_time_ns = (
            batch_size * n_heads * n_rows * (self.softmax_ns_per_element * n_columns)
        )

        performance = {"compute": compute_time_ns}
        energy = {"compute": tflops * self.pj_per_tflop}

        if self.verbose:
            print(
                "compute_softmax_ns:", data_shape, "in", compute_time_ns, "with", energy
            )

        return compute_time_ns, performance, energy
