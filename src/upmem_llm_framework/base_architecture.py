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
        data_type_bytes=2,  # float16
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

        self.data_type_bytes = data_type_bytes

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
        ratio = 2 / self.data_type_bytes  # Assume pj_per_tflop corresponds to float16
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
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_channels = input_shape[-3] if (len(input_shape) > 2) else 1
        n_height = input_shape[-2] if (len(input_shape) > 1) else 1
        n_width = input_shape[-1]

        stride = layer.stride
        padding = layer.padding

        n_height = n_height + 2 * (
            padding[0] if isinstance(padding, tuple) else padding
        )
        n_width = n_width + 2 * (padding[1] if isinstance(padding, tuple) else padding)

        # Example of how many times a kernel is applied depending on stride:
        # | 0 | 1 | 2 | 3 | 4 | 5 |
        # Stride 1:
        # 00001111
        #     11112222
        #         22223333
        #             33334444
        #                 44445555
        # Stride 2:
        # 00001111
        #         22223333
        #                 44445555

        width_times = math.ceil(
            (n_width - 1) / (stride[1] if isinstance(stride, tuple) else stride)
        )
        height_times = math.ceil(
            (n_width - 1) / (stride[0] if isinstance(stride, tuple) else stride)
        )

        # TFLOPS when applying once the kernel
        tflops_kernel = (
            2 * batch_size * n_channels * weight_shape[1] * weight_shape[0]
        ) / 1e12

        tflops = tflops_kernel * width_times * height_times

        return tflops

    def get_tflops_Conv3d(self, input_shape, layer, weight_shape):
        batch_size = input_shape[-5] if (len(input_shape) > 4) else 1
        n_channels = input_shape[-4] if (len(input_shape) > 3) else 1
        n_depth = input_shape[-3] if (len(input_shape) > 2) else 1
        n_height = input_shape[-2] if (len(input_shape) > 1) else 1
        n_width = input_shape[-1]

        stride = layer.stride
        padding = layer.padding

        # Apply padding to all three spatial dimensions
        n_depth = n_depth + 2 * (
            padding[0] if isinstance(padding, tuple) else padding
        )
        n_height = n_height + 2 * (
            padding[1] if isinstance(padding, tuple) else padding
        )
        n_width = n_width + 2 * (
            padding[2] if isinstance(padding, tuple) else padding
        )

        # Calculate how many times the kernel is applied in each dimension
        # Based on the Conv3d output size formulas:
        # D_out = floor((D_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
        # For simplicity, we're using the approximation from Conv2d
        depth_times = math.ceil(
            (n_depth - 1) / (stride[0] if isinstance(stride, tuple) else stride)
        )
        height_times = math.ceil(
            (n_height - 1) / (stride[1] if isinstance(stride, tuple) else stride)
        )
        width_times = math.ceil(
            (n_width - 1) / (stride[2] if isinstance(stride, tuple) else stride)
        )

        # TFLOPS when applying the kernel once
        # For Conv3d: batch_size * in_channels * kernel_depth * kernel_height * kernel_width * out_channels * 2
        # weight_shape typically contains [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
        tflops_kernel = (
            2 * batch_size * n_channels * weight_shape[2] * weight_shape[3] * weight_shape[4] * weight_shape[0]
        ) / 1e12

        # Total TFLOPS considering all applications of the kernel
        tflops = tflops_kernel * depth_times * height_times * width_times

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
        else:
            # Treat everything else as Linear
            tflops = self.get_tflops_Linear(input_shape, weight_shape)
            print ("get_tflops not defined for layer: ", type(layer))
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
        # output_size = batch_size * n_rows * weight_shape[1]
        return self.data_type_bytes * (weight_size + input_size)  # + output_size)

    # KV cache load
    def load_data(self, input_shape: torch.Size) -> tuple[float, dict, dict]:
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        data_size_bytes = self.data_type_bytes * (
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
        batch_size = input_shape[-4] if (len(input_shape) > 3) else 1
        n_heads = input_shape[-3] if (len(input_shape) > 2) else 1
        n_rows = input_shape[-2] if (len(input_shape) > 1) else 1
        n_columns = input_shape[-1]

        bandwidth = (
            self.host_to_device_bw_GBs
            if direction == "to_device"
            else self.device_to_host_bw_GBs
        )

        data_size_bytes = self.data_type_bytes * (
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
                "Computing scaled_dot_product:",
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
                "Computing matmul:",
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
                "Computing Activation",
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
                "Computing RMSNorm:", data_shape, "in", compute_time_ns, "with", energy
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
                "Computing softmax:", data_shape, "in", compute_time_ns, "with", energy
            )

        return compute_time_ns, performance, energy
