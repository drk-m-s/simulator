#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file implements all profiling related classes and functions

from collections import OrderedDict
import time
import torch
import sys
import transformers

from .simulator import Simulator, layer_profile
from .utils import add_dictionaries

class layer_log:
    def __init__(
        self,
        uniq_id,
        name,
        context,
        summarization,
        start_time,
        input,
        weights,
        output,
        exec_time_ms,
        performance,
        energy_pj,
        transfer_bytes,
    ):
        self.id = uniq_id
        self.name = name
        self.context = context
        self.summarization = summarization
        self.start_time = start_time
        self.input = input
        self.weights = weights
        self.output = output
        self.exec_time_ms = exec_time_ms
        self.performance = performance
        self.energy = energy_pj
        self.transfer_bytes = transfer_bytes


class UPM_Profiler:
    def __init__(self, options):
        self.n_layers = 0
        self.n_executions = 0
        self.layers = {}
        self.functions = {}

    def set_options(self, options):
        self.options = options
        # simulation related
        self.simulation = options.simulation
        self.sim_compute = options.sim_compute
        self.sim_sliding_window = options.sim_sliding_window
        self.sim_num_key_value_heads = options.sim_num_key_value_heads

        self.sim_weights_data_type = options.sim_weights_data_type

        self.sim_activation_data_type = options.sim_activation_data_type

        self.sim_weights_data_type_bytes = {
            "int4": 0.5,
            "int8": 1,
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }.get(self.sim_weights_data_type)

        self.sim_activation_data_type_bytes = {
            "int4": 0.5,
            "int8": 1,
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }.get(self.sim_activation_data_type)

        self.simulator = None
        if self.simulation:
            self.simulator = self.create_arch_simulator()

    def create_arch_simulator(self) -> Simulator:
        return Simulator(
            weights_data_type_bytes=self.sim_weights_data_type_bytes,
            activation_data_type_bytes=self.sim_activation_data_type_bytes,
            sliding_window=self.sim_sliding_window,
            num_key_value_heads=self.sim_num_key_value_heads,
            verbose=self.options.sim_verbose,
        )

    def print_layers_model(self) -> None:
        print("##### Layers of Model in order of creation #####")
        print(
            "Layer ID (creation order), Context, Function, Dimensions (rows x columns), times executed, avg. execution time (ms)"
        )
        for i in self.layers.keys():
            print(
                str(self.layers[i].id)
                + ","
                + self.layers[i].context
                + ","
                + self.layers[i].name
                + ","
                + "("
                + str(self.layers[i].dim_in)
                + "x"
                + str(self.layers[i].dim_out)
                + "),"
                + str(self.layers[i].exec_nums)
                + ","
                + str(self.layers[i].exec_time / self.n_executions / 1e6)
            )

    def print_functions_model(self):
        print("##### Functions called by the Model in order of calling #####")
        print(
            "Function name, Context, Dimensions in (columns), Dimensions out (columns), times executed, avg. execution time (ms)"
        )
        for i in self.functions.keys():
            print(
                str(i)
                + ","
                + self.functions[i].context
                + ","
                + "("
                + str(self.functions[i].dim_in)
                + "),"
                + "("
                + str(self.functions[i].dim_out)
                + "),"
                + str(self.functions[i].exec_nums)
                + ","
                + str(self.functions[i].exec_time / 1e6)
            )

    def print_log_summary(self, show_summarization=False, show_all=False):
        phase = "Generation"
        if show_summarization:
            phase = "Summarization"
        if show_all:
            phase = "All (SUM and GEN)"
        print("#####", phase, "Execution summary #####")
        name_ctxt = []
        summary_time = OrderedDict()
        summary_perf = OrderedDict()
        summary_energy = OrderedDict()
        summary_transfer_bytes = OrderedDict()
        summary_nexec = OrderedDict()
        input_shapes = OrderedDict()
        weights_shapes = OrderedDict()
        output_shapes = OrderedDict()
        for log in self.log:
            if not show_all and not show_summarization and log.summarization:
                continue
            if not show_all and show_summarization and not log.summarization:
                continue
            ctxt = log.name + ":" + log.context
            if not ctxt in summary_time.keys():
                name_ctxt.append(ctxt)
            summary_nexec[ctxt] = 1 + summary_nexec.get(ctxt, 0)
            summary_time[ctxt] = log.exec_time_ms + summary_time.get(ctxt, 0)
            summary_energy[ctxt] = add_dictionaries(
                summary_energy.get(ctxt, {}), log.energy
            )
            summary_perf[ctxt] = add_dictionaries(
                summary_perf.get(ctxt, {}), log.performance
            )
            summary_transfer_bytes[ctxt] = add_dictionaries(
                summary_transfer_bytes.get(ctxt, {}), log.transfer_bytes
            )

            input_shapes[ctxt] = "(" + ":".join([str(x) for x in log.input]) + ")"
            weights_shapes[ctxt] = "(" + ":".join([str(x) for x in log.weights]) + ")"
            output_shapes[ctxt] = "(" + ":".join([str(x) for x in log.output]) + ")"

        executed_times = 1 if show_summarization else (self.n_executions - 1)
        print(
            "Function: Context: input shape: weights shape: output shape:"
            "time(s):H2C(ms):C2H(ms):compute(ms):mem_transfer(ms):kv_load(ms)"
            "host_to_device(mJ):device_to_host(mJ):main_mem(mJ):compute(mJ)"
        )
        for key in name_ctxt:

            perf_values = []
            for perf_key in [
                "host_to_device",
                "device_to_host",
                "compute",
                "mem_transfer",
                "kv_load",
            ]:
                perf_values.append(
                    str(summary_perf[key].get(perf_key, 0) / 1e6 / executed_times)
                )
            perf_string = ":".join(perf_values)

            energy_values = []
            for ene_key in ["host_to_device", "device_to_host", "main_mem", "compute"]:
                energy_values.append(
                    str(summary_energy[key].get(ene_key, 0) / 1e6 / executed_times)
                )
            energy_string = ":".join(energy_values)
            print(
                key,
                input_shapes[key],
                weights_shapes[key],
                output_shapes[key],
                (summary_time[key] / executed_times),
                perf_string,
                energy_string,
            )

        total_time_explained = sum(summary_time.values())
        total_percentage_explained = 0
        for key in name_ctxt:
            print(
                key,
                "explains",
                (summary_time[key] / total_time_explained) * 100,
                "% of the total inference time (num. executions:",
                summary_nexec[key],
                ") average time:",
                summary_time[key] / summary_nexec[key],
                "(ms)",
            )
            total_percentage_explained += (
                summary_time[key] / total_time_explained
            ) * 100
        print(
            "Profiler captures", total_percentage_explained, "% of the total execution"
        )
        print("Profiler captures", total_time_explained, "ms of the total execution")
        print(summary_time)

    def print_log(self):
        print("##### Execution log #####")
        print(
            "Start time, exec time, ID, Function, Context, input shape, weights shape, output shape"
        )
        for log in self.log:
            input_shape = "(" + ",".join([str(x) for x in log.input]) + ")"
            weights_shape = "(" + ",".join([str(x) for x in log.weights]) + ")"
            output_shape = "(" + ",".join([str(x) for x in log.output]) + ")"
            print(
                log.start_time / 1e6,
                log.exec_time_ms,
                log.id,
                log.name,
                log.context,
                input_shape,
                weights_shape,
                output_shape,
            )

    def update_inference_perf(self, step_perf):
        if self.simulator.sum:
            for key in step_perf.keys():
                self.sum_perf[key] = self.sum_perf.get(key, 0) + step_perf[key]
        else:
            for key in step_perf.keys():
                self.gen_perf[key] = self.gen_perf.get(key, 0) + step_perf[key]

    def update_inference_energy(self, step_energy):
        if self.simulator.sum:
            for key in step_energy.keys():
                self.sum_energy[key] = self.sum_energy.get(key, 0) + step_energy[key]
        else:
            for key in step_energy.keys():
                self.gen_energy[key] = self.gen_energy.get(key, 0) + step_energy[key]

    def update_inference_transfer_bytes(self, step_transfer_bytes):
        if self.simulator.sum:
            for key in step_transfer_bytes.keys():
                self.sum_transfer_bytes[key] = (
                    self.sum_transfer_bytes.get(key, 0) + step_transfer_bytes[key]
                )
        else:
            for key in step_transfer_bytes.keys():
                self.gen_transfer_bytes[key] = (
                    self.gen_transfer_bytes.get(key, 0) + step_transfer_bytes[key]
                )

    def start(
        self,
        layer_mapping={},
        layer_attn_ctxt="",
        last_layer="lm_head",
        batch_size=1,
        moe_end="",
        experts_per_token=2,
    ):
        self.start_inference = time.time_ns()
        self.n_executions = 0
        self.inference_time = 0
        self.summarization_time = 0
        self.sum_perf = {}
        self.gen_perf = {}
        self.sum_energy = {}
        self.gen_energy = {}
        self.sum_transfer_bytes = {}
        self.gen_transfer_bytes = {}

        self.last_layer = last_layer
        self.batch_size = batch_size

        self.layers_start = {}
        self.layers_end = {}
        self.log = []

        if self.simulation:
            self.simulator:Simulator = self.create_arch_simulator()
            self.simulator.map_layers(
                layer_mapping,
                layer_attn_ctxt=layer_attn_ctxt,
                moe_end=moe_end,
                experts_per_token=experts_per_token,
            )

    def end(self):
        if self.simulation:
            step_time, step_perf, step_energy, step_transfer_bytes = (
                self.simulator.simulate_end(
                    self.forward_input_shape, generated_tokens=(self.n_executions)
                )
            )
            self.inference_time += step_time
            self.update_inference_perf(step_perf)
            self.update_inference_energy(step_energy)
            self.update_inference_transfer_bytes(step_transfer_bytes)
        if not self.simulation:
            self.inference_time = time.time_ns() - self.start_inference

        inference_time_sec = self.inference_time / 1e9
        sum_energy_mJ = 0
        gen_energy_mJ = 0
        sum_time_s = self.summarization_time / 1e9
        gen_time_s = inference_time_sec - sum_time_s
        gen_n_executions = self.n_executions - 1

        print("##### UPMEM PROFILER OUTPUT #####")
        print(
            "Total time (SUM + GEN):",
            inference_time_sec,
            "with weights data type:",
            self.sim_weights_data_type,
            "activation data type:",
            self.sim_activation_data_type,
            "batch size:",
            self.batch_size,
        )
        print(
            "Generated tokens: ",
            gen_n_executions * self.batch_size,
            "in",
            gen_time_s,
            "seconds with tokens/s:",
            (gen_n_executions * self.batch_size) / gen_time_s,
        )
        print(
            "Summarization step took:",
            sum_time_s,
            "s, weight in the execution: SUM:",
            sum_time_s / inference_time_sec,
            "%, GEN:",
            gen_time_s / inference_time_sec,
            "%",
        )

        if self.simulation:
            print("SUMMARIZATION summary")
            for key in self.sum_transfer_bytes.keys():
                print(
                    "Transferred data in",
                    key,
                    self.sum_transfer_bytes[key] / 1e6,
                    "MB",
                )
            for key in self.sum_energy.keys():
                energy_mj = self.sum_energy[key] / 1e9
                print("Energy in", key, energy_mj, "mJ")
                sum_energy_mJ += self.sum_energy[key] / 1e9
            print("Energy:", sum_energy_mJ, "(mJ)")
            print("Power:", sum_energy_mJ / 1e3 / sum_time_s, "W")

            if gen_n_executions > 0:
                print("GENERATION summary")
                for key in self.gen_transfer_bytes.keys():
                    print(
                        "Transferred data in",
                        key,
                        self.gen_transfer_bytes[key] / 1e6,
                        "MB, MB/token:",
                        self.gen_transfer_bytes[key]
                        / 1e6
                        / self.n_executions
                        / self.batch_size,
                    )
                for key in self.gen_energy.keys():
                    energy_mj = self.gen_energy[key] / 1e9
                    print(
                        "Energy in",
                        key,
                        energy_mj,
                        "mJ, mJ/token:",
                        energy_mj / gen_n_executions / self.batch_size,
                    )
                    gen_energy_mJ += self.gen_energy[key] / 1e9
                print(
                    "Energy:",
                    gen_energy_mJ,
                    "(mJ) mJ/token:",
                    gen_energy_mJ / gen_n_executions / self.batch_size,
                )
                print(
                    "Power:",
                    gen_energy_mJ
                    / gen_n_executions
                    / 1e3
                    * gen_n_executions
                    / gen_time_s,
                    "W",
                )

            print("Execution time breakdown (ms / %)")
            print("SUMMARIZATION phase")
            for perf_key in [
                "host_to_device",
                "device_to_host",
                "compute",
                "mem_transfer",
                "kv_load",
            ]:
                perf_value = self.sum_perf.get(perf_key, 0)
                print(
                    perf_key, (perf_value / 1e6), "(ms)", perf_value / 1e9 / sum_time_s
                )

            if gen_n_executions > 0:
                print("GENERATION phase")
                for perf_key in [
                    "host_to_device",
                    "device_to_host",
                    "compute",
                    "mem_transfer",
                    "kv_load",
                ]:
                    perf_value = self.gen_perf.get(perf_key, 0)
                    print(
                        perf_key,
                        (perf_value / 1e6),
                        "(ms)",
                        perf_value / 1e9 / gen_time_s,
                    )

        if self.options.report_layers:
            self.print_layers_model()

        if self.options.report_functions:
            self.print_functions_model()

        if self.options.print_log:
            self.print_log()

        if self.options.print_log_summary:
            self.print_log_summary()
            self.print_log_summary(show_summarization=True)
            self.print_log_summary(show_all=True)

        print("##### END UPMEM PROFILER OUTPUT #####")

    def add(self, layer: torch.nn.Module, context: str) -> None:
        # print ("Profiling layer...")
        name_layer = ""
        dim_in = 0
        dim_out = 0
        if issubclass(torch.nn.Linear, type(layer)):
            name_layer = "Linear"
            dim_in = layer.in_features
            dim_out = layer.out_features
        elif issubclass(transformers.activations.NewGELUActivation, type(layer)):
            name_layer = "NewGELUActivation"
            dim_in = 1
            dim_out = 1
        elif issubclass(type(layer), torch.nn.SiLU):
            name_layer = "SiLUActivation"
            dim_in = 1
            dim_out = 1
        elif issubclass(
            type(layer), transformers.models.llama.modeling_llama.LlamaRMSNorm
        ):
            name_layer = "LlamaRMSNorm"
            dim_in = layer.weight.size()[0]
            dim_out = layer.weight.size()[0]
        elif issubclass(
            type(layer), transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
        ):
            name_layer = "LlamaRotaryEmbedding"
            dim_in = 1
            dim_out = 1
        elif issubclass(torch.nn.SiLU, type(layer)):
            name_layer = "SiLU"
            dim_in = 1
            dim_out = 1
        elif issubclass(torch.nn.LayerNorm, type(layer)):
            name_layer = "LayerNorm"
            dim_in = 1
            dim_out = 1
        elif issubclass(torch.nn.Embedding, type(layer)):
            name_layer = "Embedding"
            dim_in = 1  # layer.num_embeddings
            dim_out = 1  # layer.embedding_dim
        elif issubclass(torch.nn.Dropout, type(layer)):
            name_layer = "Dropout"
            dim_in = 1
            dim_out = 1
        elif issubclass(torch.nn.Softmax, type(layer)):
            name_layer = "Softmax"
            dim_in = layer.dim
            dim_out = layer.dim
        elif issubclass(transformers.pytorch_utils.Conv1D, type(layer)):
            name_layer = "Conv1D"
            dim_in = layer.weight.shape[0]
            dim_out = layer.weight.shape[1]
        elif issubclass(torch.nn.Conv2d, type(layer)):
            name_layer = "Conv2D"
            dim_in = layer.kernel_size[0]
            dim_out = layer.kernel_size[1]
        elif issubclass(torch.nn.Conv3d, type(layer)):
            name_layer = "Conv3D"
            dim_in = layer.kernel_size[0] #Not sure if correct
            dim_out = layer.kernel_size[1] #Not sure if correct
        elif issubclass(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed, type(layer)):
            name_layer = "Qwen2_5_VisionPatchEmbed"
            dim_in = 1 # Not sure if correct
            dim_out = 1 # Not sure if correct
        elif issubclass(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding, type(layer)):
            name_layer = "Qwen2_5_VisionRotaryEmbedding"
            dim_in = 1 # Not sure if correct
            dim_out = 1
        elif issubclass(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm, type(layer)):
            name_layer = "Qwen2RMSNorm"
            dim_in = 1 # Not sure if correct
            dim_out = 1
        elif issubclass(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding, type(layer)):
            name_layer = "Qwen2_5_VLRotaryEmbedding"
            dim_in = 1 # Not sure if correct
            dim_out = 1
        elif issubclass(torch.nn.GELU, type(layer)):
            name_layer = "GELU"
            dim_in = 1 # Not sure if correct
            dim_out = 1
        elif issubclass(transformers.models.glm4v.modeling_glm4v.Glm4vRMSNorm, type(layer)):
            name_layer = "Glm4vRMSNorm"
            dim_in = 1
            dim_out = 1
        elif issubclass(transformers.models.glm4v.modeling_glm4v.Glm4vVisionRotaryEmbedding, type(layer)):
            name_layer = "Glm4vVisionRotaryEmbedding"
            dim_in = 1
            dim_out = 1
        elif issubclass(transformers.models.glm4v.modeling_glm4v.Glm4vTextRotaryEmbedding, type(layer)):
            name_layer = "Glm4vTextRotaryEmbedding"
            dim_in = 1
            dim_out = 1
        elif issubclass(transformers.models.mistral.modeling_mistral.MistralRMSNorm, type(layer)):
            name_layer = "MistralRMSNorm"
            dim_in = 1
            dim_out = 1
        elif issubclass(transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding, type(layer)):
            name_layer = "MistralRotaryEmbedding"
            dim_in = 1
            dim_out = 1
        else:
            print("Layer:", type(layer), "not supported")
            sys.exit()
        self.layers[layer] = layer_profile(
            self.n_layers,
            name_layer,
            self.n_layers,
            context,
            dim_in,
            dim_out,
            obj=layer,
        )
        self.n_layers += 1

    def forward_start(self, input_shape):
        self.forward_input_shape = input_shape

        if self.simulation:
            self.forward_time_start = self.inference_time
        else:
            self.forward_time_start = time.time_ns()

    def forward_end(self, output_shape, context, layer_obj=None):
        self.forward_time_end = time.time_ns()

        weights_shape = torch.Size(
            [self.layers[layer_obj].dim_in, self.layers[layer_obj].dim_out]
        )

        cur_exec_time = self.forward_time_end - self.forward_time_start
        performance = {}
        energy = {}
        transfer_bytes = {}  # Initialize transfer_bytes here
        relative_start_time = self.forward_time_start - self.start_inference

        if self.simulation:
            if self.layers[layer_obj].name == "LlamaRMSNorm":
                cur_exec_time, performance, energy, transfer_bytes = (
                    self.simulator.simulate_function(
                        "LlamaRMSNorm",
                        context,
                        output_shape,
                        self.layers[layer_obj].dim_out,
                    )
                )
            elif self.layers[layer_obj].name == "SiLUActivation":
                cur_exec_time, performance, energy, transfer_bytes = (
                    self.simulator.simulate_function(
                        "SiLUActivation",
                        context,
                        output_shape,
                        self.layers[layer_obj].dim_out,
                    )
                )
            elif self.layers[layer_obj].name == "SiLU":
                cur_exec_time, performance, energy, transfer_bytes = (
                    self.simulator.simulate_function(
                        "SiLUActivation", context, output_shape, self.layers[layer_obj].dim_out
                    )
                )
            else:
                cur_exec_time, performance, energy, transfer_bytes = (
                    self.simulator.simulate_layer(
                        self.layers[layer_obj],
                        self.forward_input_shape,
                        layer_obj,
                        weights_shape,
                        output_shape,
                    )
                )  # or context?
            self.inference_time += cur_exec_time
            self.update_inference_perf(performance)
            self.update_inference_energy(energy)
            self.update_inference_transfer_bytes(transfer_bytes)

            self.layers[layer_obj].exec_time += cur_exec_time
            self.layers[layer_obj].energy = add_dictionaries(
                self.layers[layer_obj].energy, energy
            )
            self.layers_start[layer_obj] = self.forward_time_start
            self.layers_end[layer_obj] = self.inference_time
            relative_start_time = self.forward_time_start
        else:
            self.inference_time += cur_exec_time
            self.layers[layer_obj].exec_time += cur_exec_time
            self.layers_start[layer_obj] = self.forward_time_start
            self.layers_end[layer_obj] = self.forward_time_end

        self.layers[layer_obj].exec_nums += 1

        summarization_phase = False if not self.simulation else self.simulator.sum

        cur_logging = layer_log(
            self.layers[layer_obj].id,
            self.layers[layer_obj].name,
            self.layers[layer_obj].context,
            summarization_phase,
            relative_start_time / 1e6,
            self.forward_input_shape,
            weights_shape,
            output_shape,
            cur_exec_time / 1e6,
            performance,
            energy,
            transfer_bytes,
        )

        self.log.append(cur_logging)
        if self.layers[layer_obj].context == self.last_layer:
            if self.simulation and not self.simulator.sum:
                self.simulator.sum_size += 1
            if self.n_executions == 0:
                if self.simulation:
                    self.simulator.start_gen()
                    self.simulator.sum_size = (
                        output_shape[-2] if (len(output_shape) > 1) else 1
                    )
                self.summarization_time = self.inference_time
            self.n_executions += 1
            print("New token generated...")

    def forward_func_start(self, name, context, input_shape):
        self.func_input_shape = input_shape

        self.start_func = time.time_ns()
        if self.simulation:
            self.start_func = self.inference_time

    def forward_func_end(self, name, context, output_shape):
        self.end_func = time.time_ns()

        func_profile = self.functions.get(
            name,
            layer_profile(
                0, name, 0, context, self.func_input_shape[-1], output_shape[-1]
            ),
        )

        cur_exec_time = self.end_func - self.start_func
        performance = {}
        energy = {}
        relative_time = self.start_func - self.start_inference

        if self.simulation:
            cur_exec_time, performance, energy, transfer_bytes = (
                self.simulator.simulate_function(
                    name, context, self.func_input_shape, output_shape
                )
            )
            self.inference_time += cur_exec_time
            self.update_inference_perf(performance)
            self.update_inference_energy(energy)
            self.update_inference_transfer_bytes(transfer_bytes)

        relative_exec_time = self.start_func

        summarization_phase = 0 if not self.simulation else self.simulator.sum
        cur_logging = layer_log(
            0,  # functions have ID set to 0
            name,
            context,
            summarization_phase,
            relative_exec_time / 1e6,
            self.func_input_shape,
            output_shape,
            output_shape,
            cur_exec_time / 1e6,
            performance,
            energy,
            transfer_bytes,
        )

        self.log.append(cur_logging)

        func_profile.exec_nums += 1
        func_profile.exec_time += cur_exec_time
        self.functions[name] = func_profile
