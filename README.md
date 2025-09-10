AI Chip LLM Simulation Framework
===================
Based on UPMEM Framework for profiling / simulation
==========================================

This library allows
1. Profiling PyTorch neural networks on a x86 server,
2. Simulate the execution of the neural network in a target hardware accelerator.

In order to use it:

1. Import the pytorch_upmem_layers library and initialize it before creating the neural network:
```
import pytorch_upmem_layers as upmem_layers
upmem_layers.profiler_init()
```

2. Call the profiler when doing a forward pass / inference:
```
upmem_layers.profiler_start()
prediction = model.forward(myTensor)
upmem_layers.profiler_end()
```
You can find examples to use it when building a model in PyTorch in nn_example.py and when loading a model through HuggingFace in hf_example.py.

## Profiler
The profiler records the start time and end time of a computation layer or function.
Currently, real power consumption is not tracked on the x86 server.

The profiler identifies a layer or function by 4 parameters:
1. Layer type (e.g. Linear module) or function (e.g. softmax),
2. Context when the layer or function is called, meaning the variable name assigned to the layer or function (e.g. `q_proj = torch.nn.Linear(...)` has a context of *q_proj*),
3. the input dimensions of the layer or function,
4. specifically for layer, an unique id is assigned when a layer is initialized.

### Profiler output
By default, the profiler reports at the end of execution a summary with execution time, energy (when simulating), and power consumption (when simulating).
When simulating, this summary is brokedown in summarization (encoding) phase and generation (decoding) phase.

More information can be shown by enabling the following flags:

* `--report-layers`: reports the created layers in the neural network with its associated parameters
* `--report-functions`: reports the called functions during the forward pass of the neural network with its associated parameters
* `--print-log`: a detailed log of each layer and function executed during the forward pass of the neural network is printed ordered by time

## Simulation
To run a simulation, it is needed to provide a dictionary mapping layer with a device or hardware accelerator.
This dictionary is composed of `name of layer:device,options`.
The name of the layer corresponds to the context concept introduced before.
The device corresponds to a one of the classes defined in `sim_architectures.py`.
Current options supported are:
* 't' or transfer point: the input of the layer where this option is specified shall come from CPU, which means that the last device should send back to the CPU its results and the CPU should send to the device specified for this layer its input.
* 'm' or moe transfer point: the input of the layer where this option is specified shall come from CPU but only once since the input is share across different MoEs.
For instance, for a neural network composed of 2 Linear layers that is executed sequentially in different chips:
```
layer_mapping = {
    "linear1":"PIM-AI-1chip,t",
    "linear2":"PIM-AI-1chip,t",
}

upmem_layers.profiler_start(layer_mapping)
prediction = model.forward(myTensor)
upmem_layers.profiler_end()
```
This mapping corresponds to the following scheme
```
CPU --> PIM-AI-1chip (execute linear1) --> CPU --> PIM-AI-1chip (execute linear2)
     |                                          |
     |--> input of linear1 is sent              |--> output of linear1 is sent to CPU
          to PIM-AI-1chip device                     and input of linear2 is sent to PIM-AI-1chip device
```

### Running a simulation
Once the layer mapping has been specified, to run a simulation:
```
python3 ./hf_example.py --simulation
```

### Adding a hardware accelerator
The file `sim_architectures.py` contains multiple hardware accelerator profiles implemented as classes inhereting from `Base_architecture` defined in file `base_architecture.py`.
`Base_architecture` class implements the simulated operations and data transfers required by the profiler in order to simulate the behavior of layers and functions.

To add a new hardware accelerator profile, a new class should be added to `sim_architectures.py` with its profile.
It is possible to add optimizations on top of the operations defined by `Base_architecture`.
A basic new profile is provided below:
```
class new_device(Base_architecture):

    def __init__(self, verbose=False):
        self.name = "new device"
        super().__init__(verbose=verbose)

        # HOST communication
        self.host_to_device_bw_GBs = 12.8
        self.host_to_device_pj_per_bit = 20
        self.device_to_host_bw_GBs = 12.8
        self.device_to_host_pj_per_bit = 20

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5
        self.pj_per_tflop = 0.4 * 1e12
```

Note that `new_device` class is referenced as `new-device` in the layer mapping.

### Notes on simulation

Several assumptions are made to simplify the modelling of the execution across hardware profiles:
1. no interconnection communication is modelled: it is assumed that the inter-communication between devices can be overlapped with compute / hidden due to be fast enough. For instance, when simulating more than one GPU, exchanging required data between them is not modelled. For an AI-PIM device (DIMM), communication within a DIMM is not modelled neither.
2. Peak performance can be reached always. All hardware profiles will be performing operations at their peak performance even in some cases is unrealistic to think so. Adding a performance ratio to model this is left for future work.

Installation
------------

### Set up environment

Python 3.8 is supported. We suggest using pipenv to set up a virtual environment:

```
pipenv --python 3.8
pipenv shell
```
Now your shell should be using Python 3.8 as expected.

### Build package

First, the package needs to be built
```
python3 -m build
```

### Install package

```
pip3 install $ROOT_PROJECT/dist/upmem_llm_framework-0.0.1.tar.gz
```
