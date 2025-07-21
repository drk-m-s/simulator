#
# Copyright (c) 2014-2024 - UPMEM
# UPMEM S.A.S France property - UPMEM confidential information covered by NDA
# For UPMEM partner internal use only - no modification allowed without permission of UPMEM
#
# This file implements multiple hardware architectures to be simulated.
# All architecture inherit from the Base_architecture class.
# If an architecture has optimizations for a given operation defined in Base_architecture, define them here

from .base_architecture import Base_architecture
from .utils import add_dictionaries


class HOST(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "HOST"
        super().__init__(*args, **kwargs)


class DGX100(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "DGX100-H100 SXM"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 450
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 450
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 26800
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 7916
        self.pj_per_tflop = 0.5 * 1e12


class V100(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "V100"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 64
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 64
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 900
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 112
        self.pj_per_tflop = 0.5 * 1e12


class H100_x3(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "H100_x3"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 64
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 64
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 2000 * 3
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 756.5 * 3
        self.pj_per_tflop = 0.5 * 1e12

        # Assuming a H100 is equivalent to 128 AI PIM cores (8 DIMMs) due to server size
        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4) / 3
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4) / 3
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4) / 3


class H100_x2(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "H100_x2"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 64
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 64
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 2000 * 2
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 756.5 * 2
        self.pj_per_tflop = 0.5 * 1e12

        # Assuming a H100 is equivalent to 128 AI PIM cores (8 DIMMs) due to server size
        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4) / 2
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4) / 2
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4) / 2


class A800(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "A800"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 64
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 64
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 1500
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 312
        self.pj_per_tflop = 0.5 * 1e12

        # Assuming a A800 is equivalent to 128 AI PIM cores (8 DIMMs) due to server size
        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4)
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4)


class H20(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "H20"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 64
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 64
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 4000
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 148
        self.pj_per_tflop = 0.5 * 1e12

        # Assuming a H20 is equivalent to 128 AI PIM cores (8 DIMMs) due to server size
        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4)
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4)


class H200(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "H200"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 64
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 64
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 2860
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 989
        self.pj_per_tflop = 0.5 * 1e12

        # Assuming a H100 is equivalent to 128 AI PIM cores (8 DIMMs) due to server size
        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4)
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4)


class H100(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "H100"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 64
        self.host_to_device_pj_per_bit = 27
        self.device_to_host_bw_GBs = 64
        self.device_to_host_pj_per_bit = 27

        # Device memory (shared memory like)
        self.mem_bw_GBs = 2000
        self.mem_pj_per_bit = 7

        # Compute
        self.tflops = 756.5
        self.pj_per_tflop = 0.5 * 1e12

        # Assuming a H100 is equivalent to 128 AI PIM cores (8 DIMMs) due to server size
        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4)
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4)


class A6000(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "A6000"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 32
        self.host_to_device_pj_per_bit = 35
        self.device_to_host_bw_GBs = 32
        self.device_to_host_pj_per_bit = 35

        # Device memory (shared memory like)
        self.mem_bw_GBs = 768
        self.mem_pj_per_bit = 15

        # Compute
        self.tflops = 155
        self.pj_per_tflop = 0.5 * 1e12


class A17Pro(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "A17Pro"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 51.2
        self.host_to_device_pj_per_bit = 20
        self.device_to_host_bw_GBs = 51.2
        self.device_to_host_pj_per_bit = 20

        # Device memory (shared memory like)
        self.mem_bw_GBs = 51.2
        self.mem_pj_per_bit = 20

        # Compute
        self.tflops = 4.3
        if self.data_type_bytes == 0.5:
            self.tflops = 35
        self.pj_per_tflop = 0.4 * 1e12


class Dimensity9300(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "Dimensity9300"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 76.8
        self.host_to_device_pj_per_bit = 10
        self.device_to_host_bw_GBs = 76.8
        self.device_to_host_pj_per_bit = 10

        # Device memory (shared memory like)
        self.mem_bw_GBs = 76.8
        self.mem_pj_per_bit = 10

        # Compute
        self.tflops = 6
        if self.data_type_bytes == 0.5:
            self.tflops = 33
        self.pj_per_tflop = 0.4 * 1e12


class Snapdragon8gen3(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "Snapdragon8gen3"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 77
        self.host_to_device_pj_per_bit = 10
        self.device_to_host_bw_GBs = 77
        self.device_to_host_pj_per_bit = 10

        # Device memory (shared memory like)
        self.mem_bw_GBs = 77
        self.mem_pj_per_bit = 10

        # Compute
        self.tflops = 4.73
        if self.data_type_bytes == 0.5:
            self.tflops = 34
        self.pj_per_tflop = 0.4 * 1e12

class WD1(Base_architecture):
    
    def __init__(self, *args, **kwargs):
        self.name = "WD1"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 1600
        self.host_to_device_pj_per_bit = 2
        self.device_to_host_bw_GBs = 1600
        self.device_to_host_pj_per_bit = 2

        # Device memory (shared memory like)
        self.mem_bw_GBs = 1600
        self.mem_pj_per_bit = 0.67

        # Compute
        self.tflops = 32
        if self.data_type_bytes == 0.5:
            self.tflops = 64
        self.pj_per_tflop = 0.4 * 1e12

class WD1_600(Base_architecture):
    
    def __init__(self, *args, **kwargs):
        self.name = "WD1_600"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 1100
        self.host_to_device_pj_per_bit = 2
        self.device_to_host_bw_GBs = 1100
        self.device_to_host_pj_per_bit = 2

        # Device memory (shared memory like)
        self.mem_bw_GBs = 1100
        self.mem_pj_per_bit = 0.67

        # Compute
        self.tflops = 19
        if self.data_type_bytes == 0.5:
            self.tflops = 38
        self.pj_per_tflop = 0.4 * 1e12

class Tenstorrent(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "Tenstorrent"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 448
        self.host_to_device_pj_per_bit = 10
        self.device_to_host_bw_GBs = 448
        self.device_to_host_pj_per_bit = 10

        # Device memory (shared memory like)
        self.mem_bw_GBs = 448
        self.mem_pj_per_bit = 20

        # Compute
        self.tflops = 30
        if self.data_type_bytes == 0.5:
            self.tflops = 60
        self.pj_per_tflop = 0.4 * 1e12


class SAM_LPDDR5PIM(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "SAM_LPDDR5PIM"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 12.8
        self.host_to_device_pj_per_bit = 22
        self.device_to_host_bw_GBs = 12.8
        self.device_to_host_pj_per_bit = 22

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 0.1024
        if self.data_type_bytes == 0.5:
            self.tflops = 4 * self.tflops
        self.pj_per_tflop = 0.8 * 1e12

class PIM_AI_1chip(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-1chip"
        super().__init__(*args, **kwargs)

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
        if self.data_type_bytes == 0.5:
            self.tflops = 32
        self.pj_per_tflop = 0.4 * 1e12


class PIM_AI_4chip(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-4chip"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 12.8
        self.host_to_device_pj_per_bit = 20 * 4
        self.device_to_host_bw_GBs = 12.8 * 4
        self.device_to_host_pj_per_bit = 20

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 4
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 4
        if self.data_type_bytes == 0.5:
            self.tflops = 32 * 4
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = self.softmax_ns_per_element / 4
        self.SiLU_ns_per_element = self.SiLU_ns_per_element / 4
        self.RMSNorm_ns_per_element = self.RMSNorm_ns_per_element / 4

class PIM_AI_2chip(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-2chip"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 12.8
        self.host_to_device_pj_per_bit = 20 * 2
        self.device_to_host_bw_GBs = 12.8 * 2
        self.device_to_host_pj_per_bit = 20

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 2
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 2
        if self.data_type_bytes == 0.5:
            self.tflops = 32 * 2
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = self.softmax_ns_per_element / 2
        self.SiLU_ns_per_element = self.SiLU_ns_per_element / 2
        self.RMSNorm_ns_per_element = self.RMSNorm_ns_per_element / 2


class PIM_AI_1dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-1dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 44
        self.host_to_device_pj_per_bit = 50
        self.device_to_host_bw_GBs = 44
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 16
        self.pj_per_tflop = 0.4 * 1e12


class PIM_AI_2dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-2dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 22
        self.host_to_device_pj_per_bit = 50 * 2
        self.device_to_host_bw_GBs = 44
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 2
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 16 * 2
        self.pj_per_tflop = 0.4 * 1e12


class PIM_AI_4dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-4dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 22
        self.host_to_device_pj_per_bit = 50 * 4
        self.device_to_host_bw_GBs = 44 * 2
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 4
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 16 * 4
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = 0.4 / (16 * 4)
        self.SiLU_ns_per_element = 0.6 / (16 * 4)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 4)


class PIM_AI_24dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-24dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 22
        self.host_to_device_pj_per_bit = 50 * 24
        self.device_to_host_bw_GBs = 44 * 12
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 24
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 8 * 16 * 24
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = 0.4 / (16 * 24)
        self.SiLU_ns_per_element = 0.6 / (16 * 24)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 24)


class PIM_AI_16dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-16dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 22
        self.host_to_device_pj_per_bit = 50 * 16
        self.device_to_host_bw_GBs = 44 * 8
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 16
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 8 * 16 * 16
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = 0.4 / (16 * 16)
        self.SiLU_ns_per_element = 0.6 / (16 * 16)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 16)


class PIM_AI_8dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-8dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 22
        self.host_to_device_pj_per_bit = 50 * 8
        self.device_to_host_bw_GBs = 44 * 4
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 8
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 8 * 16 * 8
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = 0.4 / (16 * 8)
        self.SiLU_ns_per_element = 0.6 / (16 * 8)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 8)


class PIM_AI_6dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-6dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 22
        self.host_to_device_pj_per_bit = 50
        self.device_to_host_bw_GBs = 44 * 3
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 6
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 16 * 6
        self.pj_per_tflop = 0.4 * 1e12


class CXL_PIM_BC(Base_architecture):
    # CXL board with:
    # 8-lane full duplex PCIe GEN5
    # 16 LPDDR controllers, 16 bits, 9.6 GT/s, dual rank (2 devices per IFC)
    # A device is a stack of 4 LPDDR-PIM
    # 256 GB overall memory (stacking 4 LPDDR-PIM = 8 dies)
    # This might be seen as 8x AI PIM DIMM with C2C connection between groups of 4 chips
    # Broadcast between LPDDR-PIM is possible

    def __init__(self, *args, **kwargs):
        self.name = "CXL_PIM_BC"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = (
            19.2  # 8-lane PCIe GEN5, but only one LPDDR5 at a time
        )
        self.host_to_device_pj_per_bit = (
            50  # crossing PCIe and LPDDR interfaces on both host and device
        )
        self.device_to_host_bw_GBs = (
            19.2  # 8-lane PCIe GEN5, but only one LPDDR5 at a time
        )
        self.device_to_host_pj_per_bit = (
            50  # crossing PCIe and LPDDR interfaces on both host and device
        )

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 2 * 4
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 16 * 2 * 4
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4)
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4)


class CXL_PIM_nBC(Base_architecture):
    # CXL board with:
    # 8-lane full duplex PCIe GEN5
    # 16 LPDDR controllers, 16 bits, 9.6 GT/s, dual rank (2 devices per IFC)
    # A device is a stack of 4 LPDDR-PIM
    # 256 GB overall memory (stacking 4 LPDDR-PIM = 8 dies)
    # This might be seen as 8x AI PIM DIMM with C2C connection between groups of 4 chips
    # Broadcast between LPDDR-PIM is not possible

    def __init__(self, *args, **kwargs):
        self.name = "CXL_PIM_nBC"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = (
            19.2 / 32
        )  # 8-lane PCIe GEN5, but only one LPDDR5 at a time
        self.host_to_device_pj_per_bit = (
            50 * 32
        )  # crossing PCIe and LPDDR interfaces on both host and device
        self.device_to_host_bw_GBs = (
            19.2  # 8-lane PCIe GEN5, but only one LPDDR5 at a time
        )
        self.device_to_host_pj_per_bit = (
            50  # crossing PCIe and LPDDR interfaces on both host and device
        )

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 2 * 4
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 16 * 2 * 4
        self.pj_per_tflop = 0.4 * 1e12

        self.softmax_ns_per_element = 0.4 / (16 * 2 * 4)
        self.SiLU_ns_per_element = 0.6 / (16 * 2 * 4)
        self.RMSNorm_ns_per_element = 1.04 / (16 * 2 * 4)


class PIM_AI_24dimm(Base_architecture):

    def __init__(self, *args, **kwargs):
        self.name = "PIM-AI-24dimm"
        super().__init__(*args, **kwargs)

        # HOST communication
        self.host_to_device_bw_GBs = 22
        self.host_to_device_pj_per_bit = 1200
        self.device_to_host_bw_GBs = 44 * 12
        self.device_to_host_pj_per_bit = 50

        # Device memory (shared memory like)
        self.mem_bw_GBs = 102.4 * 16 * 24
        self.mem_pj_per_bit = 0.95

        # Compute
        self.tflops = 5 * 16 * 24
        self.pj_per_tflop = 0.4 * 1e12
