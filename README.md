# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA 3070 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      172.8 | 1.3%                             |
| 2: GMEM Coalescing                  |     1226.4 | 9.5%                             |
| 3: SMEM Caching                     |     1701.4 | 13.2%                            |
| 4: 1D Blocktiling                   |     5071.8 | 39.2%                            |
| 9: Autotuning                       |     9575.8 | 74.0%                            |
| 7: Avoid Bank Conflicts (Linearize) |    10027.5 | 77.5%                            |
| 5: 2D Blocktiling                   |    10142.7 | 78.4%                            |
| 8: Avoid Bank Conflicts (Offset)    |    10323.8 | 79.8%                            |
| 6: Vectorized Mem Access            |    11558.7 | 89.3%                            |
| 10: Warptiling                      |    12550.6 | 97.0%                            |
| 0: cuBLAS                           |    12936.8 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
