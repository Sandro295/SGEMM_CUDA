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
| 10: Warptiling                      |       50.5 | 0.4%                             |
| 8: Avoid Bank Conflicts (Offset)    |       61.8 | 0.5%                             |
| 6: Vectorized Mem Access            |       61.9 | 0.5%                             |
| 7: Avoid Bank Conflicts (Linearize) |       61.9 | 0.5%                             |
| 5: 2D Blocktiling                   |       62.1 | 0.5%                             |
| 9: Autotuning                       |       64.6 | 0.5%                             |
| 2: GMEM Coalescing                  |      110.3 | 0.8%                             |
| 1: Naive                            |      111.7 | 0.8%                             |
| 4: 1D Blocktiling                   |      137.9 | 1.0%                             |
| 3: SMEM Caching                     |      141.3 | 1.1%                             |
| 0: cuBLAS                           |    13222.4 | 100.0%                           |
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
