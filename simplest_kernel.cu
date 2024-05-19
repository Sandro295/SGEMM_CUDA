#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void kernel(unsigned int *A, unsigned int *B, int row) {
  auto x = threadIdx.x / 4;
  auto y = threadIdx.x % 4;
  A[x * row + y] = x;
  B[x * row + y] = y;
}

int main(int argc, char **argv) {
  unsigned int *Xs, *Ys;
  unsigned int *Xs_d, *Ys_d;

  unsigned int SIZE = 4;

  Xs = (unsigned int *)malloc(SIZE * SIZE * sizeof(unsigned int));
  Ys = (unsigned int *)malloc(SIZE * SIZE * sizeof(unsigned int));

  cudaMalloc((void **)&Xs_d, SIZE * SIZE * sizeof(unsigned int));
  cudaMalloc((void **)&Ys_d, SIZE * SIZE * sizeof(unsigned int));

  dim3 grid_size(1, 1, 1);
  dim3 block_size(4 * 4);

  kernel<<<grid_size, block_size>>>(Xs_d, Ys_d, 4);

  cudaMemcpy(Xs, Xs_d, SIZE * SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ys, Ys_d, SIZE * SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (int row = 0; row < SIZE; ++row) {
    for (int col = 0; col < SIZE; ++col) {
      std::cout << "[" << Xs[row * SIZE + col] << "|" << Ys[row * SIZE + col]
                << "] ";
    }
    std::cout << "\n";
  }

  cudaFree(Xs_d);
  cudaFree(Ys_d);
  free(Xs);
  free(Ys);
}
