#include <iostream>
#include <memory>
#include <omp.h>

void sum_cpu(int n, int *x, int *y, int *z) {
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

__global__
void sum_gpu(int n, int* x, int *y, int *z) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    z[i] += x[i] + y[i];
  }
}


int main() {
  constexpr int N = 1 << 20;
  auto x = std::unique_ptr<int[]>(new int[N]);
  auto y = std::unique_ptr<int[]>(new int[N]);
  auto z = std::unique_ptr<int[]>(new int[N]);

  for (int i = 0; i < N; ++i) {
    x[i] = 1;
    y[i] = 2;
    z[i] = 0;
  }

  sum_cpu(N, x.get(), y.get(), z.get());

  std::cout << z[0] << ", " << z[1] << ", " << z[2] << ", " << z[N-2] << ", " << z[N - 1] << std::endl;

  for (int i = 0; i < N; ++i) {
    x[i] = 1;
    y[i] = 2;
    z[i] = 0;
  }

  int *g_x, *g_y, *g_z;
  cudaMalloc(&g_x, sizeof(int) * N);
  cudaMalloc(&g_y, sizeof(int) * N);
  cudaMalloc(&g_z, sizeof(int) * N);

  cudaMemcpy(g_x, x.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(g_y, y.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(g_z, z.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  sum_gpu<<<dim3(1024, 1), dim3(256, 1)>>>(N, g_x, g_y, g_z);
  cudaMemcpy(z.get(), g_z, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaFree(g_x);
  cudaFree(g_y);
  cudaFree(g_z);

  std::cout << z[0] << ", " << z[1] << ", " << z[2] << ", " << z[N-2] << ", " << z[N - 1] << std::endl;

  return 0;
}