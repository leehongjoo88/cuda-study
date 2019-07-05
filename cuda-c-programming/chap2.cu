#include <iostream>


void print_vector(float* vector, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    std::cout << i << ", ";
  }

  std::cout << std::endl;
}

void print_matrix(float matrix[256][256]) {
  for (size_t r = 0; r < 256; ++r) {
    for (size_t c = 0; c < 256; ++c) {
      std::cout << matrix[r][c] << ", ";
    }
    std::cout << '\n';
  }

  std::cout << '\n';
}

__global__  void vector_add(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

__global__ void vector_assign(float *A, float *B, float *C) {
  int i = threadIdx.x;
  A[i] = 1.0f;
  B[i] = 2.0f;
  C[i] = 0.0f;
}

__global__ void matrix_add(float* A, float* B, float* C, size_t pitch) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  float *a = reinterpret_cast<float *>(reinterpret_cast<char *>(A) + i * pitch);
  float *b = reinterpret_cast<float *>(reinterpret_cast<char *>(B) + i * pitch);
  float *c = reinterpret_cast<float *>(reinterpret_cast<char *>(C) + i * pitch);
  c[j] = b[j] + a[j];
}

void test_vector_add() {
  float host_c[256];

  float *A, *B, *C;

  cudaMalloc(&A, 256 * sizeof(float));
  cudaMalloc(&B, 256 * sizeof(float));
  cudaMalloc(&C, 256 * sizeof(float));
  vector_assign<<<1, 256>>>(A, B, C);
  vector_add<<<1, 256>>>(A, B, C);
  cudaMemcpy(host_c, C, sizeof(float) * 256, cudaMemcpyDeviceToHost);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  print_vector(host_c, 256);
}

void test_matrix_add() {
  float host_a[256][256];
  float host_b[256][256];
  float host_c[256][256];

  for (auto i = 0; i < 256; ++i) {
    for (auto j = 0; j < 256; ++j) {
      host_a[i][j] = 0.1;
      host_b[i][j] = 0.2;
      host_c[i][j] = 0.0f;
    }
  }

  float *device_a, *device_b, *device_c;
  size_t pitch = 0;
  cudaMallocPitch(&device_a, &pitch, 256 * sizeof(float), 256);
  cudaMallocPitch(&device_b, &pitch, 256 * sizeof(float), 256);
  cudaMallocPitch(&device_c, &pitch, 256 * sizeof(float), 256);

  cudaMemcpy2D(device_a, pitch, host_a, 256 * sizeof(float), 256 * sizeof(float), 256, cudaMemcpyHostToDevice);
  cudaMemcpy2D(device_b, pitch, host_b, 256 * sizeof(float), 256 * sizeof(float), 256, cudaMemcpyHostToDevice);
  cudaMemcpy2D(device_c, pitch, host_c, 256 * sizeof(float), 256 * sizeof(float), 256, cudaMemcpyHostToDevice);

  dim3 block_size(16, 16);
  dim3 grid_size(256 / 16, 256 / 16);
  matrix_add<<<grid_size, block_size>>>(device_a, device_b, device_c, pitch);

  cudaMemcpy2D(host_c, 256 * sizeof(float), device_c, pitch, 256 * sizeof(float), 256, cudaMemcpyDeviceToHost);

  print_matrix(host_c);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
}

int main () {
  test_vector_add();
  test_matrix_add();
  return 0;
}

