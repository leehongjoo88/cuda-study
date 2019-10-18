#include <iostream>

__global__ void twice(float *array, size_t pitch, size_t row, size_t col) {
  int i = threadIdx.x;
  int j = threadIdx.y;
  if (!(i < row && j < col)) {
    return;
  }
  float *array_row =
      reinterpret_cast<float *>(reinterpret_cast<char *>(array) + i * pitch);
  array_row[j] = array_row[j] * 2.0;
}

int main() {
  const int row = 16;
  const int col = 16;
  float *array = new float[row * col];
  // float array[16][16];

  for (auto i = 0; i < row; ++i) {
    for (auto j = 0; j < col; ++j) {
      array[i * row + j] = 1;
    }
  }

  size_t pitch = 0;
  float *gpu_array = nullptr;
  cudaMallocPitch(&gpu_array, &pitch, col * sizeof(float), row);
  std::cout << "pitch: " << pitch << std::endl;

  cudaMemcpy2D(gpu_array, pitch, array, col * sizeof(float),
               col * sizeof(float), row, cudaMemcpyHostToDevice);

  dim3 block_dim(16, 16);
  twice<<<1, block_dim>>>(gpu_array, pitch, row, col);

  for (auto i = 0; i < row; ++i) {
    for (auto j = 0; j < col; ++j) {
      array[i * row + j] = 0.0;
    }
  }
  float* host_array = array;
  float* device_array = gpu_array;
  // for (auto i = 0; i < row; host_array += col, device_array = reinterpret_cast<float *>(reinterpret_cast<char *>(device_array) + pitch), ++i) {
  //   auto err = cudaMemcpy(host_array, device_array, sizeof(float) * col, cudaMemcpyDeviceToHost);
  //   std::cout << cudaGetErrorString(err) << std::endl;
  // }
  cudaMemcpy2D(array, col * sizeof(float), gpu_array, pitch,
               col * sizeof(float), row, cudaMemcpyDeviceToHost);

  bool ok = true;
  for (auto i = 0; i < row; ++i) {
    for (auto j = 0; j < col; ++j) {
      std::cout << "array[" << i << "][" << j << "] = " << array[i * row + j] << std::endl;
      if (!(array[i * row + j] > 2.0f - 0.001f &&
            array[i * row + j] < 2.0f + 0.001f)) {
        std::cout << "array[" << i << "][" << j << "] = " << array[i * row + j] << std::endl;
        ok = false;
        break;
      }
    }
    if (!ok)
      break;
  }

  if (ok) {
    std::cout << "ok" << std::endl;
  } else {
    std::cout << "wrong" << std::endl;
  }
  cudaFree(gpu_array);
  delete[] array;

  return 0;
}