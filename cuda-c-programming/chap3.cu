#include <iostream>
#include <memory>

__global__ void tensor_1d_assign(float *tensor, size_t tensor_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < tensor_size) {
    tensor[i] = i;
  }
}

void tensor_1d_test() {
  float *device_ptr;
  cudaMalloc(&device_ptr, sizeof(float) * 1024 * 1024);
  const int block_dim = 256;
  const int grid_dim = 1024 * 1024 / 256;
  tensor_1d_assign<<<grid_dim, block_dim>>>(device_ptr, 1024 * 1024);
  auto host_ptr = std::unique_ptr<float[]>(new float[1024 * 1024]);
  cudaMemcpy(host_ptr.get(), device_ptr, sizeof(float) * 1024 * 1024,
             cudaMemcpyDeviceToHost);
  cudaFree(device_ptr);

  bool is_ok = true;
  for (auto i = 0; i < 1024 * 1024; ++i) {
    if (host_ptr[i] != i) {
      is_ok = false;
      std::cout << "host_ptr[" << i << "] = " << host_ptr[i] << std::endl;
      break;
    }
  }

  if (is_ok) {
    std::cout << "ok" << std::endl;
  } else {
    std::cout << "wrong" << std::endl;
  }
}

__global__ void tensor_2d_assign(float *tensor, int width, int height,
                                 size_t pitch) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (!(i < height && j < width)) {
    return;
  }

  float *tensor_row =
      reinterpret_cast<float *>(reinterpret_cast<char *>(tensor) + i * pitch);
  tensor_row[j] = i * width + j;
}

void tensor_2d_test() {
  const int width = 4096;
  const int height = 1024;
  size_t pitch = 0u;

  float *device_ptr;

  cudaMallocPitch(&device_ptr, &pitch, width * sizeof(float), height);

  dim3 block_dim(16, 16);
  dim3 grid_dim(1024 / 16, 4096 / 16);

  tensor_2d_assign<<<grid_dim, block_dim>>>(device_ptr, width, height, pitch);

  auto host_ptr = std::unique_ptr<float[]>(new float[width * height]);

  cudaMemcpy2D(host_ptr.get(), width * sizeof(float), device_ptr, pitch,
               width * sizeof(float), height, cudaMemcpyDeviceToHost);

  cudaFree(device_ptr);

  bool is_ok = true;
  auto value = 0;
  for (auto i = 0; i < height; ++i) {
    for (auto j = 0; j < width; ++j, ++value) {
      if (host_ptr[width * i + j] != value) {
        std::cout << host_ptr[width * i + j] << std::endl;
        break;
      }
    }
    if (!is_ok) {
      break;
    }
  }
  if (is_ok) {
    std::cout << "ok" << std::endl;
  } else {
    std::cout << "wrong" << std::endl;
  }
}

__global__ void tensor_3d_assign(cudaPitchedPtr pitched_ptr, int depth,
                                 int height, int width) {
  const size_t pitch = pitched_ptr.pitch;
  const size_t slice_pitch = pitch * height;

  int d_idx = blockDim.x * blockIdx.x + threadIdx.x;
  int h_idx = blockDim.y * blockIdx.y + threadIdx.y;
  int w_idx = blockDim.z * blockIdx.z + threadIdx.z;

  char *slice_ptr =
      reinterpret_cast<char *>(pitched_ptr.ptr) + d_idx * slice_pitch;
  int *row = reinterpret_cast<int *>(slice_ptr + h_idx * pitch);
  row[w_idx] = d_idx * height * width + h_idx * width + w_idx;
}

void tensor_3d_test() {
  const int depth = 32;
  const int height = 64;
  const int width = 128;

  cudaExtent extent = make_cudaExtent(width * sizeof(int), height, depth);
  cudaPitchedPtr pitched_ptr;
  cudaMalloc3D(&pitched_ptr, extent);

  dim3 block_dim(4, 4, 4);
  dim3 grid_dim(depth / 4, height / 4, width / 4);
  tensor_3d_assign<<<grid_dim, block_dim>>>(pitched_ptr, depth, height, width);

  int host_ptr[32][64][128];

  cudaMemcpy3DParms memcpy_params;
  memcpy_params.srcPtr = pitched_ptr;
  memcpy_params.dstPtr.ptr = host_ptr;
  memcpy_params.dstPtr.pitch = width * sizeof(int);
  memcpy_params.dstPtr.xsize = width;
  memcpy_params.dstPtr.ysize = height;
  memcpy_params.kind = cudaMemcpyDeviceToHost;
  memcpy_params.extent.depth = depth;
  memcpy_params.extent.height = height;
  memcpy_params.extent.width = width * sizeof(int);

  cudaMemcpy3D(&memcpy_params);
  cudaFree(pitched_ptr.ptr);

  bool is_ok = true;
  int value = 0;
  for (int d = 0; d < depth; ++d) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w, ++value) {
        if (host_ptr[d][h][w] != value) {
          std::cout << "wrong result. host_ptr[" << d << "][" << h << "][" << w
                    << "] = " << host_ptr[d][h][w] << ",, value = " << value
                    << std::endl;
          is_ok = false;
          break;
        }
      }
      if (!is_ok)
        break;
    }
    if (!is_ok)
      break;
  }
  if (is_ok) {
    std::cout << "ok" << std::endl;
  } else {
    std::cout << "wrong" << std::endl;
  }
}

int main() {
  // tensor_1d_test();
  // tensor_2d_test();
  tensor_3d_test();
  return 0;
}