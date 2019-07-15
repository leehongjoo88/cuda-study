#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

constexpr auto block_size = 16;

bool almost_equal(float a, float b) {
  return a - b < 0.00001f || b - a < 0.00001f;
}

class Matrix;

void swap(Matrix &left, Matrix &right) noexcept;

struct Matrix {
  size_t row;
  size_t col;
  std::unique_ptr<float[]> elements;
  float *gpu_elements;
  float get(int row, int col) const {
    return *(elements.get() + row * this->col + col);
  }
  void set(int row, int col, float value) {
    *(elements.get() + row * this->col + col) = value;
  }
  ~Matrix() {
    if (gpu_elements)
      cudaFree(gpu_elements);
  }

  Matrix(size_t row, size_t col) : row(row), col(col) {
    elements = std::unique_ptr<float[]>(new float[row * col]);
  }

  Matrix(const Matrix &other)
      : row(other.row), col(other.col),
        elements(new float[other.row * other.col]), gpu_elements(nullptr) {
    std::memcpy(elements.get(), other.elements.get(),
                row * col * sizeof(float));
  }

  Matrix &operator=(const Matrix &other) {
    using std::swap;
    if (this == &other) {
      return *this;
    }
    Matrix mat(other);
    swap(*this, mat);
    return *this;
  }

  void swap(Matrix &other) noexcept {
    using std::swap;
    swap(row, other.row);
    swap(col, other.col);
    swap(elements, other.elements);
  }
};

void swap(Matrix &left, Matrix &right) noexcept { left.swap(right); }

void print_matrix(const Matrix &mat) {
  for (int i = 0; i < mat.row; ++i) {
    for (int j = 0; j < mat.col; ++j) {
      std::cout << mat.get(i, j) << ", ";
    }
    std::cout << std::endl;
  }
}

struct GPUMatrix {
  size_t row;
  size_t col;
  size_t pitch;
  float *elements;
  __device__ float get(int row_idx, int col_idx) const {
    float *row = reinterpret_cast<float *>(reinterpret_cast<char *>(elements) +
                                           row_idx * pitch);
    return row[col_idx];
  }
  __device__ void set(int row_idx, int col_idx, float value) {
    float *row = reinterpret_cast<float *>(reinterpret_cast<char *>(elements) +
                                           row_idx * pitch);
    row[col_idx] = value;
  }
};

GPUMatrix load_to_gpu(Matrix &matrix) {
  size_t pitch;
  cudaMallocPitch(&matrix.gpu_elements, &pitch, matrix.col * sizeof(float),
                  matrix.row);
  cudaMemcpy2D(matrix.gpu_elements, pitch, matrix.elements.get(),
               matrix.col * sizeof(float), matrix.col * sizeof(float),
               matrix.row, cudaMemcpyHostToDevice);
  return GPUMatrix{matrix.row, matrix.col, pitch, matrix.gpu_elements};
}

Matrix copy_to_host(GPUMatrix matrix) {
  Matrix new_matrix(matrix.row, matrix.col);
  cudaError_t err =
      cudaMemcpy2D(new_matrix.elements.get(), matrix.col * sizeof(float),
                   matrix.elements, matrix.pitch, matrix.col * sizeof(float),
                   matrix.row, cudaMemcpyDeviceToHost);
  std::cout << "cudaMemcpy2D done: " << cudaGetErrorString(err) << std::endl;
  return new_matrix;
}

__global__ void matrix_mul_strait(GPUMatrix mat0, GPUMatrix mat1,
                                  GPUMatrix result) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;
  // target index: [row][col]

  float element = 0;
  for (int i = 0; i < mat0.col; ++i) {
    element += mat0.get(row, i) * mat1.get(i, col);
  }
  // printf("(%d, %d): %d\n", row, col, int(element));
  result.set(row, col, element);
}

__global__ void matrix_mul_shared(GPUMatrix mat0, GPUMatrix mat1,
                                  GPUMatrix result) {
  const int row_min = blockDim.x * blockIdx.x;

  const int col_min = blockDim.y * blockIdx.y;

  const int target_row = blockDim.x * blockIdx.x + threadIdx.x;
  const int target_col = blockDim.y * blockIdx.y + threadIdx.y;
  // assign matrix values in range
  // row: [row_min, row_max), col: [col_min, col_max)

  // iterate mat0.col / blockDim.x times.
  float c_value = 0.0f;
  for (int iter = 0, base = 0; iter < int(mat0.col / blockDim.y); ++iter, base += block_size) {
    __shared__ float mat0_submatrix[block_size][block_size];
    __shared__ float mat1_submatrix[block_size][block_size];

    mat0_submatrix[threadIdx.x][threadIdx.y] = mat0.get(
        row_min + threadIdx.x, base + threadIdx.y);
    mat1_submatrix[threadIdx.x][threadIdx.y] = mat1.get(
        base + threadIdx.x, col_min + threadIdx.y);

    __syncthreads();

    for (int i = 0; i < block_size; ++i) {
      c_value +=
          mat0_submatrix[threadIdx.x][i] * mat1_submatrix[i][threadIdx.y];
    }

    __syncthreads();
  }

  result.set(target_row, target_col, c_value);
}

int main() {
  Matrix host_matrix_A(64, 128);
  for (int i = 0; i < host_matrix_A.row; ++i) {
    for (int j = 0; j < host_matrix_A.col; ++j) {
      host_matrix_A.set(i, j, 1.0);
    }
  }
  Matrix host_matrix_B(128, 64);
  for (int i = 0; i < host_matrix_B.row; ++i) {
    for (int j = 0; j < host_matrix_B.col; ++j) {
      host_matrix_B.set(i, j, 1.0);
    }
  }

  Matrix host_matrix_C(64, 64);
  for (int i = 0; i < host_matrix_C.row; ++i) {
    for (int j = 0; j < host_matrix_C.col; ++j) {
      host_matrix_C.set(i, j, 0.0);
    }
  }

  GPUMatrix gpu_matrix_A = load_to_gpu(host_matrix_A);
  GPUMatrix gpu_matrix_B = load_to_gpu(host_matrix_B);
  GPUMatrix gpu_matrix_C = load_to_gpu(host_matrix_C);

  dim3 block_dim(16, 16);
  dim3 grid_dim(64 / 16, 64 / 16);
  matrix_mul_shared<<<grid_dim, block_dim>>>(gpu_matrix_A, gpu_matrix_B,
                                             gpu_matrix_C);

  host_matrix_C = copy_to_host(gpu_matrix_C);

  print_matrix(host_matrix_C);

  return 0;
}