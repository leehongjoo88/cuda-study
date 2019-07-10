CUDA C PROGRAMMING
=====================================================

The summary of NVidia's [Cuda C Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

# Introduction

## Scalable Programming Model

Three key abstractions

  1. A hierarchy of thread groups
  2. Shared memories
  3. Barrier synchronization
  
> These abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task parallelism.

(? What is data parallelism, thread parallelism and task parallelism?)

# Programming Model

## Kernels

**kernel**: C functions that when called, are executed N times in parallel by N different *CUDA* threads.

A kernel is defined using the `__global__`.
The number of CUDA threads that execute that kernel is specified using `<<<...>>>`.
Each thread that executes the kernel is given a unique *thread ID* through the `threadIdx` variable.

``` cuda
__global__ void vector_add(float* A, float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
```

([above code is in chap2.cu](chap2.cu))

## Thread Hierarchy

`threadIdx`: 3 component vector

There's a limit to the number of threads per block. All threads of a block are expected to:

- reside on the same processor core.
- must share the limited memory resources of the core.
- a thread block may contain up to 1024 threads.

The number of thread blocks in a grid can greatly exceed.

``` cuda
__global__ void matrix_add(float A[N][N], float B[N][N], float C[N][N]) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) {
    C[i][j] = A[i][j] + B[i][j];
  }
}

```

``` c++
int main() {
  dim3 block_size(16, 16);
  dim3 grid_size(N / thread_size.x, N / thread_size.y);
  matrix_add<<<grid_size, block_size>>>(A, B, C);
  return 0;
}
```

(`blockIdx.x * blockDim.x + threadIdx.x` is CUDA boilerplate for accessing x index)

## Memory Hierarchy

- Each thread: local memory.
- Each thread block: shared memory visible to all threads of the block.
- All threads: global memory

  `+`

- The constant memory space (read-only)
- The texture memory space (read-only)

## Compute Capability

A version number - `SM version`, this version number identifies the features supported by the GPU.

# Programming Interface

## Compiling with NVCC

*PTX*: The CUDA instruaction set architecture.

Kernels can be written using *PTX* or *C*.

`nvcc`: A compiler driver that simplifies the process of compiling C/PTX code.

### Compilation Workflow

#### Offline Compilation

- Separate device code from host code
- Compile the device code into an assembly form (*PTX* code) and/or binary form (*cubin* object)
- Modify the host by replacing the `<<< ... >>>` syntax by the necessary
CUDA C runtime function calls.

#### JIT Compilation

Any *PTX* code loaded by an application at runtime is compiled further to binary code.

### Binary Compatibility

Binary code is architecture specific. The architecture can be specified with `-code`.

## CUDA C Runtime

The runtime is implemented in `cudart` library.

### Initialization

It initializes the first time a runtime function is called.

During initialization the runtime creates a CUDA context for each device.

### Device Memory

Device memory can be allocated either as linear memory or as *CUDA arrays*.

- *CUDA arrays* are opaque memory layouts optimized for texture fetching.
- Linear memory exists on the device in a 40-bit address space.

#### 1D memory

- allocate with `cudaMalloc`

#### 2D memory

- allocate with `cudaMallocPitch`
- e.g.) allocate [height][width] and access in device memory

``` cuda
float* device_ptr;
size_t pitch;
cudaMallocPitch(&device_ptr, &pitch, width * sizeof(float), height);

// ...

int h_idx = blockDim.x * blockIdx.x + threadIdx.x;
int w_idx = blockDim.y * blockIdx.y + threadIdx.y;

float* row = reinterpret_cast<float *>(reinterpret_cast<char *>(ptr) + h_idx * pitch);
row[w_idx];
```

#### 3D memory

- allocate with `cudaMalloc3D`
- cudaPitchedPtr .xsize: width (in bytes) .ysize: height
- e.g.) allocate [depth][height][width]

``` cuda
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr ptr;
cudaMalloc3D(&ptr, extent);

// ...

int d_idx = blockDim.x * blockIdx.x + threadIdx.x;
int h_idx = blockDim.y * blockIdx.y + threadIdx.y;
int w_idx = blockDim.z * blockIdx.z + threadIdx.z;

char* row_c = reinterpret_cast<char *>(ptr.ptr) + d_idx * h_idx * ptr.pitch + h_idx * ptr.pitch;
float* row = reinterpret_cast<float *>(row_c);
row[w_idx];
```

([Sample code in chap3.cu](chap3.cu))
