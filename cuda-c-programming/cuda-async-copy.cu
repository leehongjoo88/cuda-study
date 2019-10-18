#include <iostream>
#include <chrono>

constexpr size_t kSize = 1000000;

class Stopwatch {
public:
  using TimePoint = decltype(std::chrono::high_resolution_clock::now());
  Stopwatch(): start(std::chrono::high_resolution_clock::now()) {}
  ~Stopwatch() {
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us\n";
  }
private:
  TimePoint start;
  TimePoint end;
};


void Transfer0(float* orig, float* target0, float* target1, float* target2) {
  cudaMemcpy(target0, orig, kSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(target1, orig, kSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(target2, orig, kSize * sizeof(float), cudaMemcpyHostToDevice);
};

void Transfer1(float* orig, float* target0, float* target1, float* target2) {
  cudaStream_t stream[3];
  float* targets[3] = {target0, target1, target2};
  for (int i = 0; i < 3; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  for (int i = 0; i < 3; ++i) {
    cudaMemcpyAsync(targets[i], orig, kSize * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
  }

  for (int i = 0; i < 3; ++i) {
    cudaStreamDestroy(stream[i]);
  }
};

int main() {
  float* original = new float[kSize];
  float* target0, *target1, *target2;

  cudaMalloc(&target0, sizeof(float) * kSize);
  cudaMalloc(&target1, sizeof(float) * kSize);
  cudaMalloc(&target2, sizeof(float) * kSize);

  {
    Stopwatch s;
    Transfer0(original, target0, target1, target2);
  }

  {
    Stopwatch s;
    Transfer1(original, target0, target1, target2);
  }

  cudaFree(target0);
  cudaFree(target1);
  cudaFree(target2);
  
  return 0;
}