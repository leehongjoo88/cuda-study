cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

project(projecttest LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
find_package(OpenMP)

set(CUDA_SEPARABLE_COMPILATION ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp -lgomp")

add_executable(test test.cu)
if (OpenMp_CXX_FOUND)
  target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
x
