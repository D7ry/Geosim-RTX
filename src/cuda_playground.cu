#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include "cuda_playground.h"


namespace CudaPlayground {
    __global__ void cudaHello() {
        printf("Hello World from CUDA thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    }

    void run() {
        std::cout << "Running CUDA Playground" << std::endl;
        int numBlocks = 16;
        int threadsPerBlock = 16;
        cudaHello<<<numBlocks, threadsPerBlock>>>();
        cudaDeviceSynchronize();

        std::cout << "CUDA Playground finished" << std::endl;
    }
}
