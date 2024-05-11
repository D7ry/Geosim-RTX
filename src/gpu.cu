#include "gpu.h"
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include <iostream>
#include <stdio.h>

#include "Camera.h"
#include "Image.h"
#include "Scene.h"
#include "Settings.h"
#include "util/CUDAMath.h"

namespace CudaPlayground
{
__global__ void cudaHello() {
    printf("Hello World from CUDA thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    glm::vec3 a(1.0f, 2.0f, 3.0f);
    glm::vec3 b(4.0f, threadIdx.x, blockIdx.x);
    glm::vec3 c = a + b;
    printf("c = [%f, %f, %f]\n", c.x, c.y, c.z);
}

void play() {
    std::cout << "Running CUDA Playground" << std::endl;
    int numBlocks = 16;
    int threadsPerBlock = 16;
    cudaHello<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    std::cout << "CUDA Playground finished" << std::endl;
}
} // namespace CudaPlayground

namespace RendererCUDA
{

void check_device() {
    printf("Checking CUDA device...\n");
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf(
            "cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<int>(error_id),
            cudaGetErrorString(error_id)
        );
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
}

__device__ glm::vec3 trace_ray(
    glm::vec3 origin,
    glm::vec3 direction,
    const Scene* scene
) {
    return glm::vec3{1.f, 0.f, 0.f};
}

// Render a single pixel
__global__ void render_pixel(
    const Scene* scene,
    const Camera* camera,
    int width,
    int height,
    glm::vec3* frameBuffer_device
) {

    // TODO: these should be passed in as parameters
    float aspectRatio = width / height; // w : h
    // todo figure out why FOV seems "off"
    float fovComponent{tanf(camera->FOV / 2.f)};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    glm::vec3 final_color{0.f};

    const glm::vec2 ndc{(x + 0.5f) / width, (y + 0.5f) / height};

    glm::uvec2 pixelCoord{ndc.x * width, ndc.y * height};

    for (int i = 0; i < RAYS_PER_PIXEL; i++) {
        float2 rayOffset = CUDAMath::randomVec2(
            i + (x * width) + y
        ); // TODO: not sure if rng works

        const glm::vec2 ndcAliased{
            (x + rayOffset.x) / width, (y + rayOffset.y) / height
        };

        // screen space
        glm::vec2 coord = glm::vec2{
            ((2.f * ndc.x) - 1.f) * fovComponent * aspectRatio,
            1.f - (2.f * ndc.y) * fovComponent // flip vertically so +y is up
        };

        // ray coords in world space
        glm::vec4 start{camera->position, 1.f};
        glm::vec4 dir{coord.x, coord.y, -1.f, 0};

        // transform ray to view space
        dir = glm::normalize(dir);
        dir = dir * camera->viewMat;

        glm::vec3 color = trace_ray(start, dir, scene);

        final_color += color;
    }

    final_color /= RAYS_PER_PIXEL;

    { // writeback to framebuffer
        int frameBufferIndex = pixelCoord.x + (pixelCoord.y * width);
        frameBuffer_device[frameBufferIndex] = final_color;
    }
}

__host__ void render(const Scene* scene, const Camera* camera, Image* image) {
    float aspectRatio = (float)image->width / image->height; // w : h

    // todo figure out why FOV seems "off"
    float fovComponent{tanf(camera->FOV / 2.f)};

    int width = image->width;
    int height = image->height;

    dim3 blockDims = dim3(16, 16); // 256 threads per block
    dim3 gridDims = dim3(
        (width + blockDims.x - 1) / blockDims.x,
        (height + blockDims.y - 1) / blockDims.y
    );

    // allocate FB
    glm::vec3* frameBuffer = image->pixels.data();
    glm::vec3* frameBuffer_Device;
    cudaMalloc(&frameBuffer_Device, width * height * sizeof(glm::vec3));

    render_pixel<<<gridDims, blockDims>>>(
        scene, camera, width, height, frameBuffer_Device
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        frameBuffer,
        frameBuffer_Device,
        width * height * sizeof(glm::vec3),
        cudaMemcpyDeviceToHost
    );
    cudaFree(frameBuffer_Device);

}

} // namespace RendererCUDA
