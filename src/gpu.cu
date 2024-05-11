#include "gpu.h"
#include <iostream>
#include <stdio.h>
#include <glm/glm.hpp>


#include "Camera.h"
#include "Image.h"
#include "Scene.h"

namespace CudaPlayground {
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

namespace RendererCUDA {

__global__ void _render(const Scene* scene, const Camera* camera, Image* image) {
    float aspectRatio = (float)image->width / image->height; // w : h

    // todo figure out why FOV seems "off"
    float fovComponent{tanf(camera->FOV / 2.f)};

    // for (int y = 0; y < image.height; ++y) {
    //     for (int x = 0; x < image.width; ++x) {
    //         const int index = x + (y * image.width);
    //
    //         // misc debug stuff
    //         const glm::uvec2 debugRay{image.width / 2, image.height / 2};
    //         isDebugRay = (x == debugRay.x && y == debugRay.y);
    //
    //         if constexpr (!INTERACTIVE_MODE) {
    //             const unsigned index{x + (y * image.width)};
    //             const unsigned numPixels{image.width * image.height};
    //
    //             const float completionPercent{100.f * index / numPixels};
    //
    //             // how many pixels per print
    //             constexpr unsigned printFreq{50};
    //
    //             static int prevPrintIndex{0};
    //
    //             if (index > prevPrintIndex + printFreq) {
    //                 prevPrintIndex = index;
    //                 std::cout << completionPercent << "%\n";
    //             }
    //         }
    //
    //         // ray tracing stuff
    //         const glm::vec2 ndc{(x + 0.5f) / image.width, (y + 0.5f) / image.height};
    //
    //         glm::vec3 color{0.f};
    //
    //         for (int i = 0; i < RAYS_PER_PIXEL; ++i) {
    //             glm::vec2 rayOffset = Math::randomVec2(rngSeed + i);
    //
    //             const glm::vec2 ndcAliased{(x + rayOffset.x) / image.width, (y + rayOffset.y) / image.height};
    //
    //             // screen space
    //             glm::vec2 coord;
    //
    //             if constexpr (ANTIALIAS) {
    //                 coord = glm::vec2{
    //                     ((2.f * ndcAliased.x) - 1.f) * fovComponent * aspectRatio,
    //                     1.f - (2.f * ndcAliased.y) * fovComponent // flip vertically so +y is up
    //                 };
    //             } else {
    //                 coord = glm::vec2{
    //                     ((2.f * ndc.x) - 1.f) * fovComponent * aspectRatio,
    //                     1.f - (2.f * ndc.y) * fovComponent // flip vertically so +y is up
    //                 };
    //             }
    //
    //             // ray coords in world space
    //             glm::vec4 start{camera.position, 1.f};
    //             glm::vec4 dir{coord.x, coord.y, -1.f, 0};
    //
    //             // transform ray to view space
    //             dir = glm::normalize(dir);
    //             dir = dir * camera.viewMat;
    //
    //             Ray ray{start, dir};
    //
    //             if (!accumulate)
    //                 resetAccumulator();
    //
    //             // isDebugRay = index == 3846;
    //
    //             color += traceRay(ray, scene);
    //
    //             frameBuffer[index] += color;
    //         }
    //
    //         glm::vec3 pixelColor{frameBuffer[index]};
    //
    //         // average color
    //         pixelColor /= RAYS_PER_PIXEL;
    //
    //         // normalize color
    //         pixelColor.r = std::clamp(pixelColor.r, 0.f, 1.f);
    //         pixelColor.g = std::clamp(pixelColor.g, 0.f, 1.f);
    //         pixelColor.b = std::clamp(pixelColor.b, 0.f, 1.f);
    //
    //         // debug visualization
    //         const bool shouldInvertColor{
    //             VISUALIZE_DEBUG_RAY && ((x == debugRay.x + 1 && y == debugRay.y + 0) || // left
    //                                     (x == debugRay.x - 1 && y == debugRay.y - 0) || // right
    //                                     (x == debugRay.x + 0 && y == debugRay.y + 1) || // top
    //                                     (x == debugRay.x - 0 && y == debugRay.y - 1))   // bottom
    //         };
    //
    //         if (shouldInvertColor)
    //             pixelColor = glm::vec3{1} - pixelColor;
    //
    //         // actually setting pixel
    //         image.setPixel(ndc, pixelColor);
    //     }
    // }

}

} // namespace RendererCUDA
