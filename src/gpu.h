#pragma once
#include "util/CUDAMath.h"
#include <iostream>

#include <glm/glm.hpp>

namespace CudaPlayground
{
void play();
}

class Scene;
class Camera;
class Image;

namespace RendererCUDA
{
void check_device();
void render(const Scene* scene, const Camera* camera, Image* image);
} // namespace RendererCUDA
  //

#define MAX_PRIMITIVES 10

namespace CUDAStruct
{
struct SpherePrimitive
{
    glm::vec3 position{0.f}; // local space
    float radius{1.f};
};

struct Geometry
{
    SpherePrimitive spheres[MAX_PRIMITIVES];
    size_t num_spheres{0};

    glm::vec3 position{0.f}; // world space
    float scale{1.f};

    // glm::mat4 rotation;

    void add(const SpherePrimitive& p) {
        if (num_spheres >= MAX_PRIMITIVES) {
            printf("Max primitives reached\n");
            return;
        }
        spheres[num_spheres++] = p;
    }
};

#define MAX_GEOMETRY 10

struct Scene
{
    Geometry geometries[MAX_GEOMETRY];
    size_t num_geometries{0};

    void add(const Geometry& object) {
        if (num_geometries >= MAX_GEOMETRY) {
            printf("Max geometries reached\n");
            return;
        }
        geometries[num_geometries++] = object;
    }
};

inline __device__ double SpherePrimitive_SDF(
    const SpherePrimitive* sphere,
    const glm::vec4& p,
    const glm::vec4& positionWorldSpace
) {
    const glm::vec3 euclideanPosition{
        glm::vec3(positionWorldSpace) + sphere->position
    };

    const glm::vec4 hyperbolicPosition{CUDAMath::constructHyperboloidPoint(
        euclideanPosition, glm::length(euclideanPosition)
    )};

    const float dist = CUDAMath::hyperbolicSphereSDF(
        p, // todo: is w supposed to be 0?
        sphere->radius,
        hyperbolicPosition
    );

    return dist;
}
} // namespace CUDAStruct
