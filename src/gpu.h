#pragma once
#include <iostream>

#include <glm/glm.hpp>

namespace CudaPlayground
{
void play();
}

class Scene;
class Camera;
class Image;

//

#define MAX_PRIMITIVES 10

namespace CUDAStruct
{

struct Intersection
{
    glm::vec3 mat_albedo{1.f};
    float mat_roughness{1.f};

    glm::vec3 mat_emissionColor{1.f};
    float mat_emissionStrength{0.f};
    // const RayIntersection math;

    glm::vec3 incidentDir; // angle at which ray hit surface
    glm::vec3 outgoingDir; // angle at which ray left surface
    glm::vec3 position;
    glm::vec3 normal;

    enum class ReflectionType
    {
        Specular,
        Diffuse,
        Refract
    };
};



struct SpherePrimitive
{

    glm::vec3 mat_albedo{1.f};
    float mat_roughness{1.f};

    glm::vec3 mat_emissionColor{1.f};
    float mat_emissionStrength{0.f};

    glm::vec3 position{0.f}; // local space
    float radius{1.f};

    float radius_dynamic{1.f};
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
            throw std::runtime_error("Max primitives reached");
            return;
        }
        spheres[num_spheres++] = p;
    }
};

#define MAX_GEOMETRY 10

struct Scene
{
    CUDAStruct::Geometry geometries[MAX_GEOMETRY];
    size_t num_geometries{0};

    void add(const CUDAStruct::Geometry& object) {
        if (num_geometries >= MAX_GEOMETRY) {
            printf("Max geometries reached\n");
            throw std::runtime_error("Max geometries reached");
            return;
        }
        geometries[num_geometries++] = object;
    }

    void tick() {
        for (int i = 0; i < this->num_geometries; i++) {
            CUDAStruct::Geometry* geometry = geometries + i;
            for (int j = 0; j < geometry->num_spheres; j++) {
                CUDAStruct::SpherePrimitive* sphere = geometry->spheres + j;
                sphere->radius_dynamic = sphere->radius * (1 + timer);
            }
        }
        if (timer_increasing) {
            timer += 0.01;
            if (timer >= 0.5) {
                timer_increasing = false;
            }
        } else {
            timer -= 0.01;
            if (timer <= 0) {
                timer_increasing = true;
            }
        }
    }

    float timer = 0;
    bool timer_increasing = true;
};

} // namespace CUDAStruct
  //

namespace RendererCUDA
{
    void check_device();
    void render(const CUDAStruct::Scene* scene, const Camera* camera, Image* image);
} // namespace RendererCUDA
