#include "Rotor.h"
#include "gpu.h"

void Rotor::add(void* p, glm::vec3 rotation_center, float rotation_radius, float rotation_period_sec){
    CUDAStruct::SpherePrimitive* sphere_primitive = reinterpret_cast<CUDAStruct::SpherePrimitive*>(p);

    Rotation r {sphere_primitive, rotation_center, rotation_radius, rotation_period_sec};

    _rotations.push_back(r);
}

void Rotor::tick(float delta_time) {

    for (auto& r : _rotations) {
        r.rotation_angle += delta_time / r.rotation_period_sec * 2 * M_PI;
        if (r.rotation_angle > 2 * M_PI) {
            r.rotation_angle -= 2 * M_PI;
        }

        CUDAStruct::SpherePrimitive* sphere_primitive = reinterpret_cast<CUDAStruct::SpherePrimitive*>(r.sphere_primitive);

        sphere_primitive->position = r.rotation_center + glm::vec3(cos(r.rotation_angle), 0, sin(r.rotation_angle)) * r.rotation_radius;
    }

}
