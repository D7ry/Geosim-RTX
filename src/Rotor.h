#pragma once
#include <vector>
#include <glm/glm.hpp>

/**
 * Animates all spheres being added to it.
 * Rotates them around their center that they originally belong to.
 */
class Rotor {

    public:
        void add(void* sphere_primitive, glm::vec3 rotation_center, float rotation_radius, float rotation_period_sec);
        void tick(float delta_time);

    private:
        struct Rotation{
            void* sphere_primitive;
            const glm::vec3 rotation_center;
            const float rotation_radius;
            const float rotation_period_sec;
            float rotation_angle{0.f};
        };
        std::vector<Rotation> _rotations;
};

class RainDrop {
    public:
        void add(void* sphere_primitive, float raindrop_ceiling, float raindrop_floor, glm::vec2 raindrop_range, float raindrop_period_sec);

};
