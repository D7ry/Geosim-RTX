#pragma once
#include "gpu.h"

namespace CUDAScenes
{

inline void solar_system(CUDAStruct::Scene* scene) {

    using Geometry = CUDAStruct::Geometry;
    using Sphere = CUDAStruct::SpherePrimitive;
    scene->cubemap
        = CUDAStruct::load_texture_device("../resource/starmap_g8k.jpg");
    Geometry solar_system;
    solar_system.position = {0, 0, -1.5};

    const int sun_idx = 0;
    const int mercury_idx = 1;
    const int venus_idx = 2;
    const int earth_idx = 3;
    const int mars_idx = 4;
    const int jupiter_idx = 5;
    const int saturn_idx = 6;
    const int uranus_idx = 7;
    const int neptune_idx = 8;
    Sphere sun;
    sun.position = {0, -1, 0};
    sun.radius = 0.5;
    sun.mat_albedo = {1, 1, 0};
    sun.mat_roughness = 0.5;
    sun.mat_emissionColor = {1, 1, 0};
    sun.mat_emissionStrength = 0.3;
    sun.mat_is_emissive = true;

    sun.texture_device
        = CUDAStruct::load_texture_device("../resource/nasa_sun.png");
    // Mercury
    Sphere mercury;
    mercury.position = {0, 0, 0};
    mercury.radius = 0.05;
    mercury.mat_albedo = {0.8, 0.8, 0.8};
    mercury.mat_roughness = 0.3;
    mercury.mat_emissionColor = {0.8, 0.8, 0.8};
    mercury.mat_emissionStrength = 0.1;

    // Venus
    Sphere venus;
    venus.position = {0, 0, 0};
    venus.radius = 0.075;
    venus.mat_albedo = {0.9, 0.7, 0.2};
    venus.mat_roughness = 0.4;
    venus.mat_emissionColor = {0.9, 0.7, 0.2};
    venus.mat_emissionStrength = 0.2;
    // Earth
    Sphere earth;
    earth.position = {0, 0, 0};
    earth.radius = 0.1;
    earth.mat_albedo = {0, 0.5, 1};
    earth.mat_roughness = 0.2;
    earth.mat_emissionColor = {0, 0.5, 1};
    earth.mat_emissionStrength = 0.3;

    // Mars
    Sphere mars;
    mars.position = {0, 0, 0};
    mars.radius = 0.08;
    mars.mat_albedo = {1, 0, 0};
    mars.mat_roughness = 0.3;
    mars.mat_emissionColor = {1, 0, 0};
    mars.mat_emissionStrength = 0.1;

    // Jupiter
    Sphere jupiter;
    jupiter.position = {0, 0, 0};
    jupiter.radius = 0.3;
    jupiter.mat_albedo = {0.8, 0.6, 0.4};
    jupiter.mat_roughness = 0.5;
    jupiter.mat_emissionColor = {0.8, 0.6, 0.4};
    jupiter.mat_emissionStrength = 0.4;

    // Saturn
    Sphere saturn;
    saturn.position = {0, 0, 0};
    saturn.radius = 0.25;
    saturn.mat_albedo = {0.9, 0.8, 0.6};
    saturn.mat_roughness = 0.6;
    saturn.mat_emissionColor = {0.9, 0.8, 0.6};
    saturn.mat_emissionStrength = 0.3;

    // Uranus
    Sphere uranus;
    uranus.position = {0, 0, 0};
    uranus.radius = 0.15;
    uranus.mat_albedo = {0.6, 0.8, 1};
    uranus.mat_roughness = 0.4;
    uranus.mat_emissionColor = {0.6, 0.8, 1};
    uranus.mat_emissionStrength = 0.2;

    // Neptune
    Sphere neptune;
    neptune.position = {0, 0, 0};
    neptune.radius = 0.15;
    neptune.mat_albedo = {0.2, 0.4, 1};
    neptune.mat_roughness = 0.3;
    neptune.mat_emissionColor = {0.2, 0.4, 1};
    neptune.mat_emissionStrength = 0.2;

    sun.texture_device
        = CUDAStruct::load_texture_device("../resource/nasa_sun.png");

    solar_system.add(sun);
    solar_system.add(mercury);
    solar_system.add(venus);
    solar_system.add(earth);
    solar_system.add(mars);
    solar_system.add(jupiter);
    solar_system.add(saturn);
    solar_system.add(uranus);
    solar_system.add(neptune);

    scene->add(solar_system);

    void* mercury_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[mercury_idx]);
    void* venus_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[venus_idx]);
    void* earth_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[earth_idx]);
    void* mars_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[mars_idx]);
    void* jupiter_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[jupiter_idx]);
    void* saturn_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[saturn_idx]);
    void* uranus_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[uranus_idx]);
    void* neptune_ptr
        = reinterpret_cast<void*>(&scene->geometries[0].spheres[neptune_idx]);

    Rotor& rotor = scene->rotor;

    // Adjusted radius and spacing for realistic representation
    rotor.add(
        mercury_ptr, {0, -1, 0}, 0.24, 586.5
    ); // Mercury: Radius - 0.24, Period - 586.5 Earth days
    rotor.add(
        venus_ptr, {0, -1, 0}, 0.6, 2430
    ); // Venus: Radius - 0.6, Period - 2430 Earth days
    rotor.add(
        earth_ptr, {0, -1, 0}, 0.63, 10
    ); // Earth: Radius - 0.63, Period - 10 Earth days
    rotor.add(
        mars_ptr, {0, -1, 0}, 0.34, 10.3
    ); // Mars: Radius - 0.34, Period - 10.3 Earth days
    rotor.add(
        jupiter_ptr, {0, -1, 0}, 1.79, 4.1
    ); // Jupiter: Radius - 1.79, Period - 4.1 Earth days
    rotor.add(
        saturn_ptr, {0, -1, 0}, 1.5, 4.3
    ); // Saturn: Radius - 1.5, Period - 4.3 Earth days
    rotor.add(
        uranus_ptr, {0, -1, 0}, 1.25, 7.2
    ); // Uranus: Radius - 1.25, Period - 7.2 Earth days
    rotor.add(
        neptune_ptr, {0, -1, 0}, 1.24, 6.7
    ); // Neptune: Radius - 1.24, Period - 6.7 Earth days
       //
}
} // namespace CUDAScenes
