#pragma once
#include <glm/glm.hpp>
#include "util/CUDAMath.h"

#define PHI ((1.0f + sqrtf(5)) / 2.0f)

inline float __host__ __device__ mmod(float x, float y) {
    return x - y * floor(x / y);
}

inline glm::vec3 __host__ __device__ mmod(glm::vec3 x, float y) {
    return glm::vec3(
        x.x - y * floor(x.x / y),
        x.y - y * floor(x.y / y),
        x.z - y * floor(x.z / y)
    );
}

inline glm::vec3 __host__ __device__ mmod(glm::vec3 x, glm::vec3 y) {
    return glm::vec3(
        x.x - y.x * floor(x.x / y.x),
        x.y - y.y * floor(x.y / y.y),
        x.z - y.z * floor(x.z / y.z)
    );
}

// Many of the SDFs here are based on Inigo Quilez' fantastic work.
// http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

inline float __host__ __device__ sdfUnion(float a, float b) {
    return min(a, b);
}

inline float __host__ __device__ sdfDifference(float a, float b) {
    return max(a, -b);
}

inline float __host__ __device__ sdfIntersection(float a, float b) {
    return max(a, b);
}

inline float __host__ __device__ sdfSphere(glm::vec3 pos, float radius) {
    return glm::length(pos) - radius;
}

inline float __host__ __device__ sdfPlane(glm::vec3 pos, glm::vec3 n) {
    return dot(pos, n);
}

inline float __host__ __device__ sdfBox(glm::vec3 pos, glm::vec3 dim) {
    glm::vec3 d = abs(pos) - dim;
    return min(max(d.x, max(d.y, d.z)), 0.0f)
           + length(glm::max(d, glm::vec3(0.0f)));
}

inline float __host__ __device__ mengerCross(glm::vec3 pos) {
    float a = sdfBox(
        glm::vec3(pos.x, pos.y, pos.z), glm::vec3(100.0f, 1.025f, 1.025f)
    );
    float b = sdfBox(
        glm::vec3(pos.y, pos.z, pos.x), glm::vec3(1.025f, 100.0f, 1.025f)
    );
    float c = sdfBox(
        glm::vec3(pos.z, pos.x, pos.y), glm::vec3(1.025f, 1.025f, 100.0f)
    );
    return sdfUnion(sdfUnion(a, b), c);
}

inline float __host__ __device__
mengerBox(glm::vec3 pos, int iterations, float time = 1.0f) {
    glm::vec3 p = pos;
    // http://iquilezles.org/www/articles/menger/menger.htm
    float main = sdfBox(p, glm::vec3(1.0f));
    float scale = 1.0f;

    for (int i = 0; i < iterations; i++) {
        glm::vec3 a = mmod(p * scale, 2.0f) - 1.0f;
        scale *= 3.0f;
        glm::vec3 r = 1.0f - 3.0f * abs(a);
        float c = mengerCross(r) / scale;
        main = sdfIntersection(main, c);
    }
    return main;
}

inline glm::vec3 __host__ __device__
rotate(glm::vec3 pos, glm::vec3 axis, float angle) {
    // https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    glm::vec3 c1 = glm::vec3(
        cos(angle) + axis.x * axis.x * (1 - cos(angle)),
        axis.x * axis.y * (1 - cos(angle)) + axis.z * sin(angle),
        axis.z * axis.x * (1 - cos(angle)) - axis.y * sin(angle)
    );
    glm::vec3 c2 = glm::vec3(
        axis.x * axis.y * (1 - cos(angle)) - axis.z * sin(angle),
        cos(angle) + axis.y * axis.y * (1 - cos(angle)),
        axis.z * axis.y * (1 - cos(angle)) + axis.x * sin(angle)
    );
    glm::vec3 c3 = glm::vec3(
        axis.x * axis.z * (1 - cos(angle)) + axis.y * sin(angle),
        axis.y * axis.z * (1 - cos(angle)) - axis.x * sin(angle),
        cos(angle) + axis.z * axis.z * (1 - cos(angle))
    );

    glm::vec3 p = glm::vec3(
        c1.x * pos.x + c2.x * pos.y + c3.x * pos.z,
        c1.y * pos.x + c2.y * pos.y + c3.y * pos.z,
        c1.z * pos.x + c2.z * pos.y + c3.z * pos.z
    );

    return p;
}

inline float __host__ __device__ cornellBoxScene(const glm::vec3& pos) {
    float rightplane
        = sdfBox(pos - glm::vec3(-2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
    float leftplane
        = sdfBox(pos - glm::vec3(2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
    float backplane
        = sdfBox(pos - glm::vec3(0.0f, 0.0, 1.0), glm::vec3(3.0f, 2.f, 0.05f));
    float topplane = sdfBox(
        pos - glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(2.05f, 0.05f, 1.0f)
    );
    float plane = sdfPlane(pos - glm::vec3(0, -1.5f, 0), glm::vec3(0, 1, 0));

    float smallSphere = sdfSphere(pos - glm::vec3(0.7f, -1.0f, 0.6f), 0.5f);
    float menger
        = mengerBox((pos - glm::vec3(-0.7f, -1.0f, 0.2f)) / 0.5f, 5) * 0.5f;
    float objs = sdfUnion(smallSphere, menger);

    return sdfUnion(
        sdfUnion(
            sdfUnion(
                sdfUnion(sdfUnion(rightplane, plane), leftplane), backplane
            ),
            topplane
        ),
        objs
    );
}

inline glm::vec3 __host__ __device__ cornellBoxColor(const glm::vec3& pos) {
    float rightplane
        = sdfBox(pos - glm::vec3(-2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
    float leftplane
        = sdfBox(pos - glm::vec3(2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
    float backplane
        = sdfBox(pos - glm::vec3(0.0f, 0.0, 1.0), glm::vec3(3.0f, 2.f, 0.05f));
    float topplane = sdfBox(
        pos - glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(2.05f, 0.05f, 1.0f)
    );
    float plane
        = sdfPlane(pos - glm::vec3(0, -1.5f, 0), glm::vec3(0.0f, 1.0f, 0.0f));

    float smallSphere = sdfSphere(pos - glm::vec3(0.7f, -1.0f, 0.6f), 0.5f);
    float bigSphere = sdfSphere(pos - glm::vec3(-0.7f, -0.9f, 0.2f), 0.6f);
    float spheres = sdfUnion(bigSphere, smallSphere);

    float whitewalls = sdfUnion(sdfUnion(topplane, plane), spheres);

    if (leftplane < rightplane && leftplane < whitewalls
        && leftplane < backplane) {
        return glm::vec3(0.05f, .5f, 0.8f);
    } else if (backplane < rightplane && backplane < whitewalls) {
        return glm::vec3(1.0f, 0.8f, 0.1f);
    } else if (rightplane < whitewalls) {
        return glm::vec3(.9f, 0.2f, 0.4f);
    }

    return glm::vec3(0.85f);
}

