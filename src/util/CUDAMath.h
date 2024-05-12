#pragma once

#include <glm/glm.hpp>

namespace CUDAMath
{

__device__ float hypDot(const glm::vec4& u, const glm::vec4& v) {
    return (u.x * v.x) + (u.y * v.y) + (u.z * v.z) - (u.w * v.w); // Lorentz Dot
}

__device__ float hypNorm(const glm::vec4& v) {
    return glm::sqrt(glm::abs(hypDot(v, v)));
}

__device__ glm::vec4 hypNormalize(const glm::vec4& u) { return u / hypNorm(u); }

__device__ glm::vec4 correctDirection(const glm::vec4& p, const glm::vec4& d) {
    // Construct an orthonormal basis for the tangent space at p
    float x = p.x;
    float y = p.y;
    float z = p.z;
    float w = p.w;

    glm::vec4 basisX = glm::vec4(w, 0, 0, x);
    glm::vec4 basisY = glm::vec4(0, w, 0, y);
    glm::vec4 basisZ = glm::vec4(0, 0, w, z);

    // Express the local direction in terms of the tangent space basis
    float dx = d.x;
    float dy = d.y;
    float dz = d.z;

    glm::vec4 rayDir = dx * basisX + dy * basisY + dz * basisZ;
    rayDir = hypNormalize(rayDir);

    return rayDir;
}

__device__ glm::vec4 constructHyperboloidPoint(
    const glm::vec3& direction,
    float distance
) {
    const float w{cosh(distance)};
    const float magSquared = w * w - 1;
    const glm::vec3 d{sqrtf(magSquared) * glm::normalize(direction)};
    return glm::vec4{d, w};
}

__device__ glm::vec3 lerp(float t, const glm::vec3& a, const glm::vec3& b) {
    t = glm::clamp(t, 0.f, 1.f);

    return (t * b) + ((1 - t) * a);
}

__device__ float rng(unsigned state) {
    state *= (state + 340147) * (state + 1273128) * (state + 782243);

    return (float)state / UINT_MAX;
}

__device__ float2 randomVec2(unsigned state) {
    return make_float2(rng(state << 0), rng(state + 1 << 1));
}

__device__ float3 randomVec3(unsigned state) {
    return make_float3(
        rng(state << 0), rng(state + 1 << 1), rng(state + 2 << 2)
    );
}

__device__ bool withinError(double approx, double expected, double tolerance) {
    return glm::abs(approx - expected) < tolerance;
}

__device__ bool isH3Point(const glm::vec4& v) {
    static constexpr float EPS{0.001};

    const bool positiveW{v.w > 0};
    const bool constCurvature{withinError(hypDot(v, v), -1, EPS)};

    return positiveW && constCurvature;
}

__device__ bool isH3Dir(const glm::vec4& p, const glm::vec4& dir) {
    const bool isDirection{withinError(hypDot(dir, dir), 1, 0.001)};
    const bool isTangentVector{withinError(hypDot(dir, p), 0, 0.001)};
    return isDirection && isTangentVector;
}

__device__ float hypDistance(const glm::vec4& u, const glm::vec4& v) {
    const float bUV = -hypDot(u, v);
    return acosh(bUV);
}

__device__ double hyperbolicSphereSDF(
    const glm::vec4& p,
    float r,
    const glm::vec4& center
) {
    return hypDistance(p, center) - r;
}

__device__ glm::vec4 hypGeoFlowPos(
    const glm::vec4& pos,
    const glm::vec4& dir,
    float t
) {
    return {cosh(t) * pos + sinh(t) * dir};
}

__device__ glm::vec4 hypGeoFlowDir(
    const glm::vec4& pos,
    const glm::vec4& dir,
    float t
) {
    return {sinh(t) * pos + cosh(t) * dir};
}

__device__ void geodesicFlowHyperbolic(
    const glm::vec4& pos,
    const glm::vec4& dir,
    float t,
    glm::vec4* new_pos,
    glm::vec4* new_dir
) {
    *new_pos = hypGeoFlowPos(pos, dir, t);
    *new_dir = hypGeoFlowDir(pos, dir, t);
}

__device__ glm::vec4 correctH3Point(const glm::vec4 p) {
    const float minkowskiNorm = hypDot(p, p);

    if (std::abs(minkowskiNorm + 1) > 1e-6) {
        // Position is not in H^3, project it back onto the hyperboloid
        const float scaleFactor = std::sqrt(-1 / minkowskiNorm);
        const glm::vec4 correctedPosition = p * scaleFactor;
        return correctedPosition;
    } else
        return p;
}

} // namespace CUDAMath
