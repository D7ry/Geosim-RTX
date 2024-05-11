#pragma once

#include <glm/glm.hpp>

namespace CUDAMath
{

__device__ glm::vec3 lerp(float t, const glm::vec3& a, const glm::vec3& b)
{
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

} // namespace CUDAMath
