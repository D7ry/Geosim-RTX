#pragma once

#include <glm/glm.hpp>

namespace CUDAMath
{

float rng(unsigned state);

glm::vec2 randomVec2(unsigned state);

} // namespace CUDAMath
