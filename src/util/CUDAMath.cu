#include "CUDAMath.h"

namespace CUDAMath {

float rng(unsigned state) {
    state *= (state + 340147) * (state + 1273128) * (state + 782243);
    return (float)state / std::numeric_limits<unsigned>::max();
}


glm::vec2 randomVec2(unsigned state) {
    return glm::vec2(rng(state << 0), rng(state + 1 << 1));
}
}
