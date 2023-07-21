#pragma once

inline constexpr bool INTERACTIVE_MODE{ 1 };

inline constexpr int RAYS_PER_PIXEL{ INTERACTIVE_MODE ? 1 : 100 };
inline constexpr bool USE_RNG_FOR_AA{ true };

inline constexpr int MAX_NUM_BOUNCES{ INTERACTIVE_MODE ? 3 : 5 };

inline constexpr int OFFLINE_WIDTH{ 16 << 6 };
inline constexpr int OFFLINE_HEIGHT{ 9 << 6 };

inline constexpr int INTERACTIVE_WIDTH{ 16 << 3 };
inline constexpr int INTERACTIVE_HEIGHT{ 9 << 3 };

inline int tickG{ 0 };