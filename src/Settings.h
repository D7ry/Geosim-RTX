#pragma once

inline constexpr bool INTERACTIVE_MODE{ 1 };

inline constexpr int RAYS_PER_PIXEL{ INTERACTIVE_MODE ? 1 : 50 };

inline constexpr int MAX_NUM_BOUNCES{ INTERACTIVE_MODE ? 3 : 6 };

inline constexpr int OFFLINE_WIDTH{ 16 << 5 };
inline constexpr int OFFLINE_HEIGHT{ 9 << 5 };

inline constexpr int INTERACTIVE_WIDTH{ 16 << 3 };
inline constexpr int INTERACTIVE_HEIGHT{ 9 << 3 };

inline constexpr unsigned WINDOW_SCALE{ 1 << 3 };

inline int rngSeed{ 0 };
inline unsigned globalTick{ 0 };