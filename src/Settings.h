#pragma once

inline constexpr bool INTERACTIVE_MODE{ 1 };

inline constexpr int RAYS_PER_PIXEL{ INTERACTIVE_MODE ? 1 : 50 };

inline constexpr int MAX_NUM_BOUNCES{ INTERACTIVE_MODE ? 4 : 8 };

inline constexpr int OFFLINE_WIDTH{ 16 << 5 };
inline constexpr int OFFLINE_HEIGHT{ 9 << 5 };

inline constexpr int INTERACTIVE_WIDTH{ 16 << 3 };
inline constexpr int INTERACTIVE_HEIGHT{ 9 << 3 };

inline constexpr unsigned WINDOW_SCALE{ 1 << 3 };

// dont modify here
inline int rngSeed{ 0 };
inline unsigned globalTick{ 0 };
inline bool isDebugRay{ false };

inline constexpr bool DEBUG{ false && INTERACTIVE_MODE };
inline constexpr bool PRINT_DEBUG{ true && DEBUG };
inline constexpr bool VISUALIZE_DEBUG_RAY{ true && DEBUG };

inline constexpr bool ANTIALIAS{ true };