#pragma once

inline constexpr bool INTERACTIVE_MODE{ 1 };

inline constexpr int RAYS_PER_PIXEL{ INTERACTIVE_MODE ? 1 : 1 };

inline constexpr int MAX_NUM_BOUNCES{ INTERACTIVE_MODE ? 3 : 4 };

inline constexpr int OFFLINE_WIDTH{ 16 << 5 };
inline constexpr int OFFLINE_HEIGHT{ 9 << 5 };

inline constexpr int INTERACTIVE_WIDTH{ 16 << 2 };
inline constexpr int INTERACTIVE_HEIGHT{ 9 << 2 };

inline constexpr unsigned WINDOW_SCALE{ 1 << 4 };

// dont modify here
inline int rngSeed{ 0 };
inline unsigned globalTick{ 0 };
inline bool isDebugRay{};

inline constexpr bool DEBUG{ true && INTERACTIVE_MODE };
inline constexpr bool PRINT_DEBUG_LIGHTING{ false && DEBUG };
inline constexpr bool PRINT_DEBUG_MARCHING{ false && DEBUG };
inline constexpr bool VISUALIZE_DEBUG_RAY{ true && DEBUG };
inline constexpr bool LOG_MARCH_PATH{ false && DEBUG };

inline constexpr bool RENDER_NORMALS{ true };
inline constexpr bool RENDER_WITH_POTATO_SETTINGS{ true };

inline constexpr bool ANTIALIAS{ false };

inline constexpr bool RAY_MARCH{ true };
inline constexpr bool EUCLIDEAN{ false && RAY_MARCH };