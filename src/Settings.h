#pragma once

inline constexpr bool INTERACTIVE_MODE{ 1 };

inline constexpr int RAYS_PER_PIXEL{ 4 };

inline constexpr int MAX_NUM_BOUNCES{ 4};

inline constexpr int OFFLINE_WIDTH{ 16 << 5 };
inline constexpr int OFFLINE_HEIGHT{ 9 << 5 };

inline constexpr int INTERACTIVE_WIDTH{ 16 << 2 };
inline constexpr int INTERACTIVE_HEIGHT{ 9 << 2 };

inline constexpr unsigned WINDOW_SCALE{ 1 << 4 };

// dont modify here
inline int rngSeed{ 0 };
inline unsigned globalTick{ 0 };
inline bool isDebugRay{};
inline float hypCamPosX;
inline float hypCamPosY;
inline float hypCamPosZ;
inline float hypCamPosW;

inline float prePitch{};
inline float preYaw{};

inline int hyperbolicErrorAcc{ 0 };

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


inline constexpr unsigned WINDOW_WIDTH{  INTERACTIVE_MODE ? INTERACTIVE_WIDTH  : OFFLINE_WIDTH  };
inline constexpr unsigned WINDOW_HEIGHT{ INTERACTIVE_MODE ? INTERACTIVE_HEIGHT : OFFLINE_HEIGHT };
