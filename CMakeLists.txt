cmake_minimum_required(VERSION 3.22)
# set(CMAKE_C_COMPILER "/usr/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

enable_language(CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(rodent_raytracer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# SFML fuckery
include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.x)
FetchContent_MakeAvailable(SFML)

add_executable(
    rodent_raytracer
    src/main.cpp
    src/Primitive.cpp
    src/Renderer.cpp
    src/util/Math.cpp
    src/util/Window.cpp
    src/gpu.cu
    src/Camera.cpp
    src/Image.cpp
)


set(SFML_LIBRARIES sfml-graphics sfml-window sfml-system)

target_include_directories(rodent_raytracer PUBLIC ${SFML_INCLUDE_DIRS})
target_link_libraries(rodent_raytracer PRIVATE ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})


# CUDA stuff
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
set_target_properties(rodent_raytracer PROPERTIES CUDA_ARCHITECTURES "60")
