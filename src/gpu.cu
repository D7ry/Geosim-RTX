#include "gpu.h"
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include <iostream>
#include <stdio.h>

#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Settings.h"
#include "util/CUDAMath.h"

// #include "gpu_sdf.h"
#include "util/Math.h"

#include "util/Ray.h"

#include "render_settings.h"

__host__ void print_cuda_error() {
    auto err = cudaGetLastError();
    if (err) {
        auto err_str = cudaGetErrorString(err);
        printf("Last CUDA Error: %s\n", err_str);
    }
}

namespace CudaPlayground
{
__global__ void cudaHello() {
    printf("Hello World from CUDA thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    glm::vec3 a(1.0f, 2.0f, 3.0f);
    glm::vec3 b(4.0f, threadIdx.x, blockIdx.x);
    glm::vec3 c = a + b;
    printf("c = [%f, %f, %f]\n", c.x, c.y, c.z);
}

void play() {
    std::cout << "Running CUDA Playground" << std::endl;
    int numBlocks = 16;
    int threadsPerBlock = 16;
    cudaHello<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    std::cout << "CUDA Playground finished" << std::endl;
}
} // namespace CudaPlayground

namespace CUDAStruct
{

__host__ CubeMap* loadCubeMap(const char* filename) {
    int channels;
    int width, height;
    unsigned char* img
        = stbi_load(filename, &width, &height, &channels, 3); // Load as RGB
    glm::vec3* colors = new glm::vec3[width * height];
    for (int i = 0; i < width * height; ++i) {
        colors[i] = glm::vec3(
            img[i * 3] / 255.0f,
            img[i * 3 + 1] / 255.0f,
            img[i * 3 + 2] / 255.0f
        );
    }
    CubeMap* cubeMap = new CubeMap();
    cubeMap->width = width;
    cubeMap->height = height;
    glm::vec3* data;
    cudaMalloc(&data, sizeof(glm::vec3) * width * height);
    cubeMap->data = data;
    cubeMap->width = width;
    cubeMap->height = height;
    cudaMemcpy(
        cubeMap->data,
        colors,
        sizeof(glm::vec3) * width * height,
        cudaMemcpyHostToDevice
    );

    CubeMap* cubeMapDevice;
    cudaMalloc(&cubeMapDevice, sizeof(CubeMap));
    cudaMemcpy(cubeMapDevice, cubeMap, sizeof(CubeMap), cudaMemcpyHostToDevice);

    free(colors);
    free(img);
    free(cubeMap);

    return cubeMapDevice;
    stbi_image_free(img);
}

inline __device__ double SpherePrimitive_SDF(
    const SpherePrimitive* sphere,
    const glm::vec4& p,
    const glm::vec4& positionWorldSpace
) {
    const glm::vec3 euclideanPosition{
        glm::vec3(positionWorldSpace) + sphere->position
    };

    const glm::vec4 hyperbolicPosition{CUDAMath::constructHyperboloidPoint(
        euclideanPosition, glm::length(euclideanPosition)
    )};

    const float dist = CUDAMath::hyperbolicSphereSDF(
        p, // todo: is w supposed to be 0?
        sphere->radius_dynamic,
        hyperbolicPosition
    );

    return dist;
}

__device__ glm::vec3 intersection_evaluate_outgoing(
    const CUDAStruct::Intersection* intersection
) {
    glm::vec3 outgoing{0.f};

    // if (reflected)
    {
        const glm::vec3 lambert{
            CUDAMath::randomHemisphereDir(0, intersection->normal)
        };
        const glm::vec3 mirror{
            glm::reflect(intersection->incidentDir, intersection->normal)
        };

        outgoing = CUDAMath::lerp(intersection->mat_roughness, mirror, lambert);
    }
    // else // light diffused or refracted
    {}

    return outgoing;
}
} // namespace CUDAStruct

namespace RendererCUDA
{

void check_device() {
    printf("Checking CUDA device...\n");
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf(
            "cudaGetDeviceCount returned %d\n-> %s\n",
            static_cast<int>(error_id),
            cudaGetErrorString(error_id)
        );
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
}

__device__ glm::vec3 environmentalLight(
    const glm::vec3& dir,
    const CUDAStruct::Scene* scene
) {

    glm::vec3 lightDir
        = glm::vec3(sinf(scene->dayTime), cosf(scene->dayTime), 0);
    lightDir = glm::normalize(lightDir);

    const glm::vec3 noonColor{0.5};

    const glm::vec3 sunsetColor{0.5, .3, .15};

    const float interpolation
        = glm::max(0.f, glm::dot(lightDir, glm::vec3{0, 1, 0}));

    const glm::vec3 lightColor
        = CUDAMath::lerp(interpolation, sunsetColor, noonColor);

    auto light_intensity = glm::dot(lightDir, dir);

    glm::vec3 light{glm::max(0.f, glm::dot(lightDir, dir)) * lightColor};

    return light;
}

// entirely written by chatgipidy
__device__ glm::vec2 DirectionToEnvMapCoords(const glm::vec3& direction) {
    // Convert direction to spherical coordinates
    float longitude = atan2(
        direction.z, direction.x
    ); // Angle in XY plane from positive X axis
    float latitude = acos(direction.y); // Angle from positive Y axis

    // Normalize longitude to range [0, 2*pi)
    if (longitude < 0.0f)
        longitude += glm::two_pi<float>();

    // Map spherical coordinates to uv coordinates [0, 1]
    float u = longitude / glm::two_pi<float>();
    float v = 1.0f - latitude / glm::pi<float>();

    u = glm::clamp(u, 0.0f, 1.0f);
    v = glm::clamp(v, 0.0f, 1.0f);

    return glm::vec2(u, v);
}

__device__ glm::vec3 sample_environment_map(
    const glm::vec3& dir,
    const CUDAStruct::Scene* scene
) {
    glm::vec2 directionUV = DirectionToEnvMapCoords(dir);

    int index
        = (int)(directionUV.y * scene->cubemap->height) * scene->cubemap->width
          + (int)(directionUV.x * scene->cubemap->width);

    if (index >= scene->cubemap->width * scene->cubemap->height)
        return glm::vec3(0.0f, 0.0f, 0.0f);

    return scene->cubemap->data[index];
}

__device__ glm::vec3 evaluate_light_path(
    glm::vec3 origin,
    CUDAStruct::Intersection* hits,
    int num_hits,
    const CUDAStruct::Scene* scene
) {

    if (num_hits != 0) {
        // printf("Num hits: %d\n", num_hits);
    }
    glm::vec3 incomingLight{0};

    // if ray bounced off a surface and never hit anything after
    const bool reachedEnvironment{num_hits < MAX_NUM_BOUNCES};

    // if primary ray hits nothing, use that as environment bound vector
    const glm::vec3 environmentDir{
        num_hits == 0 ? origin : hits[num_hits - 1].outgoingDir
    };

    if (reachedEnvironment) {
        // if (environmentDir != glm::vec3{0}) {
        // printf("Environment dir: %f, %f, %f\n", environmentDir.x,
        // environmentDir.y, environmentDir.z);
        // }
        incomingLight += environmentalLight(environmentDir, scene);
        incomingLight += sample_environment_map(environmentDir, scene);
        // if (incomingLight != glm::vec3{0}) {
        // printf("Environment light: %f, %f, %f\n", incomingLight.x,
        //        incomingLight.y, incomingLight.z);
        // }
    }

    // reverse iterate from the start of a path of light
    for (int i = num_hits - 1; i >= 0; i--) {
        const CUDAStruct::Intersection* hit = hits + i;

        // light emitted from hit surface
        const glm::vec3 emittedLight
            = hit->mat_emissionStrength * hit->mat_emissionColor;

        // cos(theta) term
        const float lightStrength{
            1 // glm::max(0.f, glm::dot(hit.normal, -hit.incidentDir))
        };

        // basically the rendering equation
        // incomingLight = emittedLight + (2.f * Math::BRDF(hit) * incomingLight
        // * lightStrength);
        incomingLight
            = emittedLight + (hit->mat_albedo * incomingLight * lightStrength);
    }

    // printf("Incoming light: %f, %f, %f\n", incomingLight.x, incomingLight.y,
    //        incomingLight.z);
    return incomingLight;
}

// TODO: no kd tree traversal yet
__device__ void getClosestPrimitive(
    const glm::vec4& p,
    const CUDAStruct::Scene* scene,
    double* distance,
    const CUDAStruct::SpherePrimitive** closestPrimitive
) {

    for (int i = 0; i < scene->num_geometries; i++) {
        const CUDAStruct::Geometry* object = scene->geometries + i;
        const glm::vec4 objHypPos{CUDAMath::constructHyperboloidPoint(
            object->position, glm::length(object->position)
        )};
        for (int j = 0; j < object->num_spheres; j++) {
            const CUDAStruct::SpherePrimitive* sphere = object->spheres + j;
            const double d
                = CUDAStruct::SpherePrimitive_SDF(sphere, p, objHypPos);
            if (d < *distance) {
                *distance = d;
                *closestPrimitive = sphere;
                // printf("Closest primitive found at %f, %f, %f\n", p.x, p.y,
                // p.z);
            }
        }
    }
}

__device__ double getClosestDistance(
    const glm::vec4& p,
    const CUDAStruct::Scene* scene
) {
    double minDistance{1000000};
    for (int i = 0; i < scene->num_geometries; i++) {
        const CUDAStruct::Geometry* object = scene->geometries + i;
        const glm::vec4 objHypPos{CUDAMath::constructHyperboloidPoint(
            object->position, glm::length(object->position)
        )};

        for (int j = 0; j < object->num_spheres; j++) {
            const CUDAStruct::SpherePrimitive* sphere = object->spheres + j;
            const double d
                = CUDAStruct::SpherePrimitive_SDF(sphere, p, objHypPos);
            minDistance = glm::min(minDistance, d);
        }
    }
    return minDistance;
}

__device__ glm::vec3 computeNormal(
    const glm::vec4& p,
    const CUDAStruct::Scene* scene
) {

    static constexpr float EPSILON{0.001f};
    // hyperbolic normalization
    // Compute basis vectors for the tangent hyperplane at p
    glm::vec4 basis_x = CUDAMath::hypNormalize(glm::vec4(p.w, 0.0f, 0.0f, p.x));
    glm::vec4 basis_y = glm::vec4(0.0f, p.w, 0.0f, p.y);
    glm::vec4 basis_z = glm::vec4(0.0f, 0.0f, p.w, p.z);

    // Gram-Schmidt orthogonalization
    basis_y = CUDAMath::hypNormalize(
        basis_y - CUDAMath::hypDot(basis_y, basis_x) * basis_x
    );
    basis_z = CUDAMath::hypNormalize(
        basis_z - CUDAMath::hypDot(basis_z, basis_x) * basis_x
        - CUDAMath::hypDot(basis_z, basis_y) * basis_y
    );

    // Compute gradients using finite differences
    float xGradient = getClosestDistance(p + EPSILON * basis_x, scene)
                      - getClosestDistance(p - EPSILON * basis_x, scene);
    float yGradient = getClosestDistance(p + EPSILON * basis_y, scene)
                      - getClosestDistance(p - EPSILON * basis_y, scene);
    float zGradient = getClosestDistance(p + EPSILON * basis_z, scene)
                      - getClosestDistance(p - EPSILON * basis_z, scene);

    // Construct the normal vector
    glm::vec4 normal = CUDAMath::hypNormalize(
        xGradient * basis_x + yGradient * basis_y + zGradient * basis_z
    );

    return normal;
}

// Get the closest intersection, returns true if hit something and stores the
// intersection in the buffer
__device__ bool get_closest_intersection(
    glm::vec3 ray_origin,
    glm::vec3 ray_dir,
    CUDAStruct::Intersection*
        intersection_buffer, // can directly store the intersection into
    const CUDAStruct::Scene* scene,
    float hypCamPosX,
    float hypCamPosY,
    float hypCamPosZ,
    float hypCamPosW
) {
    //
    //
    float totalDistanceTraveled = 0.0;
    const int MAX_NUM_STEPS = 8;
    const float MIN_HIT_DISTANCE = .01;
    const float MAX_TRACE_DISTANCE = 20; // max float value on order of 10e38

    // translate camera position from euclidean to hyperbolic (translated to
    // hyperboloid)
    glm::vec4 hypPos{
        CUDAMath::constructHyperboloidPoint(ray_origin, glm::length(ray_origin))
    };

    const glm::vec4 p{hypCamPosX, hypCamPosY, hypCamPosZ, hypCamPosW};
    //
    // generate direction then transform to hyperboloid
    const glm::vec4 hyperbolicPos{
        p // Math::correctH3Point(p)
    };
    //   //
    const glm::vec4 d{ray_dir, 0};
    //
    const glm::vec4 hyperbolicDir{CUDAMath::correctDirection(p, d)};
    //
    glm::vec4 marchPos{hyperbolicPos};
    glm::vec4 marchDir{hyperbolicDir};
    //   //
    for (int i = 0; i < MAX_NUM_STEPS; ++i) {
        if (!CUDAMath::isH3Point(marchPos)
            || !CUDAMath::isH3Dir(marchPos, marchDir)) {
            // hyperbolicErrorAcc++; TODO: implement error handling
        }
        //       //
        //       //
        double dist = DBL_MAX;
        const CUDAStruct::SpherePrimitive* closestPrimitive = nullptr;

        getClosestPrimitive(marchPos, scene, &dist, &closestPrimitive);

        // if (closestPrimitive == nullptr) {
        //     printf("Boutta crash\n");
        // }
        //
        //       double dist = closest.first;
        //       // we hit something

        if (dist < MIN_HIT_DISTANCE) {
            // glm::vec3 normal
            //     = primitive.material.get()
            //           ->albedo; // for rough quick rendering/debugging

            // printf("Hit something at %f, %f, %f\n", marchPos.x,
            // marchPos.y, marchPos.z);
            glm::vec3 normal{0}; // TODO: implemenet normal computation

            // if (!RENDER_WITH_POTATO_SETTINGS)
            normal = computeNormal(marchPos, scene);

            // populate intersection buffer

            CUDAStruct::Intersection* intersection = intersection_buffer;
            intersection->position = marchPos;
            intersection->normal = normal;
            intersection->incidentDir = ray_dir;
            intersection->outgoingDir
                = CUDAStruct::intersection_evaluate_outgoing(intersection);

            intersection->mat_albedo = closestPrimitive->mat_albedo;
            intersection->mat_emissionColor
                = closestPrimitive->mat_emissionColor;
            intersection->mat_emissionStrength
                = closestPrimitive->mat_emissionStrength;
            intersection->mat_roughness = closestPrimitive->mat_roughness;

            return true;
        } else if (!CUDAMath::isH3Point(marchPos) 
                || !CUDAMath::isH3Dir(marchPos, marchDir) 
                || totalDistanceTraveled + dist > MAX_TRACE_DISTANCE 
                || glm::isnan(marchPos.x) 
                || glm::isnan(marchDir.x))
        {
            break;
        } else {
            const float ss{(float)dist / 1}; // substep size
            while (dist > 0) {
                glm::vec4 new_pos;
                glm::vec4 new_dir;

                // march the ray forward
                CUDAMath::geodesicFlowHyperbolic(
                    marchPos, marchDir, ss, &new_pos, &new_dir
                );

                marchPos = CUDAMath::correctH3Point(new_pos);
                marchDir = CUDAMath::correctDirection(marchPos, new_dir);
                totalDistanceTraveled += ss;
                dist -= ss;
            }
        }
    }

    return false;
    //
    //   if (closestHit)
    //       return true;
    //
    //   return false;
}

// trace a single ray and return the color
__device__ glm::vec3 trace_ray(
    glm::vec3 origin,
    glm::vec3 direction,
    CUDAStruct::Intersection* hitsBuffer_ray, // guaranteed to allow for storing
                                              // num_bounces intersection
    int num_bounces,
    const CUDAStruct::Scene* scene,
    float hypCamPosX,
    float hypCamPosY,
    float hypCamPosZ,
    float hypCamPosW
) {

    int num_hits = 0;
    for (int i = 0; i < num_bounces; i++) {
        CUDAStruct::Intersection* hitsBuffer_bounce
            = hitsBuffer_ray
              + i; // if intersection happens, store it into this buffer

        bool hit = get_closest_intersection(
            origin,
            direction,
            hitsBuffer_bounce,
            scene,
            hypCamPosX,
            hypCamPosY,
            hypCamPosZ,
            hypCamPosW

        );

        if (!hit) {
            break;
        }

        // update ray origin and direction
        origin = hitsBuffer_bounce->position
                 + hitsBuffer_bounce->outgoingDir * 0.02f;
        direction = hitsBuffer_bounce->outgoingDir;

        num_hits++;
    }

    CUDAStruct::Intersection* hits = hitsBuffer_ray;
    const glm::vec3 finalColor{
        evaluate_light_path(direction, hits, num_hits, scene)
    };

    return finalColor;
}

// Render a single pixel
__global__ void render_pixel(
    const CUDAStruct::Scene* scene,
    const Camera* camera,
    int width,
    int height,
    int rays_per_pixel,
    int bounces_per_ray,
    glm::vec3* frameBuffer_device,
    CUDAStruct::Intersection* hitsBuffer_device,
    float hypCamPosX,
    float hypCamPosY,
    float hypCamPosZ,
    float hypCamPosW
) {
    // printf("Rendering pixel. Width: %d, Height: %d, rays_per_pixel: %d,
    // bounces_per_ray: %d\n", width, height, rays_per_pixel, bounces_per_ray);

    // TODO: these should be passed in as parameters
    float aspectRatio = width / height; // w : h
    // todo figure out why FOV seems "off"
    float fovComponent{tanf(camera->FOV / 2.f)};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // check if out of bounds
    if (x >= width || y >= height) {
        return;
    }

    // printf("Rendering pixel at x: %d, y: %d\n", x, y);

    CUDAStruct::Intersection* hitsBuffer_pixel
        = hitsBuffer_device
          + (x + (y * width) * rays_per_pixel * bounces_per_ray);

    glm::vec3 final_color{0.f};

    const glm::vec2 ndc{(x + 0.5f) / width, (y + 0.5f) / height};

    glm::uvec2 pixelCoord{ndc.x * width, ndc.y * height};

    for (int i = 0; i < rays_per_pixel; i++) {
        float2 rayOffset = CUDAMath::randomVec2(
            i + (x * width) + y
        ); // TODO: not sure if rng works

        const glm::vec2 ndcAliased{
            (x + rayOffset.x) / width, (y + rayOffset.y) / height
        };

        // screen space
        glm::vec2 coord = glm::vec2{
            ((2.f * ndc.x) - 1.f) * fovComponent * aspectRatio,
            1.f - (2.f * ndc.y) * fovComponent // flip vertically so +y is up
        };

        // ray coords in world space
        glm::vec4 start{camera->position, 1.f};
        glm::vec4 dir{coord.x, coord.y, -1.f, 0};

        // transform ray to view space
        dir = glm::normalize(dir);
        dir = dir * camera->viewMat;

        CUDAStruct::Intersection* hitsBuffer_ray
            = hitsBuffer_pixel + (i * bounces_per_ray);

        glm::vec3 color = trace_ray(
            start,
            dir,
            hitsBuffer_ray,
            bounces_per_ray,
            scene,
            hypCamPosX,
            hypCamPosY,
            hypCamPosZ,
            hypCamPosW
        );

        final_color += color;
    }

    final_color /= RAYS_PER_PIXEL;

    { // writeback to framebuffer
        int frameBufferIndex = x + (y * width);
        frameBuffer_device[frameBufferIndex] = final_color;
    }
}

__host__ void render(
    const CUDAStruct::Scene* scene,
    const Camera* camera,
    Image* image
) {
    // printf("Rendering with CUDA\n");
    print_cuda_error();
    float aspectRatio = (float)image->width / image->height; // w : h

    // todo figure out why FOV seems "off"
    float fovComponent{tanf(camera->FOV / 2.f)};

    int width = image->width;
    int height = image->height;

    dim3 blockDims = dim3(16, 16); // 256 threads per block
    dim3 gridDims = dim3(
        (width + blockDims.x - 1) / blockDims.x,
        (height + blockDims.y - 1) / blockDims.y
    );

    // allocate FB
    glm::vec3* frameBuffer = image->pixels.data();
    glm::vec3* frameBuffer_Device;
    size_t frameBuffer_size = width * height;
    cudaMalloc(&frameBuffer_Device, frameBuffer_size * sizeof(glm::vec3));

    // allocate buffer to store intersections data
    CUDAStruct::Intersection* hitsBuffer_Device;
    cudaMalloc(
        &hitsBuffer_Device,
        width * height * RAYS_PER_PIXEL * MAX_NUM_BOUNCES
            * sizeof(CUDAStruct::Intersection)
    ); // each ray (bounce) needs to store its hit

    // allocate mem for cudascene
    CUDAStruct::Scene* scene_Device;
    cudaMalloc(&scene_Device, sizeof(CUDAStruct::Scene));
    cudaMemcpy(
        scene_Device, scene, sizeof(CUDAStruct::Scene), cudaMemcpyHostToDevice
    );

    // allocate camera
    Camera* camera_Device;
    cudaMalloc(&camera_Device, sizeof(Camera));
    cudaMemcpy(camera_Device, camera, sizeof(Camera), cudaMemcpyHostToDevice);

    render_pixel<<<gridDims, blockDims>>>(
        scene_Device,
        camera_Device,
        width,
        height,
        RAYS_PER_PIXEL,
        MAX_NUM_BOUNCES,
        frameBuffer_Device,
        hitsBuffer_Device,
        // FIXME: these values shouldn't be passed as params, instead store them
        // in camera
        hypCamPosX,
        hypCamPosY,
        hypCamPosZ,
        hypCamPosW
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        frameBuffer,
        frameBuffer_Device,
        width * height * sizeof(glm::vec3),
        cudaMemcpyDeviceToHost
    );
    cudaFree(frameBuffer_Device);
    cudaFree(hitsBuffer_Device);
    cudaFree(scene_Device);
    cudaFree(camera_Device);

    // print_cuda_error();
    // printf("Finished rendering with CUDA\n");
}

} // namespace RendererCUDA
