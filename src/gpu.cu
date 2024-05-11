#include "gpu.h"
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include <iostream>
#include <stdio.h>

#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Scene.h"
#include "Settings.h"
#include "util/CUDAMath.h"

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


__device__ glm::vec3 environmentalLight(const glm::vec3& dir) {
    const float dayTime{64 / 128.f}; // FIXME: actually add global tick

    // glm::vec3 lightDir{ sinf(dayTime), cosf(dayTime), 0 };
    glm::vec3 lightDir{0, 1, 0};
    lightDir = glm::normalize(lightDir);

    const glm::vec3 noonColor{1};
    const glm::vec3 sunsetColor{1, .6, .3};

    const float interpolation
        = glm::max(0.f, glm::dot(lightDir, glm::vec3{0, 1, 0}));

    const glm::vec3 lightColor
        = CUDAMath::lerp(interpolation, sunsetColor, noonColor);

    glm::vec3 light{glm::max(0.f, glm::dot(lightDir, dir)) * lightColor};

    return light;
}

__device__ glm::vec3 evaluate_light_path(
    glm::vec3 origin,
    Intersection* hits,
    int num_hits
) {

    glm::vec3 incomingLight{0};

    // if ray bounced off a surface and never hit anything after
    const bool reachedEnvironment{num_hits < MAX_NUM_BOUNCES};

    // if primary ray hits nothing, use that as environment bound vector
    const glm::vec3 environmentDir{
        num_hits == 0 ? origin : hits[num_hits - 1].outgoingDir
    };

    if (reachedEnvironment) {
        incomingLight += environmentalLight(environmentDir);
    }

    // reverse iterate from the start of a path of light
    for (int i = num_hits - 1; i >= 0; i--) {
        const Intersection& hit{hits[i]};

        // light emitted from hit surface
        const glm::vec3 emittedLight{
            hit.material.emissionColor * hit.material.emissionStrength
        };

        // cos(theta) term
        const float lightStrength{
            1 // glm::max(0.f, glm::dot(hit.normal, -hit.incidentDir))
        };

        // basically the rendering equation
        // incomingLight = emittedLight + (2.f * Math::BRDF(hit) * incomingLight
        // * lightStrength);
        incomingLight = emittedLight
                        + (hit.material.albedo * incomingLight * lightStrength);
    }

    return incomingLight;
}


bool march(glm::vec3 origin, const Scene& scene)
{
// 	std::unique_ptr<Intersection> closestHit;
//
// 	//if (isDebugRay)
// 	//	std::cout << "here!\n";
//
// 	float totalDistanceTraveled = 0.0;
// 	const int MAX_NUM_STEPS = 8;
// 	const float MIN_HIT_DISTANCE = .01;
// 	const float MAX_TRACE_DISTANCE = 20;	// max float value on order of 10e38
//
// 	// translate camera position from euclidean to hyperbolic (translated to hyperboloid)
// 	//glm::vec4 hypPos{ Math::constructHyperboloidPoint(
// 	//	ray.origin,
// 	//	glm::length(ray.origin)
// 	//)};
//
//
//
// 	
// 	const glm::vec4 p{ hypCamPosX, hypCamPosY, hypCamPosZ, hypCamPosW };
//
// 	// generate direction then transform to hyperboloid
// 	const glm::vec4 hyperbolicPos{
// 		p//Math::correctH3Point(p)
// 	};
//
// 	const glm::vec4 d{ ray.dir, 0 };
//
// 	const glm::vec4 hyperbolicDir{
// 		Math::correctDirection(p,d)
// 	};
//
// 	glm::vec4 marchPos{ hyperbolicPos };
// 	glm::vec4 marchDir{ hyperbolicDir };
//
// 	//if (isDebugRay && PRINT_DEBUG_MARCHING && !EUCLIDEAN)
// 	//	std::cout << "Starting March!\n";
//
// 	if (isDebugRay && LOG_MARCH_PATH)
// 	{
// 		rayMarchPathPositions.clear();
// 		rayMarchPathDirections.clear();
// 	}
//
// 	for (int i = 0; i < MAX_NUM_STEPS; ++i)
// 	{
// 		if (!Math::isH3Point(marchPos) || !Math::isH3Dir(marchPos, marchDir))
// 		{
// 			hyperbolicErrorAcc++;
// 		}
//
// 		
// 		if (isDebugRay)
// 		{
// 			if (!Math::isH3Point(marchPos))
// 				std::cout << "ray not in h3 (step: " << i << ")\n";
// 			if (!Math::isH3Dir(marchPos, marchDir))
// 				std::cout << "raydir not in h3 (step: " << i << ")\n";
//
// 			{
// 				//std::cout << "ray not in h3 (step: " << i << "\n)";
// 				//Math::printH3Point("p", marchPos);
// 				//Math::printH3Dir("d", marchPos, marchDir);
// 				
// 				//std::cout << "bad step\npos: " << toStr(marchPos)
// 				//	<< ", dot: " << Math::hypDot(marchPos, marchPos) 
// 				//	<< "\ndir: " << toStr(marchDir)
// 				//	<< ", dot: " << Math::hypDot(marchDir, marchDir) << '\n';
// 			}
// 		}
//
// 		const auto closest = getClosestPrimitive(marchPos, scene);
//
// 		double dist = closest.first;
//
// 		// we hit something
// 		if (dist < MIN_HIT_DISTANCE)
// 		{
// 			const Primitive& primitive = closest.second;
// 			glm::vec3 normal = primitive.material.get()->albedo; // for rough quick rendering/debugging
//
// 			if (!RENDER_WITH_POTATO_SETTINGS)
// 				normal = computeNormal(marchPos, scene);
// 			
// 			RayIntersection rayIntersection{ ray, -1, marchPos, normal };
// 			Intersection i{ *primitive.material.get(), rayIntersection };
//
// 			closestHit = std::make_unique<Intersection>(i);
// 		}
// 		else if (
// 				!Math::isH3Point(marchPos) ||
// 				!Math::isH3Dir(marchPos, marchDir) ||
// 				totalDistanceTraveled + dist > MAX_TRACE_DISTANCE ||
// 				std::isnan(marchPos.x) ||
// 				std::isnan(marchDir.x)
// 			)
// 		{
// 			if (isDebugRay && PRINT_DEBUG_MARCHING)
// 				std::cout << "toofar, termininat, dist = " << dist << '\n';
// 			break;
// 		}
// 		else
// 		{
// 			const float ss{ (float)dist / 1 };	// substep size
// 			while (dist > 0)
// 			{
// 				auto newMarch = march(marchPos, marchDir, ss);
//
// 				if (isDebugRay && LOG_MARCH_PATH)
// 				{
// 					if (rayMarchPathDirections.empty())
// 						rayMarchPathDirections.emplace_back(marchDir);
// 					if (rayMarchPathPositions.empty())
// 						rayMarchPathPositions.emplace_back(marchPos);
//
// 					rayMarchPathPositions.emplace_back(newMarch.first);
// 					rayMarchPathDirections.emplace_back(newMarch.second);
// 				}
//
// 				if (isDebugRay && PRINT_DEBUG_MARCHING && !EUCLIDEAN)
// 				{
// 					//std::cout << "Step size: " << dist
// 					//	<< ", sub step size: " << ss
// 					//	<< "\npos: " << toStr(marchPos) << " => " << toStr(newMarch.first)
// 					//	<< "\ndir: " << toStr(marchDir) << " => " << toStr(newMarch.second)
// 					//	<< '\n';
// 				}
// 				
// 				marchPos = Math::correctH3Point(newMarch.first);
// 				//marchPos = newMarch.first;
// 				//marchDir = Math::hypNormalize(newMarch.second);
// 				//marchDir = Math::hypDirection(marchPos, newMarch.second);
// 				marchDir = Math::correctDirection(marchPos, newMarch.second);
// 				totalDistanceTraveled += ss;
// 				dist -= ss;
// 			}
// 		}
// 	}
//
// 	//if (isDebugRay && PRINT_DEBUG_MARCHING && !EUCLIDEAN)
// 	//	std::cout << "total distance traveled: " << totalDistanceTraveled << '\n';
//
// 	if (isDebugRay && LOG_MARCH_PATH)
// 	{
// 		std::cout << "\nMarch Path:\nPositions:\n{\n";
// 		for (const auto p : rayMarchPathPositions)
// 			std::cout << toStr(p) << ",\n";
//
// 		std::cout << "}\n\nDirections:\n{\n";
// 		for (const auto d : rayMarchPathDirections)
// 			std::cout << toStr(d) << ",\n";
// 		std::cout << "}\n";
// 	}
//
// 	if (closestHit)
// 		return *closestHit.get();
//
// 	return std::nullopt;
}

// trace a single ray and return the color
__device__ glm::vec3 trace_ray(
    glm::vec3 origin,
    glm::vec3 direction,
    Intersection* hitsBuffer_ray, // guaranteed to allow for storing num_bounces intersection
    int num_bounces,
    const Scene* scene
) {
    for (int i = 0; i < num_bounces; i++) {
        Intersection* hitsBuffer_bounce = hitsBuffer_ray + i; // if intersection happens, store it into this buffer

    }

    return glm::vec3{1.f, 0.f, 0.f};
}

// Render a single pixel
__global__ void render_pixel(
    const Scene* scene,
    const Camera* camera,
    int width,
    int height,
    int rays_per_pixel,
    int bounces_per_ray,
    glm::vec3* frameBuffer_device,
    Intersection* hitsBuffer_device
) {

    // TODO: these should be passed in as parameters
    float aspectRatio = width / height; // w : h
    // todo figure out why FOV seems "off"
    float fovComponent{tanf(camera->FOV / 2.f)};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    Intersection* hitsBuffer_pixel = hitsBuffer_device + (x + (y * width) * rays_per_pixel * bounces_per_ray);
    
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

        Intersection* hitsBuffer_ray = hitsBuffer_pixel + (i * bounces_per_ray);


        glm::vec3 color = trace_ray(start, dir, hitsBuffer_ray, bounces_per_ray, scene);

        final_color += color;
    }

    final_color /= RAYS_PER_PIXEL;

    { // writeback to framebuffer
        int frameBufferIndex = pixelCoord.x + (pixelCoord.y * width);
        frameBuffer_device[frameBufferIndex] = final_color;
    }
}

__host__ void render(const Scene* scene, const Camera* camera, Image* image) {
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
    cudaMalloc(&frameBuffer_Device, width * height * sizeof(glm::vec3));


    // allocate buffer to store intersections data
    Intersection* hitsBuffer_Device;
    cudaMalloc(&hitsBuffer_Device, width * height * RAYS_PER_PIXEL * MAX_NUM_BOUNCES * sizeof(Intersection)); // each ray (bounce) needs to store its hit

    render_pixel<<<gridDims, blockDims>>>(
        scene,
        camera,
        width,
        height,
        RAYS_PER_PIXEL,
        MAX_NUM_BOUNCES,
        frameBuffer_Device,
        hitsBuffer_Device
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
}

} // namespace RendererCUDA
