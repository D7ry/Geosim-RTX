#include "Renderer.h"

#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Scene.h"

#include "util/Math.h"

#include <iostream>

#include "Settings.h"

void Renderer::render(const Scene& scene, const Camera& camera, Image& image)
{
	aspectRatio = (float)image.width / image.height;	// w : h

	// todo figure out why FOV seems "off"
	float fovComponent{ tanf(camera.FOV / 2.f) };

	for (int y = 0; y < image.height; ++y)
		for (int x = 0; x < image.width; ++x)
		{
			if constexpr (!INTERACTIVE_MODE)
			{
				const unsigned index{ x + (y * image.width) };
				const unsigned numPixels{ image.width * image.height };

				const float completionPercent{ 100.f * index / numPixels };

				static float prevPercent{ 0.f };

				if (completionPercent > prevPercent + 10)
				{
					prevPercent = (int)completionPercent;
					std::cout << completionPercent << "%\n";
				}
			}

			const glm::vec2 ndc
			{
				(x + 0.5f) / image.width,
				(y + 0.5f) / image.height
			};

			glm::vec4 pixelColor{ 0.f };

			for (int i = 0; i < RAYS_PER_PIXEL; ++i)
			{
				glm::vec2 rayOffset;

				if constexpr (USE_RNG_FOR_AA)
					rayOffset = Math::randomVec2(i);
				else
					rayOffset = glm::vec2{ ((2.f * i) + 1.f) / (RAYS_PER_PIXEL * 2.f) };

				const glm::vec2 ndcAliased
				{
					(x + rayOffset.x) / image.width,
					(y + rayOffset.y) / image.height
				};

				// screen space
				const glm::vec2 coord{
					((2.f * ndcAliased.x) - 1.f) * fovComponent * aspectRatio,
					1.f - (2.f * ndcAliased.y) * fovComponent		// flip vertically so +y is up
				};

				// ray coords in world space
				glm::vec4 start{ camera.position, 1.f };
				glm::vec4 dir{ coord.x, coord.y, -1.f, 1.f };
				
				// transform ray to view space
				dir = dir * camera.viewMat;

				Ray ray{
					start,
					dir
				};

				pixelColor += traceRay(ray, scene);

				//pixelColor += perPixel(scene, camera, coord);
			}
			
			pixelColor /= RAYS_PER_PIXEL;

			image.setPixel(ndc, pixelColor);
		}
}

glm::vec4 Renderer::perPixel(const Scene& scene, const Camera& camera, const glm::vec2& coord)
{
	// find closest hit object
	std::unique_ptr<PrimitiveIntersection> closestHit;

	for (const Geometry& object : scene.geometry )
		for (const auto& primitivePtr : object.primitives)
		{
			const Primitive& primitive = *primitivePtr.get();

			/// todo: i think something is wrong with the positioning, need to check
			// create ray
			Ray ray{ 
				glm::vec3{0},
				glm::vec3{ coord.x, coord.y, -1.f } 
			};

			const auto hitCheck = primitive.checkRayIntersection(ray, object.position);

			if (hitCheck.has_value())
			{
				const PrimitiveIntersection& hit{ hitCheck.value() };

				// replace closest hit if it is null or closer
				if (!closestHit || hit.intersection.t < closestHit->intersection.t)
					closestHit = std::make_unique<PrimitiveIntersection>(hit);
			}
		}

	if (closestHit)
		return glm::vec4{ closestHit->intersection.surfaceNormal, 1.f};

	return glm::vec4{ 0.f, 0.f, 0.f, 1.f };
}

glm::vec4 Renderer::traceRay(Ray ray, const Scene& scene)
{
	glm::vec4 incomingLight{ 0.f };
	glm::vec4 rayColor{ 1.f };

	for (int i = 0; i <= MAX_NUM_BOUNCES; ++i)
	{
		auto potentialIntersection = getClosestIntersection(ray, scene);
	
		const bool rayHit{ potentialIntersection.has_value() };

		if (rayHit)
		{
			PrimitiveIntersection hit{ potentialIntersection.value() };

			glm::vec4 surfaceEmittedLight = hit.material.emissionColor * hit.material.emissionStrength;
			incomingLight += surfaceEmittedLight * rayColor;

			tickG++;
			
			rayColor *= hit.material.color;
			const glm::vec3 lambertDir = Math::randomDir(tickG, hit.intersection.surfaceNormal);
			const glm::vec3 reflectedDir = glm::reflect(ray.dir, hit.intersection.surfaceNormal);

			ray.dir = Math::lerp(hit.material.roughness, lambertDir, reflectedDir);
			ray.origin = hit.intersection.position;
		}
		else
		{
			glm::vec3 environmentLightDir{ 0,1,0 };

			glm::vec4 environmentLight{ 
				std::max(0.f, glm::dot(environmentLightDir, ray.dir)) * rayColor 
			};

			incomingLight += environmentLight;

			break;
		}
	}
	
	return incomingLight;
}

PotentialPrimitiveIntersection Renderer::getClosestIntersection(const Ray& ray, const Scene& scene)
{
	std::unique_ptr<PrimitiveIntersection> closestHit;

	for (const Geometry& object : scene.geometry)
		for (const auto& primitivePtr : object.primitives)
		{
			const Primitive& primitive = *primitivePtr.get();

			const auto hitCheck = primitive.checkRayIntersection(ray, object.position);

			if (hitCheck.has_value())
			{
				const PrimitiveIntersection& hit{ hitCheck.value() };

				// replace closest hit if it is null or closer
				if (!closestHit || hit.intersection.t < closestHit->intersection.t)
					closestHit = std::make_unique<PrimitiveIntersection>(hit);
			}
		}

	if (closestHit)
		return *closestHit.get();

	return std::nullopt;
}