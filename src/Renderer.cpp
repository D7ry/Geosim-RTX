#include "Renderer.h"

#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Scene.h"

#include "util/Math.h"

#include <iostream>

void Renderer::render(const Scene& scene, const Camera& camera, Image& image)
{
	aspectRatio = (float)image.width / image.height;	// w : h

	// todo figure out why FOV seems "off"
	float fovComponent{ tanf(camera.FOV / 2.f) };

	for (int y = 0; y < image.height; ++y)
		for (int x = 0; x < image.width; ++x)
		{
			const glm::vec2 ndc
			{
				(x + 0.5f) / image.width,
				(y + 0.5f) / image.height
			};

			static constexpr int RAYS_PER_PIXEL{ 1 };

			glm::vec4 pixelColor{ 0.f };

			for (int i = 0; i < RAYS_PER_PIXEL; ++i)
			{
				const float rayOffset{
					((2.f * i) + 1.f) / (RAYS_PER_PIXEL * 2.f)
				};

				const glm::vec2 ndcAliased
				{
					(x + rayOffset) / image.width,
					(y + rayOffset) / image.height
				};

				// screen space
				const glm::vec2 coord{
					((2.f * ndcAliased.x) - 1.f) * fovComponent * aspectRatio,
					1.f - (2.f * ndcAliased.y) * fovComponent		// flip vertically so +y is up
				};

				Ray ray{
					camera.position,
					glm::vec3{ coord.x, coord.y, -1.f }
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
				camera.position, 
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

	static constexpr int MAX_NUM_BOUNCES{ 4 };

	for (int i = 0; i < MAX_NUM_BOUNCES; ++i)
	{
		auto potentialIntersection = getClosestIntersection(ray, scene);
	
		const bool rayHit{ potentialIntersection.has_value() };

		if (rayHit)
		{
			PrimitiveIntersection hit{ potentialIntersection.value() };

			glm::vec4 surfaceEmittedLight = hit.material.emissionColor * hit.material.emissionStrength;
			incomingLight += surfaceEmittedLight * rayColor;
			rayColor *= hit.material.color;

			ray.dir = glm::reflect(ray.dir, hit.intersection.surfaceNormal);
			ray.origin = hit.intersection.position;
		}
		else
		{
			incomingLight += glm::vec4{ 1.f, 0.5f, 0.5f, 1.f } * rayColor;
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