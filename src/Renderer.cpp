#include "Renderer.h"

#include "Camera.h"
#include "Image.h"
#include "Ray.h"
#include "Scene.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <iostream>

void Renderer::render(const Scene& scene, const Camera& camera, Image& image)
{
	aspectRatio = (float)image.width / image.height;	// w : h

	for (int y = 0; y < image.height; ++y)
		for (int x = 0; x < image.width; ++x)
		{
			const glm::vec2 ndc
			{
				(x + 0.5f) / image.width,
				(y + 0.5f) / image.height
			};

			float fovComponent{ atanf(camera.FOV / 2.f) };

			// screen space
			const glm::vec2 coord{
				((2.f * ndc.x) - 1.f) * fovComponent * aspectRatio,
				1.f - (2.f * ndc.y)	  * fovComponent		// flip vertically so +y is up
			};

			const glm::vec4 color{ perPixel(scene, camera, coord) };

			image.setPixel(ndc, color);
		}
}

// calculates the color of a pixel within a scene at a given ND coordinate
glm::vec4 Renderer::perPixel(const Scene& scene, const Camera& camera, const glm::vec2& coord)
{
	for (const Geometry& sphere : scene.geometry)
	{
		// create ray
		Ray ray{ 
			camera.position - sphere.position, 
			glm::vec3{coord.x, coord.y, -1.f} 
		};

		const float r{ 1 };		// todo, make based off sphere's scale attribute
		
		const float a = glm::dot(ray.dir, ray.dir);
		const float b = 2 * glm::dot(ray.dir, ray.origin);
		const float c = glm::dot(ray.origin, ray.origin) - (r*r);

		const float descriminant = (b * b) - (4 * a * c);
		
		if (descriminant > 0)
		{
			const float t = (-b - sqrtf(descriminant)) / 2.f;	// take smallest positive root

			if (t >= 0)
			{
				glm::vec3 intersectionPoint = (ray.dir * t) + ray.origin;

				glm::vec3 sphereNormal = intersectionPoint - ray.origin;

				return glm::vec4{ intersectionPoint, 1.f };
			}
		}
	}

	return glm::vec4{ 0.f };
}