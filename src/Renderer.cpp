#include "Renderer.h"

#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Scene.h"

#include "util/Math.h"

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
	std::unique_ptr<PrimitiveIntersection> closestHit;

	for (const Geometry& object : scene.geometry )
		for (const auto& primitivePtr : object.primitives)
		{
			const Primitive& primitive = *primitivePtr.get();

			/// todo: i think something is wrong with the positioning, need to check
			// create ray
			Ray ray{ 
				camera.position - object.position, 
				glm::vec3{ coord.x, coord.y, -1.f } 
			};
			
			const auto hitCheck = primitive.checkRayIntersection(ray);

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