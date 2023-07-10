#include "Renderer.h"

#include "Camera.h"
#include "Image.h"
#include "Scene.h"

#include <iostream>

void Renderer::render(const Scene& scene, const Camera& camera, Image& image)
{
	aspectRatio = (float)image.width / image.height;	// w : h

	for (int y = 0; y < image.height; ++y)
		for (int x = 0; x < image.width; ++x)
		{
			const glm::vec2 coord{
				((float)x / image.width)  * 2.f - 1,
				((float)y / image.height) * 2.f - 1
			};

			const glm::vec4 color{ perPixel(scene, camera, coord) };

			image.setPixel(coord, color);
		}
}

// calculates the color of a pixel within a scene at a given ND coordinate
glm::vec4 Renderer::perPixel(const Scene& scene, const Camera& camera, const glm::vec2& coord)
{
	if (std::hypotf(coord.x, coord.y) <= 1)
		return glm::vec4{ 1.f };

	return glm::vec4{ 0.f };
}