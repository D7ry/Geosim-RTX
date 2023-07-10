#pragma once

#include <glm/glm.hpp>

#include <vector>
#include <algorithm>

struct Image
{
	const unsigned width;
	const unsigned height;

	std::vector<glm::vec4> pixels;

	Image(unsigned width, unsigned height)
		:
		width{ width },
		height{ height },
		pixels{ width * height }
	{}

	// sets pixel in image using normalized device coordinates
	void setPixel(const glm::vec2& coord, const glm::vec4& pixel)
	{
		glm::uvec2 pixelCoord{
			((coord.x + 1) / 2) * width,	// x
			((coord.y + 1) / 2) * height	// y
		};

		int index = pixelCoord.x + (pixelCoord.y * height);

		pixels[index] = pixel;
	}
};