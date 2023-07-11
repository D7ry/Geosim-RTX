#pragma once

#include <glm/glm.hpp>

#include <vector>

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

	void setPixel(const glm::vec2& ndc, const glm::vec4& pixelData)
	{
		glm::uvec2 pixelCoord{
			ndc.x * width,
			ndc.y * height
		};

		int index = pixelCoord.x + (pixelCoord.y * width);

		pixels[index] = pixelData;
	}
};