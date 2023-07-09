#pragma once

#include <vector>

#include "util/Color.h"

struct Image
{
	const unsigned width;
	const unsigned height;

	std::vector<Colorf> pixels;

	Image(unsigned width, unsigned height)
		:
		width{ width },
		height{ height },
		pixels{ width * height }
	{}
};