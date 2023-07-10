#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

struct Camera
{
	glm::vec4 position{ 0, 0, 0, 1 };
	glm::vec4 forwardDir{ 0, 0, -1, 1 };
	glm::vec4 upDir{ 0, 1, 0, 1 };

	float FOV{ 0.78539816339 };	// pi/4, 45 deg
};