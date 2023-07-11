#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

struct Camera
{
	glm::vec3 position{ 0.f };
	glm::vec3 forwardDir{ 0.f, 0.f, -1.f };	// looking into screen
	glm::vec3 upDir{ 0.f, 1.f, 0.f };

	float FOV{ glm::half_pi<float>() };	// reminder, FOV is not the angle from the horizontal, but twice that
};