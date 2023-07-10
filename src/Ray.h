#pragma once

#include <glm/glm.hpp>

struct Ray
{
	glm::vec3 origin;
	glm::vec3 dir;

	Ray(const glm::vec3& origin, const glm::vec3& dir)
		:
		origin{ origin },
		dir{ dir }
	{}
};