#pragma once

#include <glm/glm.hpp>

#include <vector>

struct Geometry
{
	glm::vec3 position{ 0, 0, 0 };
	glm::vec3 scale{ 1.f, 1.f, 1.f };
	// glm::mat4 rotation;
	// primatives[]
};

struct Scene
{
	std::vector<Geometry> geometry;
};