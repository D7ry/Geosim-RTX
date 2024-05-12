#pragma once

#include <glm/glm.hpp>
struct Intersection;

struct Ray
{
	glm::vec3 origin;
	glm::vec3 dir;

	Ray(const glm::vec3& origin, const glm::vec3& dir);
};

struct RayIntersection
{
	const Ray ray;
	const float t;
	glm::vec3 position;
	const glm::vec3 normal;
};
