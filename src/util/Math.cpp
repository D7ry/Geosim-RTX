#include "Math.h"

#include <limits>

#include <glm/glm.hpp>

Ray::Ray(const glm::vec3& origin, const glm::vec3& dir)
	:
	origin{ origin },
	dir{ glm::normalize(dir) }
{}

double Math::rng(unsigned state)
{
	state *= (state + 340147) * (state + 1273128) * (state + 782243);

	return state / (double)std::numeric_limits<unsigned>::max();
}

std::optional<RayIntersection> Math::raySphereIntersection(
	Ray ray, 
	const glm::vec3& pos, 
	float r,
	float minT,
	float maxT
)
{
	/// todo unit test

	// orient sphere at origin
	ray.origin -= pos;

	const float a = glm::dot(ray.dir, ray.dir);
	const float b = 2 * glm::dot(ray.dir, ray.origin);
	const float c = glm::dot(ray.origin, ray.origin) - pow(r, 2);

	const float descriminant = (b * b) - (4 * a * c);

	if (descriminant < 0)
		return std::nullopt;
	
	// take smallest positive root for t
	const float t1 = (-b - sqrtf(descriminant)) / 2.f;
	const float t2 = (-b + sqrtf(descriminant)) / 2.f;
	const float t = (t1 > 0) ? t1 : t2;
	
	if (t < minT || t > maxT)	// t is out of interval
		return std::nullopt;

	// reposition ray at original position
	ray.origin += pos;

	const glm::vec3 intersectionPoint{ getPoint(ray, t) };

	const RayIntersection intersection{
		ray,
		t,
		intersectionPoint,
		sphereNormal(pos, intersectionPoint)
	};

	return intersection;
}

std::optional<RayIntersection> Math::rayTriangleIntersection(
	const Ray& ray, 
	const glm::vec3 vertices[3],
	float minT,
	float maxT
)
{
	// todo
	return std::optional<RayIntersection>();
}

glm::vec3 Math::getPoint(const Ray& r, float t)
{
	return glm::vec3{ r.origin + (r.dir * t) };
}

glm::vec3 Math::sphereNormal(const glm::vec3& origin, const glm::vec3& point)
{
	// todo unit test
	return normalize(point - origin);
}

glm::vec3 Math::triangleNormal(const glm::vec3 const vertices[3])
{
	// todo unit test
	return glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
}