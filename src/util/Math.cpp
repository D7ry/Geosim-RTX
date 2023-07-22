#include "Math.h"

#include <limits>
#include <algorithm>

#include <glm/glm.hpp>

#include <iostream>

Ray::Ray(const glm::vec3& origin, const glm::vec3& dir)
	:
	origin{ origin },
	dir{ glm::normalize(dir) }
{}

glm::vec3 Math::lerp(float t, const glm::vec3& a, const glm::vec3& b)
{
	t = std::clamp(t, 0.f, 1.f);

	return (t * a) + ((1 - t) * b);
}

double Math::rng(unsigned state)
{
	state *= (state + 340147) * (state + 1273128) * (state + 782243);

	return state / (double)std::numeric_limits<unsigned>::max();
}

glm::vec2 Math::randomVec2(unsigned state)
{
	return glm::vec2(rng(state), rng(state+1<<1));
}

glm::vec3 Math::randomVec3(unsigned state)
{
	return glm::vec3(rng(state), rng(state + 1 << 1), rng(state + 2 << 2));
}

glm::vec3 Math::randomDir(unsigned state, const glm::vec3& dir)
{
	glm::vec3 randDirection;

	bool isInHemisphere{ false };

	while (!isInHemisphere)
	{
		float x = rng(state + 0) * 2.0 - 1.0;
		float y = rng(state + 1) * 2.0 - 1.0;
		float z = rng(state + 2) * 2.0 - 1.0;

		randDirection = glm::vec3{ x,y,z };
		randDirection = glm::normalize(randDirection);

		isInHemisphere = glm::dot(randDirection, dir) >= 0;
		state += 3;
	}
	
	return randDirection;
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

	const bool isInside{ glm::length(ray.origin) < r };

	// reposition ray at original position
	ray.origin += pos;

	const glm::vec3 intersectionPoint{ getPoint(ray, t) };
	const RayIntersection intersection{
		ray,
		t,
		intersectionPoint,
		sphereNormal(pos, intersectionPoint, isInside)
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

glm::vec3 Math::sphereNormal(
	const glm::vec3& origin,
	const glm::vec3& point,
	bool isInside
)
{
	// todo unit test
	const glm::vec3 normal{ glm::normalize(point - origin) };

	if (isInside)
		return -normal;
	else
		return normal;
}

glm::vec3 Math::triangleNormal(const glm::vec3 const vertices[3])
{
	// todo unit test
	return glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
}

float Math::SchlickRefractionApprox(
	const glm::vec3& incident,
	const glm::vec3& normal, 
	float ior1, 
	float ior2
)
{
	const float r0 = pow((ior1 - ior2) / (ior1 + ior2), 2);
	const float cosTheata = glm::dot(-incident, normal);

	return r0 + (1 - r0) * pow(1 - cosTheata, 5);
}
