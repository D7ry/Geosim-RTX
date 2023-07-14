#pragma once

#include <optional>

#include <glm/vec3.hpp>

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
	const glm::vec3 position;
	const glm::vec3 surfaceNormal;
};

class Math
{
	static inline float MIN_T{ 0.01f  };
	static inline float MAX_T{ 1000.f };

public:

	// bad random number generation function which returns normalized double
	static double rng();

	static std::optional<RayIntersection> raySphereIntersection(
		Ray ray, 
		const glm::vec3& pos, 
		float r,
		float minT = Math::MIN_T,
		float maxT = Math::MAX_T
	);

	static std::optional<RayIntersection> rayTriangleIntersection(
		const Ray& ray,
		const glm::vec3 vertices[3],
		float minT = Math::MIN_T,
		float maxT = Math::MAX_T
	);

	static glm::vec3 getPoint(const Ray& r, float t);

	// returns surface normal of a sphere at a given point
	static glm::vec3 sphereNormal(const glm::vec3& origin, const glm::vec3& point);

	static glm::vec3 triangleNormal(const glm::vec3 const vertices[3]);

};