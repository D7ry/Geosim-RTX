#pragma once

#include <array>
#include <optional>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

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
	const glm::vec3 surfaceNormal;
};

class Math
{
	static inline float MIN_T{ 0.01f  };
	static inline float MAX_T{ 1000.f };

public:

	// linearly interpolates between 2 vec3's
	// at t=0 : a, at t=1 : b
	static glm::vec3 lerp(float t, const glm::vec3& a, const glm::vec3& b);

	// bad random number generation function which returns normalized double [0,1]
	static double rng(unsigned state);

	static glm::vec2 randomVec2(unsigned state);
	static glm::vec3 randomVec3(unsigned state);

	// returns a random normalized direction in a hemisphere
	static glm::vec3 randomDir(unsigned state, const glm::vec3& dir);

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
	static glm::vec3 sphereNormal(
		const glm::vec3& origin, 
		const glm::vec3& point, 
		bool isInside = false
	);

	static glm::vec3 triangleNormal(const glm::vec3 const vertices[3]);

	typedef std::array<glm::vec3, 3> Vertices;

	static Vertices transform(const Vertices& v, const glm::mat4& m);
	static glm::vec3 transform(const glm::vec3& v, const glm::mat4& m);

	static float SchlickRefractionApprox(
		const glm::vec3& incident,
		const glm::vec3& normal,
		float ior1, 
		float ior2
	);

	static glm::vec3 refract(
		const glm::vec3& incident, 
		const glm::vec3& normal, 
		float ior1, 
		float ior2
	);

};