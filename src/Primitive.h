#pragma once

#include "util/Math.h"

#include <optional>

#include <glm/vec4.hpp>

struct Material
{
	glm::vec4 color{ 1.f };
	float roughness{ 1.f };
	
	float opacity{ 1.f };
	float ior{ 1.f };

	glm::vec4 emissionColor{ 1.f };
	float emissionStrength{ 0.f };
};

struct PrimitiveIntersection
{
	const Material& material;
	RayIntersection intersection;
};

typedef std::optional<PrimitiveIntersection> PotentialPrimitiveIntersection;

// primitives must implement their own geometric representation
// geometry is assumed to be in normalized local space
struct Primitive
{
	Material material;

	// evaluates ray intersects primitive at a given position in world space
	virtual PotentialPrimitiveIntersection checkRayIntersection(
		const Ray& r,
		const glm::vec3& position
	) const = 0;
};

struct Triangle : Primitive
{
	glm::vec3 vertices[3];	// local space

	PotentialPrimitiveIntersection checkRayIntersection(
		const Ray& r,
		const glm::vec3& position
	) const;
};

struct Sphere : Primitive
{
	glm::vec3 position{ 0.f };	// local space
	//glm::vec3 scale{ 1.f };	// todo
	float radius{ 1.f };

	PotentialPrimitiveIntersection checkRayIntersection(
		const Ray& r,
		const glm::vec3& position
	) const;
};