#pragma once

#include "util/Math.h"

#include <optional>

#include <glm/vec4.hpp>

struct Material
{
	glm::vec4 color{ 1.f };
	float roughness{ 1.f };
	//float opacity

	glm::vec4 emissionColor{ 1.f };
	float emissionStrength{ 0.f };
};

struct PrimitiveIntersection
{
	const Material& material;
	const RayIntersection intersection;
};

typedef std::optional<PrimitiveIntersection> PotentialPrimitiveIntersection;

// primitives must implement their own geometric representation
// geometry is assumed to be in normalized local space
struct Primitive
{
	Material material;

	virtual PotentialPrimitiveIntersection checkRayIntersection(const Ray& r) const = 0;
};

struct Triangle : Primitive
{
	glm::vec3 vertices[3];	// local space

	PotentialPrimitiveIntersection checkRayIntersection(const Ray& r) const;
};

struct Sphere : Primitive
{
	glm::vec3 position{ 0.f };	// local space
	float radius{ 1.f };

	PotentialPrimitiveIntersection checkRayIntersection(const Ray& r) const;
};