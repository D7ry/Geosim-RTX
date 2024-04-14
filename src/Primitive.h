#pragma once

#include "util/Math.h"

#include <memory>
#include <optional>

#include <glm/vec4.hpp>

struct Material
{
	glm::vec3 albedo{ 1.f };
	float roughness{ 1.f };	

	glm::vec3 emissionColor{ 1.f };
	float emissionStrength{ 0.f };

	// returns Ks
	virtual glm::vec3 reflectionCoeff() const = 0;
	virtual float refractionProbability() const = 0;
};

struct Metal : public Material
{
	glm::vec3 baseReflectivity{ 1.f };

	virtual glm::vec3 reflectionCoeff() const;
	virtual float refractionProbability() const;
};

struct Dielectric : public Material
{
	float opacity{ 1.f };
	float ior{ 1.5f };
	
	virtual glm::vec3 reflectionCoeff() const;
	virtual float refractionProbability() const;
};

struct Intersection
{
	const Material& material;
	const RayIntersection math;

	glm::vec3 incidentDir;	// angle at which ray hit surface
	glm::vec3 outgoingDir;	// angle at which ray left surface
	glm::vec3 position;
	glm::vec3 normal;

	enum class ReflectionType
	{
		Specular,
		Diffuse,
		Refract
	};

	//ReflectionType reflection;

	//bool reflected{ false };
	// Ks
	// Kd
	// Kt

	Intersection(const Material& m, const RayIntersection& i);


private:
	// gives direction light took after intersecting surface
	glm::vec3 redirect(const glm::vec3& i) const;

	bool evaluateReflectivity();
	bool shouldRefract();

};

typedef std::optional<Intersection> PotentialIntersection;

// primitives must implement their own geometric representation
// geometry is assumed to be in normalized local space
struct Primitive
{
	std::shared_ptr<Material> material{ nullptr };

	// evaluates ray intersects primitive at a given position in world space
	virtual PotentialIntersection checkRayIntersection(
		const Ray& r,
		const glm::vec3& positionWorldSpace
	) const = 0;

	virtual double SDF(
		const glm::vec3& p,
		const glm::vec3& positionWorldSpace
	) const = 0;
};

struct Triangle : Primitive
{
	glm::vec3 vertices[3];	// local space

	PotentialIntersection checkRayIntersection(
		const Ray& r,
		const glm::vec3& positionWorldSpace
	) const;

	double SDF(
		const glm::vec3& p,
		const glm::vec3& positionWorldSpace
	) const;
};

struct Sphere : Primitive
{
	glm::vec3 position{ 0.f };	// local space
	//glm::vec3 scale{ 1.f };	// todo
	float radius{ 1.f };

	PotentialIntersection checkRayIntersection(
		const Ray& r,
		const glm::vec3& positionWorldSpace
	) const;

	double SDF(
		const glm::vec3& p,
		const glm::vec3& positionWorldSpace
	) const;
};

// todo need to fix bug of bright pixels at intersection of plane and other primitives
struct Plane : Primitive
{
	glm::vec3 position{ 0.f };
	glm::vec3 normal{ 0, 0, 1 };

	PotentialIntersection checkRayIntersection(
		const Ray& r,
		const glm::vec3& positionWorldSpace
	) const;

	double SDF(
		const glm::vec3& p,
		const glm::vec3& positionWorldSpace
	) const;
};