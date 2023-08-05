#include "Math.h"

#include <limits>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "../Primitive.h"

Ray::Ray(const glm::vec3& origin, const glm::vec3& dir)
	:
	origin{ origin },
	dir{ glm::normalize(dir) }
{}

glm::vec3 Math::lerp(float t, const glm::vec3& a, const glm::vec3& b)
{
	t = std::clamp(t, 0.f, 1.f);

	return (t * b) + ((1 - t) * a);
}

float Math::rng(unsigned state)
{
	state *= (state + 340147) * (state + 1273128) * (state + 782243);

	return (float)state / std::numeric_limits<unsigned>::max();
}

glm::vec2 Math::randomVec2(unsigned state)
{
	return glm::vec2(rng(state<<0), rng(state+1<<1));
}

glm::vec3 Math::randomVec3(unsigned state)
{
	return glm::vec3(rng(state<<0), rng(state+1<<1), rng(state+2<<2));
}

glm::vec3 Math::randomHemisphereDir(unsigned state, const glm::vec3& dir)
{
	// take random direction in on a sphere (here my distribution isn't uniform :P)
	// todo: make uniform distribution
	const glm::vec3 randDirection{ 
		glm::normalize((randomVec3(state) * 2.f) - 1.f )
	};

	const bool inWrongHemisphere{ glm::dot(randDirection, dir) < 0 };

	// if the random direction is in the wrong hemisphere
	if (inWrongHemisphere)
		return -randDirection;	// flip it to the other hemisphere

	// otherwise its fine
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

	const float b = glm::dot(ray.dir, ray.origin);
	const float c = glm::dot(ray.origin, ray.origin) - pow(r, 2);

	const float descriminant = (b * b) - (c);

	if (descriminant < 0)
		return std::nullopt;
	
	// take smallest positive root for t
	const float t1 = (-b - sqrtf(descriminant));
	const float t2 = (-b + sqrtf(descriminant));
	const float t = (t1 > 0) ? t1 : t2;
	
	if (t < minT || t > maxT || t != t)	// t is out of interval or nan
		return std::nullopt;

	// inch ray forward a tiny bit to see whether it is really inside or outside
	const bool isInside{ glm::length(getPoint(ray, MIN_T)) < r };

	// reposition ray at original position
	ray.origin += pos;

	const glm::vec3 intersectionPoint{ getPoint(ray, t) };

	return RayIntersection{
		ray,
		t,
		intersectionPoint,
		sphereNormal(pos, intersectionPoint, isInside)
	};
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

glm::vec3 Math::SchlickR0(float ior)
{
	return glm::vec3{ SchlickR0(ior, 1.f) };
}

glm::vec3 Math::SchlickR0(float ior1, float ior2)
{
	return glm::vec3{
		powf((ior1 - ior2) / (ior1 + ior2), 2)
	};
}
glm::vec3 Math::Schlick(
	const glm::vec3& incident,
	const glm::vec3& normal,
	const glm::vec3& r0
)
{
	const float cosTheata = glm::max(0.f, glm::dot(incident, normal));

	const glm::vec3 one{ 1.f };

	return r0 + (one - r0) * powf(1 - cosTheata, 5);
}

glm::vec3 Math::BRDF(const Intersection& hit)
{
	const glm::vec3& V = -hit.incidentDir;	// omega_o (outgoing light)
	const glm::vec3& L = hit.outgoingDir;	// omega_i (negative incoming light)
	const glm::vec3& N = hit.normal;

	const glm::vec3& albedo = hit.material.albedo;
	const float& roughness = hit.material.roughness;

	const glm::vec3 r0 = hit.material.reflectionCoeff();
	const glm::vec3 Ks = Schlick(L, N, r0);
	const glm::vec3 Kd = glm::vec3(1.f) - Ks;

	const glm::vec3 diffuse = Lambert(albedo);
	const glm::vec3 specular = CookTorrance(V, L, N, roughness, TrowbridgeReitzGGX, SchlickGGX);

	return (Ks * specular) + (Kd * diffuse);
}

glm::vec3 Math::Lambert(const glm::vec3& albedo)
{
	return albedo / glm::pi<float>();
}

glm::vec3 Math::CookTorrance(
	const glm::vec3& w_o, 
	const glm::vec3& w_i, 
	const glm::vec3& n,
	float roughness,
	NormalDistributionFunction D, 
	GeometryShadingFunction G
)
{
	const float alpha = roughness * roughness;
	const glm::vec3 halfAngle = glm::normalize((w_i + w_o) / glm::length(w_i + w_o));

	const float numerator = D(n, halfAngle, alpha) * G(w_o, w_i, n, alpha);
	const float denominator = 4.f * glm::dot(w_i, n) * glm::dot(w_o, n);

	return glm::vec3(numerator / denominator);
}

float Math::TrowbridgeReitzGGX(
	const glm::vec3& n, 
	const glm::vec3& H, 
	float alpha
)
{
	const float alpha2 = alpha * alpha;

	const float theta2 = glm::pow(glm::dot(n, H), 2);

	const float denominator = glm::pi<float>() * glm::pow(theta2 * (alpha2 - 1) + 1, 2);

	return alpha2 / glm::max(0.001f,denominator);
}

float Math::SchlickGGX(
	const glm::vec3& w_o, 
	const glm::vec3& w_i, 
	const glm::vec3& n, 
	float alpha
)
{
	// Smith
	auto G = [](const glm::vec3& x, const glm::vec3& n, float k)
	{
		return glm::dot(n, x) / glm::max(0.0001f, (glm::dot(n, x) * (1.f - k)) + k);
	};

	const float k = alpha / 2.f;

	return G(w_i, n, k) * G(w_o, n, k);
}
