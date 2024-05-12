#pragma once

#include <array>
#include <optional>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <functional>
#include <string>

#include "Ray.h"

struct Intersection;


class Math
{
	static inline float MIN_T{ 0.01f  };
	static inline float MAX_T{ 1000.f };

public:

	// linearly interpolates between 2 vec3's
	// at t=0 : a, at t=1 : b
	static glm::vec3 lerp(float t, const glm::vec3& a, const glm::vec3& b);

	// has a percentage probability of returning true
	static bool probability(float percentage);

	// bad random number generation function which returns normalized double [0,1]
	static float rng(unsigned state);

	// returns a vector where each component is between 0 and 1
	static glm::vec2 randomVec2(unsigned state);

	// returns a vector where each component is between 0 and 1
	static glm::vec3 randomVec3(unsigned state);

	// returns a random normalized direction in the hemisphere of dir
	static glm::vec3 randomHemisphereDir(unsigned state, const glm::vec3& dir);

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

	static std::optional<RayIntersection> rayPlaneIntersection(
		const Ray& ray,
		const glm::vec3& p,
		const glm::vec3& n,
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

	static glm::vec3 triangleNormal(const glm::vec3 vertices[3]);

	typedef std::array<glm::vec3, 3> Vertices;

	static Vertices transform(const Vertices& v, const glm::mat4& m);
	static glm::vec3 transform(const glm::vec3& v, const glm::mat4& m);

	static glm::vec3 Schlick(
		const glm::vec3& incident,
		const glm::vec3& normal,
		const glm::vec3& r0
	);

	// computes Schlick approximation's R0 term for dielectric materials (the dumb way)
	static glm::vec3 SchlickR0(float ior);

	// computes Schlick approximation's R0 term for dielectric materials
	static glm::vec3 SchlickR0(float ior1, float ior2);

	static glm::vec3 BRDF(const Intersection& hit);

	static glm::vec3 Lambert(const glm::vec3& albedo);

	typedef float (*NormalDistributionFunction)(
		const glm::vec3& normal, 
		const glm::vec3& halfAngle, 
		float roughness
	);

	typedef float (*GeometryShadingFunction)(
		const glm::vec3& outgoingLight,
		const glm::vec3& negIncomingLight,
		const glm::vec3& normal,
		float roughness
	);

	static glm::vec3 CookTorrance(
		const glm::vec3& outgoingLight, 
		const glm::vec3& negIncomingLight, 
		const glm::vec3& normal,
		float roughness,
		NormalDistributionFunction D,
		GeometryShadingFunction G
	);

	// Normal Distribution Function
	static float TrowbridgeReitzGGX(
		const glm::vec3& normal,
		const glm::vec3& halfAngle,
		float roughness
	);

	// Geometry Shading Function
	static float SchlickGGX(
		const glm::vec3& outgoingLight,
		const glm::vec3& negIncomingLight,
		const glm::vec3& normal,
		float roughness
	);

	static double euclideanSphereSDF(
		const glm::vec4& p, 
		float r,
		const glm::vec4& center = { 0,0,0,0 }
	);

	static double hyperbolicSphereSDF(
		const glm::vec4& p, 
		float r, 
		const glm::vec4& center = { 0,0,0,0 }
	);

	static glm::mat4 generateHyperbolicExponentialMap(const glm::vec3& displacement);

	static std::pair<glm::vec4, glm::vec4> geodesicFlowEuclidean(
		const glm::vec4& pos,
		const glm::vec4& dir,
		float t
	);

	static std::pair<glm::vec4, glm::vec4> geodesicFlowHyperbolic(
		const glm::vec4& pos,
		const glm::vec4& dir,
		float t
	);

	// Get point at distance t on the geodesic from pos in the direction dir
	static glm::vec4 hypGeoFlowPos(
		const glm::vec4& pos,
		const glm::vec4& dir,
		float t
	);

	// Get velocity/direction of point at distance t on 
	// the geodesic from pos in the direction dir
	static glm::vec4 hypGeoFlowDir(
		const glm::vec4& pos,
		const glm::vec4& dir,
		float t
	);

	// hyperbolic funcs
	static float cosh(float x);
	static float acosh(float x);
	static float sinh(float x);

	static float hypDot(const glm::vec4& u, const glm::vec4& v);
	static float hypNorm(const glm::vec4& v);

	static glm::vec4 hypNormalize(const glm::vec4& u);

	static float hypDistance(const glm::vec4& u, const glm::vec4& v);

	static glm::vec4 hypDirection(const glm::vec4& u, const glm::vec4& v);

	// Constructs a point on the hyperboloid from a direction and a hyperbolic distance.
	static glm::vec4 constructHyperboloidPoint(
		const glm::vec3& direction,
		float distance
	);

	static glm::vec4 correctH3Point(const glm::vec4 p);
	static glm::vec4 correctDirection(const glm::vec4& p, const glm::vec4& d);

	static bool isH3Point(const glm::vec4& p);
	static bool isH3Dir(const glm::vec4& p, const glm::vec4& dir);
	static void printH3Point(const std::string& s, const glm::vec4& v);
	static void printH3Dir(const std::string& s, const glm::vec4& p, const glm::vec4& d);
	static bool hyperbolicUnitTests();

	
	// Get point at distance t on the geodesic from pos in the direction dir
	static glm::vec4 sphGeoFlowPos(
		const glm::vec4& pos,
		const glm::vec4& dir,
		float t
	);

	// Get velocity/direction of point at distance t on 
	// the geodesic from pos in the direction dir
	static glm::vec4 sphGeoFlowDir(
		const glm::vec4& pos,
		const glm::vec4& dir,
		float t
	);

	static float sphDot(const glm::vec4& u, const glm::vec4& v);
	static float sphNorm(const glm::vec4& v);

	static glm::vec4 sphNormalize(const glm::vec4& u);

	static float sphhDistance(const glm::vec4& u, const glm::vec4& v);

	static glm::vec4 sphDirection(const glm::vec4& u, const glm::vec4& v);

	static glm::vec4 constructSpherePoint(
		const glm::vec3& direction,
		float distance
	);

	static glm::mat4 makeSphTranslation(const glm::vec4& p);

	static bool isInS3(const glm::vec4& v);
	static void printS3(const std::string& s, const glm::vec4& v);
	static bool sphereUnitTests();

	
	static bool withinError(double approx, double expected, double tolerance);
	static double getAbsoluteError(double approx, double expected);
	static double getRelativeError(double approx, double expected);

	
};
