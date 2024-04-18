#pragma once

#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "Primitive.h"

#include <vector>

struct Camera;
struct Image;
struct Ray;
struct Scene;

class Renderer
{
public:
	// outputs render to Image perameter
	void render(const Scene& scene, const Camera& camera, Image& image);

	bool accumulate{ false };
	int samplesPerPixel{ 0 };
	std::vector<glm::vec3> frameBuffer;

	void resetAccumulator();

private:
	// cast ray out into scene, see where it goes, and determin color off that
	glm::vec3 traceRay(Ray ray, const Scene& scene);

	// evaluates a path that light that did (or didnt) make it to the eye
	glm::vec3 evaluateLightPath(const Ray& primary, const std::vector<Intersection>& hits);

	PotentialIntersection getClosestIntersection(const Ray& ray, const Scene& scene);
	PotentialIntersection getClosestIntersectionMarch(const Ray& ray, const Scene& scene);
	std::pair<double, const Primitive&> getClosestPrimitive(const glm::vec4& p, const Scene& scene);
	double getClosestDistance(const glm::vec4& p, const Scene& scene);

	glm::vec3 computeNormal(const glm::vec4& p, const Scene& scene);

	glm::vec3 environmentalLight(const glm::vec3& dir);

	std::pair<glm::vec4, glm::vec4> march(
		const glm::vec4& pos, 
		const glm::vec4& dir, 
		float dist
	);

	void debugRayCast(const Ray& primary, std::vector<Intersection>& hits);
	void debugLightPath(const Ray& primary, std::vector<Intersection>& hits);

	std::vector<glm::vec4> rayMarchPathPositions;
	std::vector<glm::vec4> rayMarchPathDirections;

private:
	float aspectRatio{};
};