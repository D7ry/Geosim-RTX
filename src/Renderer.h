#pragma once

#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "Primitive.h"

struct Camera;
struct Image;
struct Ray;
struct Scene;

class Renderer
{
public:
	// outputs render to Image perameter
	void render(const Scene& scene, const Camera& camera, Image& image);

private:
	// calculation for color of a pixel at a given NDC
	glm::vec4 perPixel(const Scene& scene, const Camera& camera, const glm::vec2& coord);

	glm::vec4 traceRay(Ray ray, const Scene& scene);

	PotentialPrimitiveIntersection getClosestIntersection(const Ray& ray, const Scene& scene);

private:
	float aspectRatio{};
};