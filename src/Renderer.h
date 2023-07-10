#pragma once

#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

struct Scene;
struct Camera;
struct Image;

class Renderer
{
public:
	void render(const Scene& scene, const Camera& camera, Image& image);

private:
	glm::vec4 perPixel(const Scene& scene, const Camera& camera, const glm::vec2& coord);

private:
	float aspectRatio{};
};