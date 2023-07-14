#pragma once

#include <memory>
#include <vector>

#include <glm/vec3.hpp>

struct Primitive;

struct Geometry
{
	glm::vec3 position{ 0.f };		// world space
	float scale{ 1.f };
	//glm::mat4 rotation;
	std::vector<std::shared_ptr<Primitive>> primitives;

	template<typename PrimitiveType>
	void add(const PrimitiveType& p)
	{
		primitives.emplace_back(std::make_shared<PrimitiveType>(p));
	}
};

struct Scene
{
	std::vector<Geometry> geometry;

	void add(Geometry& object)
	{
		geometry.emplace_back(object);
	}
};