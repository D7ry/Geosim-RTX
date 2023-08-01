#include "Scene.h"

#include "Primitive.h"

// sphere
PotentialPrimitiveIntersection Sphere::checkRayIntersection(
    const Ray& r,
    const glm::vec3& positionWorldSpace
) const
{
    Ray ray = r;
    ray.origin -= positionWorldSpace;

    auto intersection = Math::raySphereIntersection(ray, this->position, this->radius);

    if (!intersection.has_value())
        return std::nullopt;

    intersection.value().position += positionWorldSpace;

    return PrimitiveIntersection{ material, intersection.value() };
}

// triangle
PotentialPrimitiveIntersection Triangle::checkRayIntersection(
    const Ray& r,
    const glm::vec3& position
) const
{
    auto intersection = Math::rayTriangleIntersection(r, this->vertices);

    if (!intersection.has_value())
        return std::nullopt;

    return PrimitiveIntersection{ material, intersection.value() };
}
