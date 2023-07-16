#include "Scene.h"

#include "Primitive.h"

// sphere
std::optional<PrimitiveIntersection> Sphere::checkRayIntersection(
    const Ray& r,
    const glm::vec3& offset
) const
{
    Ray ray = r;
    ray.origin -= offset;
    auto intersection = Math::raySphereIntersection(ray, this->position, this->radius);

    if (!intersection.has_value())
        return std::nullopt;

    intersection.value().position += offset;

    return PrimitiveIntersection{ material, intersection.value() };
}

// triangle
std::optional<PrimitiveIntersection> Triangle::checkRayIntersection(
    const Ray& r,
    const glm::vec3& offset
) const
{
    auto intersection = Math::rayTriangleIntersection(r, this->vertices);

    if (!intersection.has_value())
        return std::nullopt;

    return PrimitiveIntersection{ material, intersection.value() };
}
