#include "Scene.h"

#include "Primitive.h"

// sphere
std::optional<PrimitiveIntersection> Sphere::checkRayIntersection(const Ray& r) const
{
    auto intersection = Math::raySphereIntersection(r, this->position, this->radius);

    if (!intersection.has_value())
        return std::nullopt;

    return PrimitiveIntersection{ material, intersection.value() };
}

// triangle
std::optional<PrimitiveIntersection> Triangle::checkRayIntersection(const Ray& r) const
{
    auto intersection = Math::rayTriangleIntersection(r, this->vertices);

    if (!intersection.has_value())
        return std::nullopt;

    return PrimitiveIntersection{ material, intersection.value() };
}
