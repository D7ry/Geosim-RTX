#include "Scene.h"

#include "Primitive.h"

#include "Settings.h"

#include <cmath>

Intersection::Intersection(const Material& m, const RayIntersection& i)
    :
    material{ m },
    incidentDir{ i.ray.dir },
    outgoingDir{ },
    position{ i.position },
    normal{ i.normal },
    math{ i }
{
    outgoingDir = redirect(i.ray.dir);
}

glm::vec3 Intersection::redirect(const glm::vec3& i) const
{
    glm::vec3 outgoing{ 0.f };

    //if (reflected)
    {
        const glm::vec3 lambert{ Math::randomHemisphereDir(rngSeed, normal) };
        const glm::vec3 mirror{ glm::reflect(incidentDir, normal) };

        outgoing = Math::lerp(material.roughness, mirror, lambert);
    }
    //else // light diffused or refracted
    {
    }

    return outgoing;
}

bool Intersection::evaluateReflectivity()
{
    return true;
}

bool Intersection::shouldRefract()
{
    return false;
}

// sphere
PotentialIntersection Sphere::checkRayIntersection(
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

    return Intersection{ *material.get(), intersection.value()};
}

// triangle
PotentialIntersection Triangle::checkRayIntersection(
    const Ray& r,
    const glm::vec3& positionWorldSpace
) const
{
    Ray ray = r;
    ray.origin -= positionWorldSpace;

    auto intersection = Math::rayTriangleIntersection(ray, this->vertices);

    if (!intersection.has_value())
        return std::nullopt;

    intersection.value().position += positionWorldSpace;

    return Intersection{ *material.get(), intersection.value() };
}

// plane
PotentialIntersection Plane::checkRayIntersection(
    const Ray& r,
    const glm::vec3& positionWorldSpace
) const
{   
    Ray ray = r;
    ray.origin -= positionWorldSpace;

    auto intersection = Math::rayPlaneIntersection(ray, this->position, this->normal);

    if (!intersection.has_value())
        return std::nullopt;

    intersection.value().position += positionWorldSpace;

    return Intersection{ *material.get(), intersection.value() };
}

glm::vec3 Metal::reflectionCoeff() const
{
    return baseReflectivity;
}

float Metal::refractionProbability() const
{
    return 0.0f;
}

glm::vec3 Dielectric::reflectionCoeff() const
{
    return glm::vec3{
        Math::SchlickR0(ior)
    };
}

float Dielectric::refractionProbability() const
{
    return 1 - opacity;
}
