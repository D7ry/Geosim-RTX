#include "Scene.h"

#include "Primitive.h"

#include "Settings.h"

#include <cmath>

#include <string>
#include <iostream>

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

double Sphere::SDF(const glm::vec4& p, const glm::vec4& positionWorldSpace) const
{
    const glm::vec3 euclideanPosition{ 
        glm::vec3(positionWorldSpace) + this->position 
    };

    const glm::vec4 hyperbolicPosition{ Math::constructHyperboloidPoint(
        euclideanPosition,
        glm::length(euclideanPosition)
        ) };
    auto toStr = [](const glm::vec4& v)
    {
        // takes float and returns string to 3 decimals
        auto helper = [](float f)
        {
            std::string s = std::to_string(f);
            return s.substr(0, s.find(".") + 4);
        };

        return std::string{
            "(" + helper(v.x) + ", "
            + helper(v.y) + ", "
            + helper(v.z) + ", "
            + helper(v.w) + ")"
        };
    };

    // todo: figure out how this works and when to use
    //const glm::mat4 translation{ Math::generateHyperbolicExponentialMap(
    //    hyperbolicPosition
    //) };
    //const glm::vec4 d{ translation * hyperbolicPosition };


    ////// NOTICE: hyperbolic sphere SDF does not work, almost always returns NAN
    if (EUCLIDEAN)
        return Math::euclideanSphereSDF(
            p, // todo: is w supposed to be 0?
            this->radius,
            glm::vec4{euclideanPosition, 0}
        );
    else
    {
        const float dist = Math::hyperbolicSphereSDF(
            p, // todo: is w supposed to be 0?
            this->radius,
            hyperbolicPosition
        );

        if (isDebugRay && PRINT_DEBUG_MARCHING)
        {
            if (!Math::isInH3(p) )//|| !Math::isInH3(d))
                std::cout << "prim\n";

            //std::cout << "displacement: " << toStr(displacement) << '\n';
            std::cout << "p: " << toStr(p) << '\n';
            std::cout << "distance ; " << dist << '\n';
        }

        return dist;
    }
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

double Triangle::SDF(const glm::vec4& p, const glm::vec4& positionWorldSpace) const
{
    // todo
    return 10000.0;
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

double Plane::SDF(const glm::vec4& p, const glm::vec4& positionWorldSpace) const
{
    // todo
    return 10000.0;
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
