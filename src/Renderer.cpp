#include "Renderer.h"

#include "Camera.h"
#include "Image.h"
#include "Scene.h"

#include "util/Math.h"

#include <iostream>

#include "Settings.h"

void Renderer::render(const Scene& scene, const Camera& camera, Image& image)
{
	if (frameBuffer.size() != image.pixels.size())
	{
		frameBuffer.clear();
		frameBuffer.resize(image.pixels.size());
	}

	aspectRatio = (float)image.width / image.height;	// w : h

	// todo figure out why FOV seems "off"
	float fovComponent{ tanf(camera.FOV / 2.f) };

	for (int y = 0; y < image.height; ++y)
		for (int x = 0; x < image.width; ++x)
	{
			const int index = x + (y * image.width);

			// misc debug stuff
			const glm::uvec2 debugRay{ image.width / 2, image.height / 2 };
			//isDebugRay = (x == debugRay.x && y == debugRay.y);
			
			if constexpr (!INTERACTIVE_MODE)
			{
				const unsigned index{ x + (y * image.width) };
				const unsigned numPixels{ image.width * image.height };

				const float completionPercent{ 100.f * index / numPixels };

				// how many pixels per print
				constexpr unsigned printFreq{ 50 };

				static int prevPrintIndex{ 0 };

				if (index > prevPrintIndex + printFreq)
				{
					prevPrintIndex = index;
					std::cout << completionPercent << "%\n";
				}
			}

			// ray tracing stuff
			const glm::vec2 ndc
			{
				(x + 0.5f) / image.width,
				(y + 0.5f) / image.height
			};

			glm::vec3 color{ 0.f };

			for (int i = 0; i < RAYS_PER_PIXEL; ++i)
			{
				glm::vec2 rayOffset = Math::randomVec2(rngSeed + i);

				const glm::vec2 ndcAliased
				{
					(x + rayOffset.x) / image.width,
					(y + rayOffset.y) / image.height
				};

				// screen space
				glm::vec2 coord;

				if constexpr (ANTIALIAS)
				{
					coord = glm::vec2{
						((2.f * ndcAliased.x) - 1.f) * fovComponent * aspectRatio,
						1.f - (2.f * ndcAliased.y) * fovComponent		// flip vertically so +y is up
					};
				}
				else
				{
					coord = glm::vec2{
						((2.f * ndc.x) - 1.f) * fovComponent * aspectRatio,
						1.f - (2.f * ndc.y) * fovComponent		// flip vertically so +y is up
					};
				}

				// ray coords in world space
				glm::vec4 start{ camera.position, 1.f };
				glm::vec4 dir{ coord.x, coord.y, -1.f, 1.f };

				// transform ray to view space
				dir = dir * camera.viewMat;

				Ray ray{
					start,
					dir
				};

				if (!accumulate)
					resetAccumulator();

				isDebugRay = index == 3846;
				
				color += traceRay(ray, scene);

				frameBuffer[index] += color;
			}

			glm::vec3 pixelColor{ frameBuffer[index] };

			// average color
			pixelColor /= samplesPerPixel * RAYS_PER_PIXEL;

			// normalize color
			pixelColor.r = std::clamp(pixelColor.r, 0.f, 1.f);
			pixelColor.g = std::clamp(pixelColor.g, 0.f, 1.f);
			pixelColor.b = std::clamp(pixelColor.b, 0.f, 1.f);

			// debug visualization
			const bool shouldInvertColor{
				VISUALIZE_DEBUG_RAY  && (
				(x == debugRay.x + 1 && y == debugRay.y + 0)  ||	// left
				(x == debugRay.x - 1 && y == debugRay.y - 0)  ||	// right
				(x == debugRay.x + 0 && y == debugRay.y + 1)  ||	// top
				(x == debugRay.x - 0 && y == debugRay.y - 1))		// bottom
			};

			if (shouldInvertColor)
				pixelColor = glm::vec3{ 1 } - pixelColor;

			// actually setting pixel
			image.setPixel(ndc, pixelColor);
		}

	samplesPerPixel++;
}

void Renderer::resetAccumulator()
{
	std::fill(frameBuffer.begin(), frameBuffer.end(), glm::vec3{ 0 });
	samplesPerPixel = 1;
}

glm::vec3 Renderer::traceRay(Ray ray, const Scene& scene)
{
	std::vector<Intersection> hits;

	for (int i = 0; i <= MAX_NUM_BOUNCES; ++i)
	{
		auto potentialIntersection = getClosestIntersection(ray, scene);
	
		const bool rayHit{ potentialIntersection.has_value() };

		if (rayHit)
		{
			rngSeed++;

			const Intersection& hit{ potentialIntersection.value() };

			// record intersection
			hits.push_back(hit);

			// redirect ray at point of intersection
			ray = Ray{ hit.position, hit.outgoingDir };
		}
		else
			break;
	}

	if (PRINT_DEBUG && isDebugRay)
	{
		debugRayCast(ray, hits);
		debugLightPath(ray, hits);
	}
	const glm::vec3 finalColor{ evaluateLightPath(ray, hits) };

	return finalColor;
}

glm::vec3 Renderer::evaluateLightPath(const Ray& primary, const std::vector<Intersection>& hits)
{
	glm::vec3 incomingLight{ 0 };

	// if ray bounced off a surface and never hit anything after
	const bool reachedEnvironment{ hits.size() < MAX_NUM_BOUNCES };

	// if primary ray hits nothing, use that as environment bound vector
	const glm::vec3 environmentDir{ hits.empty() ? primary.dir : hits.back().outgoingDir };

	if (reachedEnvironment)
		incomingLight += environmentalLight(environmentDir);

	// reverse iterate from the start of a path of light
	for (int i = hits.size()-1; i >= 0; i--)
	{
		const Intersection& hit{ hits[i] };

		// light emitted from hit surface
		const glm::vec3 emittedLight{ 
			hit.material.emissionColor * hit.material.emissionStrength 
		};

		// cos(theta) term
		const float lightStrength{
			1//glm::max(0.f, glm::dot(hit.normal, -hit.incidentDir))
		};

		// basically the rendering equation
		//incomingLight = emittedLight + (2.f * Math::BRDF(hit) * incomingLight * lightStrength);
		incomingLight = emittedLight + (hit.material.albedo * incomingLight * lightStrength);
	}

	return incomingLight;
}

PotentialIntersection Renderer::getClosestIntersection(const Ray& ray, const Scene& scene)
{
	std::unique_ptr<Intersection> closestHit;

	for (const Geometry& object : scene.geometry)
		for (const auto& primitivePtr : object.primitives)
		{
			const Primitive& primitive = *primitivePtr.get();

			const auto hitCheck = primitive.checkRayIntersection(ray, object.position);

			if (hitCheck.has_value())
			{
				const Intersection& hit{ hitCheck.value() };

				// replace closest hit if it is null or closer
				if (!closestHit || hit.math.t < closestHit->math.t)
					closestHit = std::make_unique<Intersection>(hit);
			}
		}

	if (closestHit)
		return *closestHit.get();

	return std::nullopt;
}

glm::vec3 Renderer::environmentalLight(const glm::vec3& dir)
{
	const float dayTime{ globalTick / 128.f };

	//glm::vec3 lightDir{ sinf(dayTime), cosf(dayTime), 0 };
	glm::vec3 lightDir{ 0, 1, 0 };
	lightDir = glm::normalize(lightDir);

	const glm::vec3 noonColor{ 1 };
	const glm::vec3 sunsetColor{ 1,.6,.3 };

	const float interpolation = std::max(0.f, glm::dot(lightDir, glm::vec3{ 0,1,0 }));

	const glm::vec3 lightColor = Math::lerp(interpolation, sunsetColor, noonColor);

	glm::vec3 light{
		std::max(0.f, glm::dot(lightDir, dir)) * lightColor
	};

	return light;
}

void Renderer::debugRayCast(const Ray& primary, std::vector<Intersection>& hits)
{
	auto toStr = [](const glm::vec3& v)
	{
		// takes float and returns string to 3 decimals
		auto helper = [](float f)
		{
			std::string s = std::to_string(f);
			return s.substr(0, s.find(".") + 4);
		};

		return std::string{ "(" + helper(v.x) + ", " + helper(v.y) + ", " + helper(v.z) + ")" };
	};

	std::cout << "////////////////// RAY CAST START\n\n";

	std::cout << "A primary ray, ray 0 is shot out of the camera at "
		<< toStr(primary.origin) << " and looking at " << toStr(primary.dir) << "\n\n";

	int i = 0;
	for (; i < hits.size(); ++i)
	{
		const Intersection& hit{ hits[i] };

		std::cout << "ray " << i << " goes out and hits a material: " << &hit.material
			<< " at " << toStr(hit.position) << " (t=" << hit.math.t
			<< ")\n and bounces off in the direction of " << toStr(hit.outgoingDir) << "\n\n";
	}

	if (hits.size() == MAX_NUM_BOUNCES)
		std::cout << "With that we reach the max number of bounces...\n\n";
	else
		std::cout << "and ray " << i << " goes off forever...\n\n\n";
}

void Renderer::debugLightPath(const Ray& primary, std::vector<Intersection>& hits)
{
	auto toStr = [](const glm::vec3& v)
	{
		// takes float and returns string to 3 decimals
		auto helper = [](float f)
		{
			std::string s = std::to_string(f);
			return s.substr(0, s.find(".") + 4);
		};

		return std::string{ "(" + helper(v.x) + ", " + helper(v.y) + ", " + helper(v.z) + ")" };
	};

	std::cout << "////////////////// LIGHT PATH START\n\n";

	glm::vec3 incomingLight{ 0 };
	const bool reachedEnvironment{ hits.size() < MAX_NUM_BOUNCES };
	const glm::vec3 environmentDir{ hits.empty() ? primary.dir : hits.back().outgoingDir };

	if (reachedEnvironment)
	{
		incomingLight += environmentalLight(environmentDir);

		std::cout << "A ray of light comes from the environment at direction "
			<< toStr(-environmentDir) << " with color " << toStr(incomingLight) << '\n';
	}

	// reverse iterate from the start of a path of light
	for (int i = hits.size() - 1; i >= 0; i--)
	{
		const Intersection& hit{ hits[i] };

		std::cout << "It hits something!\n";

		// light emitted from hit surface
		const glm::vec3 emittedLight{
			hit.material.emissionColor * hit.material.emissionStrength
		};

		if (emittedLight != glm::vec3{ 0 })
			std::cout << "That something is also emitting light: " << toStr(emittedLight)
			<< "\ncombining that with the current light: " << toStr(incomingLight)
			<< "\nwe now have: " << toStr(emittedLight + incomingLight) << "\n\n";

		// cos(theta) term
		const float lightStrength{
			glm::max(0.f, glm::dot(hit.normal, -hit.incidentDir))
		};

		std::cout << "incoming light = " << toStr(emittedLight) << " + ("
			<< toStr(hit.material.albedo) << " * " << toStr(incomingLight) << " * " << lightStrength << ")\n";

		// basically the rendering equation
		incomingLight = emittedLight + (hit.material.albedo * incomingLight * lightStrength);

		std::cout << " == " << toStr(incomingLight) << "\n\n";
	}

	std::cout << "and that ray of light makes its way into your eye, you percieve the color " 
		<< toStr(incomingLight) << "\n\n";
}