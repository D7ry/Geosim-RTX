#include "Math.h"

#include <limits>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "../Primitive.h"

#include "../Settings.h"

#include <iostream>

#include <string>


Ray::Ray(const glm::vec3& origin, const glm::vec3& dir)
	:
	origin{ origin },
	dir{ glm::normalize(dir) }
{}

glm::vec3 Math::lerp(float t, const glm::vec3& a, const glm::vec3& b)
{
	t = std::clamp(t, 0.f, 1.f);

	return (t * b) + ((1 - t) * a);
}

bool Math::probability(float percentage)
{
	return rng(rngSeed) > percentage;
}

float Math::rng(unsigned state)
{
	state *= (state + 340147) * (state + 1273128) * (state + 782243);

	return (float)state / std::numeric_limits<unsigned>::max();
}

glm::vec2 Math::randomVec2(unsigned state)
{
	return glm::vec2(rng(state<<0), rng(state+1<<1));
}

glm::vec3 Math::randomVec3(unsigned state)
{
	return glm::vec3(rng(state<<0), rng(state+1<<1), rng(state+2<<2));
}

glm::vec3 Math::randomHemisphereDir(unsigned state, const glm::vec3& dir)
{
	// take random direction in on a sphere (here my distribution isn't uniform :P)
	// todo: make uniform distribution
	const glm::vec3 randDirection{ 
		glm::normalize((randomVec3(state) * 2.f) - 1.f )
	};

	const bool inWrongHemisphere{ glm::dot(randDirection, dir) < 0 };

	// if the random direction is in the wrong hemisphere
	if (inWrongHemisphere)
		return -randDirection;	// flip it to the other hemisphere

	// otherwise its fine
	return randDirection;
}

std::optional<RayIntersection> Math::raySphereIntersection(
	Ray ray, 
	const glm::vec3& pos,
	float r,
	float minT,
	float maxT
)
{
	/// todo unit test

	// orient sphere at origin
	ray.origin -= pos;

	const float b = glm::dot(ray.dir, ray.origin);
	const float c = glm::dot(ray.origin, ray.origin) - pow(r, 2);

	const float descriminant = (b * b) - (c);

	if (descriminant < 0)
		return std::nullopt;
	
	// take smallest positive root for t
	const float t1 = (-b - sqrtf(descriminant));
	const float t2 = (-b + sqrtf(descriminant));
	const float t = (t1 > 0) ? t1 : t2;
	
	if (t < minT || t > maxT || t != t)	// t is out of interval or nan
		return std::nullopt;

	// inch ray forward a tiny bit to see whether it is really inside or outside
	const bool isInside{ glm::length(getPoint(ray, MIN_T)) < r };

	// reposition ray at original position
	ray.origin += pos;

	const glm::vec3 intersectionPoint{ getPoint(ray, t) };

	return RayIntersection{
		ray,
		t,
		intersectionPoint,
		sphereNormal(pos, intersectionPoint, isInside)
	};
}

std::optional<RayIntersection> Math::rayTriangleIntersection(
	const Ray& ray, 
	const glm::vec3 vertices[3],
	float minT,
	float maxT
)
{
	const glm::vec3& a = vertices[0];
	const glm::vec3& b = vertices[1];
	const glm::vec3& c = vertices[2];

	const glm::vec3 ab = b - a;
	const glm::vec3 ac = c - a;

	const glm::vec3 triCross = glm::cross(ac, ab);
	const glm::vec3 normal = glm::normalize(triCross);
	const glm::vec3& pos = a;

	auto potentialIntersection = rayPlaneIntersection(ray, pos, normal);

	if (!potentialIntersection.has_value())
		return std::nullopt;

	const RayIntersection& intersection = potentialIntersection.value();
	const glm::vec3& point = intersection.position;

	const float area = glm::length(triCross) / 2.f;
	
	const glm::vec3 ap = point - a;
	const glm::vec3 bp = point - b;
	const glm::vec3 cp = point - c;

	const float uArea = glm::length(glm::cross(bp,cp)) / 2.f;
	const float vArea = glm::length(glm::cross(ap,cp)) / 2.f;
	const float wArea = glm::length(glm::cross(ap,bp)) / 2.f;

	// barycentric coordinates are u, v, w for a, b, c, respectivley
	const float u = uArea / area;
	const float v = vArea / area;
	const float w = wArea / area;

	const float sum = u + v + w;

	const bool isInBounds{
		(0 <= u && u <= 1) &&
		(0 <= v && v <= 1) &&
		(0 <= w && w <= 1) &&
		sum < 1.00001 && sum > 0.99999
	};

	if (!isInBounds)
		return std::nullopt;

	return intersection;
}

std::optional<RayIntersection> Math::rayPlaneIntersection(
	const Ray& ray, 
	const glm::vec3& p, 
	const glm::vec3& n, 
	float minT, 
	float maxT
)
{
	// make sure input is valid
	if constexpr (DEBUG)
		assert(glm::length(n) == 1.f);


	const float d = glm::dot(n, p);
	const float t = (d - glm::dot(n, ray.origin)) / glm::dot(n, ray.dir);

	if (t < minT || t > maxT || t != t)	// t is out of interval or nan
		return std::nullopt;

	glm::vec3 normal = -n;

	const bool intersectionBehind{ glm::dot(ray.dir, n) < 0 };
	
	if (intersectionBehind)
		normal = -normal;
	
	return RayIntersection(
		ray, 
		t, 
		getPoint(ray, t), 
		normal
	);
}

glm::vec3 Math::getPoint(const Ray& r, float t)
{
	return glm::vec3{ r.origin + (r.dir * t) };
}

glm::vec3 Math::sphereNormal(
	const glm::vec3& origin,
	const glm::vec3& point,
	bool isInside
)
{
	// todo unit test
	const glm::vec3 normal{ glm::normalize(point - origin) };

	if (isInside)
		return -normal;
	else
		return normal;
}

glm::vec3 Math::triangleNormal(const glm::vec3 const vertices[3])
{
	// todo unit test
	return glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
}

glm::vec3 Math::SchlickR0(float ior)
{
	return glm::vec3{ SchlickR0(ior, 1.f) };
}

glm::vec3 Math::SchlickR0(float ior1, float ior2)
{
	return glm::vec3{
		powf((ior1 - ior2) / (ior1 + ior2), 2)
	};
}
glm::vec3 Math::Schlick(
	const glm::vec3& incident,
	const glm::vec3& normal,
	const glm::vec3& r0
)
{
	const float cosTheata = glm::max(0.f, glm::dot(incident, normal));

	const glm::vec3 one{ 1.f };

	return r0 + (one - r0) * powf(1 - cosTheata, 5);
}

glm::vec3 Math::BRDF(const Intersection& hit)
{
	const glm::vec3& V = -hit.incidentDir;	// omega_o (outgoing light)
	const glm::vec3& L = hit.outgoingDir;	// omega_i (negative incoming light)
	const glm::vec3& N = hit.normal;

	const glm::vec3& albedo = hit.material.albedo;
	const float& roughness = hit.material.roughness;

	const glm::vec3 r0 = hit.material.reflectionCoeff();
	const glm::vec3 Ks = Schlick(L, N, r0);
	const glm::vec3 Kd = glm::vec3(1.f) - Ks;

	const glm::vec3 diffuse = Lambert(albedo);
	const glm::vec3 specular = CookTorrance(V, L, N, roughness, TrowbridgeReitzGGX, SchlickGGX);

	return (Ks * specular) + (Kd * diffuse);
}

glm::vec3 Math::Lambert(const glm::vec3& albedo)
{
	return albedo / glm::pi<float>();
}

glm::vec3 Math::CookTorrance(
	const glm::vec3& w_o, 
	const glm::vec3& w_i, 
	const glm::vec3& n,
	float roughness,
	NormalDistributionFunction D, 
	GeometryShadingFunction G
)
{
	const float alpha = roughness * roughness;
	const glm::vec3 halfAngle = glm::normalize((w_i + w_o) / glm::length(w_i + w_o));

	const float numerator = D(n, halfAngle, alpha) * G(w_o, w_i, n, alpha);
	const float denominator = 4.f * glm::dot(w_i, n) * glm::dot(w_o, n);

	return glm::vec3(numerator / denominator);
}

float Math::TrowbridgeReitzGGX(
	const glm::vec3& n, 
	const glm::vec3& H, 
	float alpha
)
{
	const float alpha2 = alpha * alpha;

	const float theta2 = glm::pow(glm::dot(n, H), 2);

	const float denominator = glm::pi<float>() * glm::pow(theta2 * (alpha2 - 1) + 1, 2);

	return alpha2 / glm::max(0.001f,denominator);
}

float Math::SchlickGGX(
	const glm::vec3& w_o, 
	const glm::vec3& w_i, 
	const glm::vec3& n, 
	float alpha
)
{
	// Smith
	auto G = [](const glm::vec3& x, const glm::vec3& n, float k)
	{
		return glm::dot(n, x) / glm::max(0.0001f, (glm::dot(n, x) * (1.f - k)) + k);
	};

	const float k = alpha / 2.f;

	return G(w_i, n, k) * G(w_o, n, k);
}

double Math::euclideanSphereSDF(
	const glm::vec4& p, 
	float r,
	const glm::vec4& center
)
{
	return glm::length(glm::vec3(p - center)) - r;
}

double Math::hyperbolicSphereSDF(
	const glm::vec4& p, 
	float r,
	const glm::vec4& center
)
{
	return hypDistance(p, center) - r;
}

glm::mat4 Math::generateHyperbolicExponentialMap(const glm::vec3& dr)
{
	const glm::mat4 M = {
		0,	  0,	0,	  dr.x,
		0,	  0,	0,	  dr.y,
		0,	  0,	0,	  dr.z,
		dr.x, dr.y, dr.z, 0
	};

	const float len{ glm::length(dr) };
	static constexpr glm::mat4 I{ 1 };

	const glm::mat4 exp{
		I + 
		(glm::sinh(len) / len) * M + 
		((glm::cosh(len) - 1.f) / (len * len)) * (M * M)
	};
	
	return exp;
}

std::pair<glm::vec4, glm::vec4> Math::geodesicFlowEuclidean(
	const glm::vec4& pos, 
	const glm::vec4& dir, 
	float t
)
{
	return { pos + t*dir, dir };
}

std::pair<glm::vec4, glm::vec4> Math::geodesicFlowHyperbolic(
	const glm::vec4& pos, 
	const glm::vec4& dir, 
	float t
)
{
	return { hypGeoFlowPos(pos, dir, t), hypGeoFlowDir(pos, dir, t) };
}

glm::vec4 Math::hypGeoFlowPos(const glm::vec4& pos, const glm::vec4& dir, float t)
{
	return { cosh(t)*pos + sinh(t)*dir };
}

glm::vec4 Math::hypGeoFlowDir(const glm::vec4& pos, const glm::vec4& dir, float t)
{
	return { sinh(t)*pos + cosh(t)*dir };
}

float Math::cosh(float x)
{
	//float eX = exp(x);
	//return (0.5 * (eX + 1.0 / eX));
	return glm::cosh(x);
}

float Math::acosh(float x)
{
	//return log(x + sqrt(x*x-1.0));
	return glm::acosh(x);
}

float Math::sinh(float x)
{
	//float eX = exp(x);
	//return (0.5 * (eX - 1.0 / eX));
	return glm::sinh(x);
}

float Math::hypDot(const glm::vec4& u, const glm::vec4& v)
{
	return (u.x * v.x) + (u.y * v.y) + (u.z * v.z) - (u.w * v.w); // Lorentz Dot
}

float Math::hypNorm(const glm::vec4& v)
{
	return std::sqrt(std::abs(hypDot(v, v)));
}

glm::vec4 Math::hypNormalize(const glm::vec4& u)
{
	return u / hypNorm(u);
}

float Math::hypDistance(const glm::vec4& u, const glm::vec4& v)
{
	const float bUV = -hypDot(u, v);
	return acosh(bUV);
}

glm::vec4 Math::hypDirection(const glm::vec4& u, const glm::vec4& v)
{
	const glm::vec4 w = v + hypDot(u, v) * u;
	return hypNormalize(w);
}

glm::vec4 Math::constructHyperboloidPoint(const glm::vec3& direction, float distance)
{
	const float w{ cosh(distance) };
	const float magSquared = w * w - 1;
	const glm::vec3 d{ std::sqrtf(magSquared) * glm::normalize(direction) };
	return glm::vec4{ d, w };
}

bool Math::isH3Point(const glm::vec4& v)
{
	static constexpr float EPS{ 0.001 };

	const bool positiveW{ v.w > 0 };
	const bool constCurvature{ withinError(hypDot(v,v), -1, EPS) };
	
	return positiveW && constCurvature;
}

bool Math::isH3Dir(const glm::vec4& p, const glm::vec4& dir)
{
	const bool isDirection{ withinError(hypDot(dir, dir), 1, 0.001) };
	const bool isTangentVector{ withinError(hypDot(dir, p), 0, 0.001) };
	return isDirection && isTangentVector;
}

void Math::printH3Point(const std::string& s, const glm::vec4& v)
{
	auto toStr = [](const glm::vec4& v)
	{
		// takes float and returns string to 3 decimals
		auto helper = [](float f)
		{
			std::string s = std::to_string(f);
			return s.substr(0, s.find(".") + 10);
		};

		return std::string{
			"(" + helper(v.x) + ", "
			+ helper(v.y) + ", "
			+ helper(v.z) + ", "
			+ helper(v.w) + ")"
		};
	};

	std::cout << s <<" = " << toStr(v)
		<< "\nnormalize(" << s << ") = " << toStr(hypNormalize(v))
		<< "\n<" << s << "," << s << "> = " << hypDot(v, v)
		<< "\nnorm(" << s << ") = " << hypNorm(v)
		<< "\nisH3Point(" << s << ") = " << (isH3Point(v) ? "true" : "false")
		<< "\n\n";
}

void Math::printH3Dir(const std::string& s, const glm::vec4& p, const glm::vec4& d)
{
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

	std::cout << s << " = " << toStr(d)
		<< '\n' << s << "'s position = " << toStr(p)
		<< "\n<" << s << "," << s << "> = " << hypDot(d, d)
		<< "\n<" << "pos" << "," << s << "> = " << hypDot(p, d)
		<< "\nnorm(" << s << ") = " << hypNorm(d)
		<< "\nisH3Dir(" << "pos" << ", " << s << ") = " << (isH3Dir(p, d) ? "true" : "false")
		<< "\n\n";
}

bool Math::hyperbolicUnitTests()
{
	// random direction vector scaled to random length
	auto randH3Point = [](int s) -> glm::vec4 
	{
		const glm::vec3 randDir{ rng(s), rng(s + 1), rng(s + 2) };
		const float randScalar{ ((rng(s+3) - 0.5f)) * 10 };	// scaling blows up quick, keep it below ~10
		
		return constructHyperboloidPoint(randDir, randScalar);
	};


	// THIS PROBABLY DOESNT WORK
	auto randH3Dir = [](const glm::vec4& pos, int s) -> glm::vec4
	{
		glm::vec3 randDir{ rng(s), rng(s + 1), rng(s + 2) };

		// Make the direction orthogonal to the position
		glm::vec3 projDir = randDir - glm::dot(glm::vec3(pos), randDir) / pos.w * glm::vec3(pos);

		return glm::vec4(glm::normalize(projDir), 0.0f);

		// Generate a random 3D unit vector
		//const glm::vec3 randDir = glm::normalize(glm::vec3{ rng(s), rng(s + 1), rng(s + 2) });

		// Compute the Lorentz orthonormal basis vectors at the given point
		const glm::vec4 e1{ pos.w, 0.0f, 0.0f, pos.x };
		const glm::vec4 e2{ 0.0f, pos.w, 0.0f, pos.y };
		const glm::vec4 e3{ 0.0f, 0.0f, pos.w, pos.z };

		// Express the random direction in the Lorentz orthonormal basis
		const glm::vec4 tangentVec = randDir.x * e1 + randDir.y * e2 + randDir.z * e3;

		// Normalize the tangent vector with respect to the Lorentz inner product
		const glm::vec4 dir = hypNormalize(tangentVec);

		// Ensure the direction vector is orthogonal to the position vector
		const glm::vec4 orthogonalDir = dir - hypDot(pos, dir) * pos;

		return hypNormalize(orthogonalDir);
	};

	// verify that constructHyperboloidPoint maps E3 to H3
	for (int i = 0; i < 1000; ++i)
	{
		const glm::vec4 p{ randH3Point(i) };

		const bool inH3{ isH3Point(p) };

		printH3Point("v", p);


		if (!inH3)
		{
			std::cout << "failed constructHyperboloidPoint\n";
			printH3Point("v", p);
			return false;
		}
	}

	// verify that hypNormalize maps points in H3 to points in H3
	for (int i = 0; i < 1000; ++i)
	{
		const glm::vec4 v{ randH3Point(i) };
		const glm::vec4 vHypNormalized{ hypNormalize(v) };

		const bool inH3{ isH3Point(vHypNormalized) };
		const bool normalized{ withinError(hypNorm(vHypNormalized), 1, 0.001) };
		
		if (!(inH3 && normalized))
		{
			std::cout << "failed hypNormalize\n";
			printH3Point("n", vHypNormalized);
			return false;
		}
	}

	// verify that taking a point in the hyperboloid model of H3
	// and following flow of its geodesic gives a point still in H3
	for (int i = 0; i < 1000; ++i)
	{
		const glm::vec4 pos{ randH3Point(i) };
		const glm::vec4 dir{ randH3Dir(pos, i+1) };
		const float t{ rng(i + 2) * 10 };	// [0,10]
	
		{
			const glm::vec4 ORIGIN{ 0,0,0,1 };
			const glm::vec4 nextPos{ hypGeoFlowPos(ORIGIN, dir, 1) };
			const glm::vec4 nextDir{ hypGeoFlowDir(ORIGIN, dir, 1) };

			const bool sane{ isH3Point(nextPos) && isH3Dir(nextPos, nextDir) };

			{
				std::cout << "Origin flow 1 test: \n\n";

				printH3Point("p_0", pos);
				printH3Dir("d_0", pos, dir);
				printH3Point("p_n", nextPos);
				printH3Dir("d_n", nextPos, nextDir);
			}
		}

		{
			const glm::vec4 ORIGIN{ 0,0,0,1 };
			const glm::vec4 nextPos{ hypGeoFlowPos(ORIGIN, dir, 2) };
			const glm::vec4 nextDir{ hypGeoFlowDir(ORIGIN, dir, 2) };

			const bool sane{ isH3Point(nextPos) && isH3Dir(nextPos, nextDir) };

			{
				std::cout << "Origin flow 2 test: \n\n";
				
				printH3Point("p_0", pos);
				printH3Dir("d_0", pos, dir);
				printH3Point("p_n", nextPos);
				printH3Dir("d_n", nextPos, nextDir);
			}
		}

		{
			const glm::vec4 nextPos{ hypGeoFlowPos(pos, dir, t) };
			const glm::vec4 nextDir{ hypGeoFlowDir(pos, dir, t) };

			const bool nextPosInH3{ isH3Point(nextPos) };
			const bool nextDirInH3{ isH3Dir(nextPos, nextDir) };
			const bool nextDirNormalized{ withinError(hypNorm(nextDir), 1, 0.01) };
			
			if (!(nextPosInH3 && nextDirInH3 && nextDirNormalized))
			{
				std::cout << "failed geodesic flow\n";
				std::cout << "flowed out distance t = " << t << '\n';

				printH3Point("pos", pos);
				printH3Dir("dir", pos, dir);
				printH3Point("flowed pos", nextPos);
				printH3Dir("flowed dir", nextPos, nextDir);

				return false;
			}
		}
	}

	
	// verify that generateHyperbolicExponentialMap (should really rename)
	// translates points from H3 to H3
	// that it actually takes a vector from E3 and gives 
	// matrix/linear transform which maps H3 to H3
	{
		// todo
	}

	return true;
}

glm::vec4 Math::sphGeoFlowPos(const glm::vec4& pos, const glm::vec4& dir, float t)
{
	return { cos(t) * pos + sin(t) * dir };
}

glm::vec4 Math::sphGeoFlowDir(const glm::vec4& pos, const glm::vec4& dir, float t)
{
	return { sin(t) * pos + cos(t) * dir };
}

float Math::sphDot(const glm::vec4& u, const glm::vec4& v)
{
	return glm::dot(u,v);
}

float Math::sphNorm(const glm::vec4& v)
{
	return std::sqrt(std::abs(sphDot(v, v)));
}

glm::vec4 Math::sphNormalize(const glm::vec4& u)
{
	return u / sphNorm(u);
}

float Math::sphhDistance(const glm::vec4& u, const glm::vec4& v)
{
	const float bUV = sphDot(u, v);
	return acos(bUV);
}

glm::vec4 Math::sphDirection(const glm::vec4& u, const glm::vec4& v)
{
	return glm::vec4();	// todo
}

glm::vec4 Math::constructSpherePoint(const glm::vec3& direction, float distance)
{
	//this.matrix.identity();
	//const [x, y, z, w] = point.coords.toArray();
	//const u = new Vector3(x, y, z);
	//const c1 = u.length();
	//
	//if (c1 == = 0) {
	//	return this;
	//}
	//
	//const c2 = 1 - w;
	//u.normalize();
	//const m = new Matrix4().set(
	//	0, 0, 0, u.x,
	//	0, 0, 0, u.y,
	//	0, 0, 0, u.z,
	//	-u.x, -u.y, -u.z, 0
	//);
	//const m2 = m.clone().multiply(m);
	//m.multiplyScalar(c1);
	//this.matrix.add(m);
	//m2.multiplyScalar(c2);
	//this.matrix.add(m2);
	//
	//return this;
	return glm::vec4();
}

glm::mat4 Math::makeSphTranslation(const glm::vec4& p)
{
	return glm::mat4();
}

bool Math::isInS3(const glm::vec4& v)
{
	return withinError(sphDot(v,v), 1, 0.001);
}

void Math::printS3(const std::string& s, const glm::vec4& v)
{
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

	std::cout << s << " = " << toStr(v)
		<< "\n<" << s << "," << s << "> = " << sphDot(v, v)
		<< "\nnorm(" << s << ") = " << sphNorm(v)
		<< "\nisInS3(" << s << ") = " << (isInS3(v) ? "true" : "false")
		<< "\n\n";
}

bool Math::sphereUnitTests()
{
	// random direction vector scaled to random length
	auto randS3Point = [](int s) -> glm::vec4
	{
		const glm::vec4 v{ rng(s), rng(s + 1), rng(s + 2),  rng(s + 3) };

		return sphNormalize(v);
	};

	if (false)
	{
		const glm::vec4 ORIGIN{ 0,0,0,1 };
		const glm::vec4 E_X{ 1,0,0,0 };
		const glm::vec4 E_Y{ 0,1,0,0 };
		const glm::vec4 E_Z{ 0,0,1,0 };
		const glm::vec4 E_W{ 0,0,0,1 };

		const glm::vec4 xStep{ sphGeoFlowPos(ORIGIN, E_X, 2) };
		const glm::vec4 yStep{ sphGeoFlowPos(ORIGIN, E_Y, 2) };
		const glm::vec4 zStep{ sphGeoFlowPos(ORIGIN, E_Z, 2) };
		const glm::vec4 wStep{ sphGeoFlowPos(ORIGIN, E_W, 2) };

		const glm::vec4 xStepDir{ sphGeoFlowDir(ORIGIN, E_X, 2) };
		const glm::vec4 yStepDir{ sphGeoFlowDir(ORIGIN, E_Y, 2) };
		const glm::vec4 zStepDir{ sphGeoFlowDir(ORIGIN, E_Z, 2) };
		const glm::vec4 wStepDir{ sphGeoFlowDir(ORIGIN, E_W, 2) };

		printS3("x", xStep);
		printS3("y", yStep);
		printS3("z", zStep);
		printS3("w", wStep);

		printS3("xDir", xStepDir);
		printS3("yDir", yStepDir);
		printS3("zDir", zStepDir);
		printS3("wDir", wStepDir);
	}

	// verify that taking a point in the hyperboloid model of H3
		// and following flow of its geodesic gives a point still in H3
	for (int i = 0; i < 1000; ++i)
	{
		const glm::vec4 pos{ randS3Point(i) };

		const glm::vec3 dirEuc{ randS3Point(i+4) };
		const glm::vec4 dir{ sphNormalize(glm::vec4{dirEuc, 0}) };
		const float t{ rng(i) * 10 };	// [0,10]

		{
			const glm::vec4 nextPos{ sphNormalize(sphGeoFlowPos(pos, dir, t)) };
			const glm::vec4 nextDir{ sphNormalize(sphGeoFlowDir(pos, dir, t)) };

			const bool nextPosInS3{ isInS3(nextPos) };
			const bool nextDirInS3{ isInS3(nextDir) };
			const bool nextDirNormalized{ withinError(sphNorm(nextDir), 1, 0.01) };

			if (!(nextPosInS3 && nextDirInS3 && nextDirNormalized))
			{
				std::cout << "failed sphereical geodesic flow\n";
				printS3("pos", pos);
				printS3("dir", dir);
				printS3("next pos", nextPos);
				printS3("next dir", nextDir);
				return false;
			}
		}
	}
	return true;

}

bool Math::withinError(double approx, double expected, double tolerance)
{
	return std::abs(approx - expected) < tolerance;
}

double Math::getAbsoluteError(double approx, double expected)
{
	return std::abs(approx - expected);
}

double Math::getRelativeError(double approx, double expected)
{
	return std::abs((approx - expected) / expected);
}
