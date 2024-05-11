#pragma once

#include <glm/matrix.hpp>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Camera
{
	glm::vec3 position{ 0.f };
	glm::vec4 positionHyp{ 0,0,0,1 };

	float pitch{};
	float yaw{};
	//float roll{}; todo

	glm::vec3 forwardDir{ 0.f, 0.f, -1.f };	// looking into screen
	glm::vec3 rightDir{ 1.f, 0.f, 0.f };
	glm::vec3 upDir{ 0.f, 1.f, 0.f };

	glm::vec4 forwardDirHyp{ 0.f, 0.f, -1.f, 0 };	// looking into screen
	glm::vec4 rightDirHyp{ 1.f, 0.f, 0.f, 0 };
	glm::vec4 upDirHyp{ 0.f, 1.f, 0.f, 0 };

	float FOV{ glm::half_pi<float>() };	// reminder, FOV is not the angle from the horizontal, but twice that

	glm::mat4 viewMat{ 1.f };

    void updateViewMat();
};
