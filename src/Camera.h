#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

struct Camera
{
	glm::vec3 position{ 0.f };

	float pitch{};
	float yaw{};
	//float roll{}; todo

	glm::vec3 forwardDir{ 0.f, 0.f, -1.f };	// looking into screen
	glm::vec3 rightDir{ 1.f, 0.f, 0.f };
	glm::vec3 upDir{ 0.f, 1.f, 0.f };

	float FOV{ glm::half_pi<float>() };	// reminder, FOV is not the angle from the horizontal, but twice that

	glm::mat4 viewMat{ 1.f };

	void updateViewMat()
	{
		// derive forward direction of camera from its euler angles
		forwardDir.x = cosf(yaw) * cosf(pitch);
		forwardDir.y = sinf(pitch);
		forwardDir.z = sinf(yaw) * cosf(pitch);

		static constexpr glm::vec3 GLOBAL_UP{ 0.f, 1.f, 0.f };

		// unitize and calculate relative up and right direction of the camera
		forwardDir = glm::normalize(forwardDir);
		rightDir = glm::normalize(glm::cross(forwardDir, GLOBAL_UP));
		upDir = glm::normalize(glm::cross(rightDir, forwardDir));

		const glm::vec3 targetPosition{ position + forwardDir };
		viewMat = glm::lookAt(position, targetPosition, upDir);
	}
};