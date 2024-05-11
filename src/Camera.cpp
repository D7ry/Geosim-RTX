#include "Camera.h"

void Camera::updateViewMat() {
    // derive forward direction of camera from its euler angles
    forwardDir.x = cosf(yaw) * cosf(pitch);
    forwardDir.y = sinf(pitch);
    forwardDir.z = sinf(yaw) * cosf(pitch);

    static constexpr glm::vec3 GLOBAL_UP{0.f, 1.f, 0.f};

    // unitize and calculate relative up and right direction of the camera
    forwardDir = glm::normalize(forwardDir);
    rightDir = glm::normalize(glm::cross(forwardDir, GLOBAL_UP));
    upDir = glm::normalize(glm::cross(rightDir, forwardDir));

    const glm::vec3 targetPosition{position + forwardDir};
    viewMat = glm::lookAt(position, targetPosition, upDir);
}
