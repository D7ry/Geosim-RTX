#pragma once

#include <string>
#include <glm/glm.hpp>

#include <vector>


struct Image {
    const unsigned width;
    const unsigned height;

    std::vector<glm::vec3> pixels;

    Image(unsigned width, unsigned height) : width{width}, height{height}, pixels{width * height} {}

    void setPixel(const glm::vec2& ndc, const glm::vec3& pixelData);

    void saveToFile(const std::string& filename);
};
