#include <SFML/Graphics/Image.hpp>

#include "Image.h"


void Image::setPixel(const glm::vec2& ndc, const glm::vec3& pixelData) {
    glm::uvec2 pixelCoord{ndc.x * width, ndc.y * height};

    int index = pixelCoord.x + (pixelCoord.y * width);

    pixels[index] = pixelData;
}
void Image::saveToFile(const std::string& filename) {
    std::vector<sf::Uint8> uint8Pixels;

    for (const glm::vec3& pixel : pixels) {
        uint8Pixels.push_back(pixel.r * 255);
        uint8Pixels.push_back(pixel.g * 255);
        uint8Pixels.push_back(pixel.b * 255);

        static constexpr bool ALWAYS_OPAQUE{true};

        if constexpr (ALWAYS_OPAQUE)
            uint8Pixels.push_back(255);
        // else
        //	uint8Pixels.push_back(pixel.a * 255);
    }

    sf::Image i;
    i.create(width, height, uint8Pixels.data());
    i.saveToFile(filename + " hash" + std::to_string((uint64_t)&i) + ".png");
}
