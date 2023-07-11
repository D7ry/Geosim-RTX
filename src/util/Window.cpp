#include "Window.h"

#include "../Image.h"

#include <SFML/Window/Event.hpp>

#include <algorithm>
#include <iostream>

Window::Window(int width, int height, int scale, const std::string& title)
    :
    window{ sf::VideoMode(width * scale, height * scale), title, sf::Style::Close },
    scale{ scale }
{
    // sprite should always be located at 0,0
    s.setPosition({ 0, 0 });
    s.setScale(sf::Vector2f( scale, scale ));
    onResize();
}

void Window::update()
{
    sf::Event event;

    while (window.pollEvent(event))
    {
        switch (event.type)
        {
            typedef sf::Event::EventType Event;

        case Event::Closed:
            window.close();
            break;

        case Event::Resized:
            onResize();
            break;

        default:
            constexpr bool PRINT_EVENT_CODES{ false };
            if constexpr (PRINT_EVENT_CODES)
                std::cout << "Event: " << event.type << '\n';
        }
    }
}

void Window::clear()
{
    window.clear();
}

void Window::setBuffer(const Image& img)
{
    const bool sameSize{
        buffer.getSize().x == img.width &&
        buffer.getSize().y == img.height
    };

    // todo make resizeable
    //assert(sameSize);

    const size_t imgDim{ img.pixels.size() };

    static std::vector<sf::Uint8> pixels;

    constexpr int NUM_CHANNELS{ 4 };

    if (pixels.size() != imgDim * NUM_CHANNELS)
        pixels.resize(imgDim * NUM_CHANNELS);

    // iterate through every pixel in img
    for (int i = 0; i < imgDim; ++i)
    {
        const glm::vec4& color{ img.pixels[i] };

        pixels[(i * 4) + 0] = (sf::Uint8)(std::clamp(color.r, 0.f, 1.f) * 255);
        pixels[(i * 4) + 1] = (sf::Uint8)(std::clamp(color.g, 0.f, 1.f) * 255);
        pixels[(i * 4) + 2] = (sf::Uint8)(std::clamp(color.b, 0.f, 1.f) * 255);
        pixels[(i * 4) + 3] = (sf::Uint8)(std::clamp(color.a, 0.f, 1.f) * 255);
    }

    buffer.update(pixels.data());

    s.setTexture(buffer, true);
    
    window.draw(s);
}

void Window::display()
{
    window.display();
}

bool Window::isOpen() const
{
    return window.isOpen();
}

bool Window::isFocused() const
{
    return window.hasFocus();
}

const sf::Vector2u& Window::getDim() const
{
    return window.getSize();
}

void Window::onResize()
{
    unsigned width{  getDim().x  };
    unsigned height{ getDim().y };

    buffer.create(width / scale, height / scale);
    
    s.setTexture(buffer, true);
}