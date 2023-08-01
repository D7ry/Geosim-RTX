#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>

struct Image;

// todo: make resizing window resize application resolution
class Window
{
public:
	Window(int width, int height, int scale, const std::string& title);

	void update();

	void clear();
	void setBuffer(const Image& pixels);
	void display();

	bool isOpen() const;
	bool isFocused() const;
	
	const sf::Vector2u& getDim() const;

private:
	sf::RenderWindow window;
	
	// contains pixel data of screen
	sf::Texture buffer;
	
	// buffer will be displayed through sprite
	sf::Sprite s;

	int scale{ 1 };

private:
	void onResize();
};