
#include "Image.h"

#include "util/Window.h"

int main()
{
    constexpr unsigned WIN_W{ 256 };
    constexpr unsigned WIN_H{ 256 };

    Window window{ WIN_W, WIN_H, "Rodent-Raytracer" };

    Image img{ WIN_W, WIN_H };

    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x)
        {
            const unsigned index{ (y * img.width) + x };

            img.pixels[index].r = (float)(x) / img.width;
            img.pixels[index].g = (float)(y) / img.height;
            img.pixels[index].b = 0.f;
            img.pixels[index].a = 1.f;
        }

    while (window.isOpen())
    {
        window.update();

        window.clear();
        window.setBuffer(img);
        window.display();
    }

    return 0;
}