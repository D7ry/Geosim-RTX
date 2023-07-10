
#include "Camera.h"
#include "Image.h"
#include "Renderer.h"
#include "Scene.h"

#include "util/Window.h"

int main()
{
    constexpr unsigned WIN_W{ 256 };
    constexpr unsigned WIN_H{ 256 };

    Window window{ WIN_W, WIN_H, "Rodent-Raytracer" };

    Image image{ WIN_W, WIN_H };

    Scene scene;
    scene.geometry.push_back(Geometry{});   // scene is one unit sphere about origin

    Renderer renderer;
    Camera camera;
    
    while (window.isOpen())
    {
        window.update();

        // render scene to image
        renderer.render(scene, camera, image);

        window.clear();
        window.setBuffer(image);
        window.display();
    }

    return 0;
}