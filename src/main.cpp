
#include "Camera.h"
#include "Image.h"
#include "Renderer.h"
#include "Scene.h"

#include "util/Window.h"

int main()
{
    constexpr unsigned WIN_W{ 16 << 4 };
    constexpr unsigned WIN_H{ 9 << 4 };

    Window window{ WIN_W, WIN_H, 2, "Rodent-Raytracer" };

    Image image{ WIN_W, WIN_H };

    Scene scene;
    Geometry unitSphere;
    //unitSphere.position = { 1.f, 2.f, -5.f };
    unitSphere.position = { 0.f, 0.f, -3.f };
    scene.geometry.push_back(unitSphere);   // scene is one unit sphere about origin

    Renderer renderer;
    Camera camera;
    
    int tick{};

    while (window.isOpen())
    {
        window.update();

        tick++;
        camera.FOV = glm::half_pi<float>() + sinf(tick / 32.f) / glm::quarter_pi<float>();
        //scene.geometry.back().position.z = -3.f + sinf(tick / 32.f);

        // render scene to image
        renderer.render(scene, camera, image);

        window.clear();
        window.setBuffer(image);
        window.display();
    }

    return 0;
}