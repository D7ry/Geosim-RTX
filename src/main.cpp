#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Renderer.h"
#include "Scene.h"

#include "util/Window.h"

#include <iostream>

int main()
{
    constexpr unsigned WIN_W{ 16 << 4 };
    constexpr unsigned WIN_H{ 9 << 4 };

    Window window{ WIN_W, WIN_H, 4, "Rodent-Raytracer" };

    Image image{ WIN_W, WIN_H };

    Scene scene;
    Geometry object;    // scene will have one object
    Sphere s;           // object is made up of primitives (three spheres)

    s.position = { 0.f, 1.f, 0.f };   // here is one sphere (head)
    s.radius = .5;
    object.add(s);                     // add primitive to object

    s.position = { 0.f, 0.f, 0.f };   // here is a second sphere (middle)
    s.radius = .7;
    object.add(s);                     // add primitive to object

    s.position = { 0.f, -1.1f, 0.f }; // here is the last sphere (bottom)
    s.radius = .9;
    object.add(s);                     // add primitive to object

    object.position = { 0, 0, -5 };
    scene.add(object);  // add object to the geometry of the scene

    object.position = { 3, 1, -4 };
    scene.add(object);  // add object to the geometry of the scene

    Renderer renderer;
    Camera camera;
    
    int tick{};

    while (window.isOpen())
    {
        window.update();

        tick++;

        auto& first = scene.geometry[0];
        auto& second = scene.geometry[1];

        first.position.z += sinf(tick / 32.f) / 16.f;

        second.position.x = cosf(tick / 4.f) * 4;
        second.position.y = sinf(tick / 4.f) * 4;
        
        for (auto& o : scene.geometry)
        {}

        // render scene to image
        renderer.render(scene, camera, image);

        window.clear();
        window.setBuffer(image);
        window.display();
    }

    return 0;
}