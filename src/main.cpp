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

    // head
    s.position = { 0.f, 1.f, 0.f };
    s.radius = .5;
    s.material.color = { 1,0,0,1 };
    object.add(s);

   // middle
    s.position = { 0.f, 0.f, 0.f };
    s.radius = .7;
    s.material.color = { 0,1,0,1 };
    object.add(s);

    // bottom
    s.position = { 0.f, -1.1f, 0.f }; 
    s.radius = .9;
    s.material.color = { 0,0,1,1 };
    object.add(s);

    // add one instance of obj to scene
    object.position = { 0, 0, -5 };
    scene.add(object);

    // add another, positioned elsewhere
    object.position = { 3, 1, -4 };
    scene.add(object);

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