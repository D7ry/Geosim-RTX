#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Renderer.h"
#include "Scene.h"

#include "util/Window.h"

#include <iostream>

#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>

static constexpr sf::Keyboard::Key FORWARD{ sf::Keyboard::Key::W };
static constexpr sf::Keyboard::Key BACKWARD{ sf::Keyboard::Key::S };
static constexpr sf::Keyboard::Key LEFT{ sf::Keyboard::Key::A };
static constexpr sf::Keyboard::Key RIGHT{ sf::Keyboard::Key::D };
static constexpr sf::Keyboard::Key UP{ sf::Keyboard::Key::Space };
static constexpr sf::Keyboard::Key DOWN{ sf::Keyboard::Key::LShift };

static constexpr sf::Keyboard::Key CAM_SLOW_SPD_KEY{ sf::Keyboard::Key::Num1 };
static constexpr sf::Keyboard::Key CAM_MED_SPD_KEY{ sf::Keyboard::Key::Num2 };
static constexpr sf::Keyboard::Key CAM_FAST_SPD_KEY{ sf::Keyboard::Key::Num3 };

int main()
{
    constexpr unsigned WIN_W{ 16 << 3 };
    constexpr unsigned WIN_H{ 9 << 3 };

    Window window{ WIN_W, WIN_H, 8, "Rodent-Raytracer" };

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

    Geometry floor;

    s.radius = 1000;
    s.material.color = { 0.8f, 0.8f, 1.f, 1.f };
    
    floor.add(s);
    floor.position = { 0, -1001, 0 };
    scene.add(floor);

    Renderer renderer;
    Camera camera;
    

    // to face -z
    const float zYaw{ 3.14f * 3 / 2 };
    camera.yaw = zYaw;

    sf::Vector2i mPosPrev{ sf::Mouse::getPosition() };
    sf::Vector2i mPosCur{ sf::Mouse::getPosition() };

    float camMoveSpd{ 1 };

    int tick{};

    while (window.isOpen())
    {
        window.update();

        {
            // keyboard input for moving camera pos
            if (sf::Keyboard::isKeyPressed(FORWARD))
                camera.position += camera.forwardDir * camMoveSpd;
            if (sf::Keyboard::isKeyPressed(BACKWARD))
                camera.position -= camera.forwardDir * camMoveSpd;
            if (sf::Keyboard::isKeyPressed(RIGHT))
                camera.position += camera.rightDir * camMoveSpd;
            if (sf::Keyboard::isKeyPressed(LEFT))
                camera.position -= camera.rightDir * camMoveSpd;
            if (sf::Keyboard::isKeyPressed(UP))
                camera.position += camera.upDir * camMoveSpd;
            if (sf::Keyboard::isKeyPressed(DOWN))
                camera.position -= camera.upDir * camMoveSpd;

            if (sf::Keyboard::isKeyPressed(CAM_SLOW_SPD_KEY))
                camMoveSpd = 0.01;
            if (sf::Keyboard::isKeyPressed(CAM_MED_SPD_KEY))
                camMoveSpd = 0.1;
            if (sf::Keyboard::isKeyPressed(CAM_FAST_SPD_KEY))
                camMoveSpd = 1;

            // mouse input for angling/pointing camera
            mPosPrev = mPosCur;
            mPosCur = sf::Mouse::getPosition();

            const sf::Vector2i mouseMove{ mPosCur - mPosPrev };
            camera.yaw += mouseMove.x / 64.f;
            camera.pitch += -mouseMove.y / 64.f;    // subtract so controls aren't inverted
        }

        // update
        tick++;

        auto& first = scene.geometry[0];
        auto& second = scene.geometry[1];

        first.position.z += sinf(tick / 32.f) / 16.f;
        
        second.position.x = cosf(tick / 4.f) * 4;
        second.position.y = sinf(tick / 4.f) * 4;

        //camera.yaw = zYaw + cosf(tick / 16.f) / 4.f;
        //camera.pitch = cosf(tick / 16.f) / 4.f;

        //camera.position.z = cosf(tick / 16.f) * 2.f;

        //camera.yaw   = std::clamp(camera.yaw,   -glm::pi<float>(), glm::pi<float>());
        //camera.pitch = std::clamp(camera.pitch, -glm::pi<float>(), glm::pi<float>());

        camera.updateViewMat();

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