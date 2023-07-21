#include "Camera.h"
#include "Image.h"
#include "Primitive.h"
#include "Renderer.h"
#include "Scene.h"

#include "util/Window.h"

#include <iostream>

#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Mouse.hpp>

#include <SFML/System/Time.hpp>

#include "Settings.h"

// settings and controls
static constexpr sf::Keyboard::Key FORWARD{ sf::Keyboard::Key::W };
static constexpr sf::Keyboard::Key BACKWARD{ sf::Keyboard::Key::S };
static constexpr sf::Keyboard::Key LEFT{ sf::Keyboard::Key::A };
static constexpr sf::Keyboard::Key RIGHT{ sf::Keyboard::Key::D };
static constexpr sf::Keyboard::Key UP{ sf::Keyboard::Key::Space };
static constexpr sf::Keyboard::Key DOWN{ sf::Keyboard::Key::LShift };
static constexpr sf::Keyboard::Key CLOSE{ sf::Keyboard::Key::Escape };

static constexpr sf::Keyboard::Key CAM_DEBUG{ sf::Keyboard::Key::C };

static constexpr sf::Keyboard::Key CAM_SLOW_SPD_KEY{ sf::Keyboard::Key::Num1 };
static constexpr sf::Keyboard::Key CAM_MED_SPD_KEY{ sf::Keyboard::Key::Num2 };
static constexpr sf::Keyboard::Key CAM_FAST_SPD_KEY{ sf::Keyboard::Key::Num3 };

static constexpr float CAM_SLOW_SPD{ 0.01f };
static constexpr float CAM_MED_SPD{  0.10f };
static constexpr float CAM_FAST_SPD{ 1.00f };

static constexpr float MOUSE_SENSITIVITY{ 1.5f };

static constexpr unsigned WINDOW_WIDTH{  INTERACTIVE_MODE ? INTERACTIVE_WIDTH  : OFFLINE_WIDTH  };
static constexpr unsigned WINDOW_HEIGHT{ INTERACTIVE_MODE ? INTERACTIVE_HEIGHT : OFFLINE_HEIGHT };
static constexpr unsigned WINDOW_SCALE{ 1 << 3 };

int main()
{
    sf::Clock timer;
    timer.restart();

    Window* window{ nullptr };
    
    if constexpr (INTERACTIVE_MODE)
    {
        window = new Window{
           WINDOW_WIDTH,
           WINDOW_HEIGHT,
           WINDOW_SCALE,
           "Rodent-Raytracer"
        };
    }

    Image image{ WINDOW_WIDTH, WINDOW_HEIGHT };

    Scene scene;

    // create scene
    {
        Geometry object;    // scene will have one object
        Sphere s;           // object is made up of primitives (three spheres)

        // head
        s.position = { 0.f, 1.f, 0.f };
        s.radius = .5;
        s.material.color = { 1,0,0,1 };
        s.material.roughness = 1;
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
        object.position = { 0, 1, -5 };
        scene.add(object);

        // add another, positioned elsewhere
        object.position = { 3, 2, -4 };
        scene.add(object);

        Geometry floor;

        s.radius = 1000;
        s.material.color = { 0.6f, 0.6f, 1.f, 1.f };
        s.material.roughness = 1;
        s.material.emissionStrength = .5;
        s.material.emissionColor = { .5,.5,1,1 };

        floor.add(s);

        floor.position = { 0, -1000, 0 };
        scene.add(floor);
    }

    {
        Geometry object;

        Sphere mirror;
        mirror.material.color = { 1,1,1,1 };
        mirror.material.roughness = 0;

        mirror.position = { 0,0,0 };

        Sphere tomato;
        tomato.material.color = { 1, 0.3, 0.3, 1 };
        tomato.material.roughness = .3;

        tomato.position = { 2,0,0 };

        Sphere watermelon;
        watermelon.material.color = { 0.1, 1, 0.1, 1 };
        watermelon.material.roughness = .8;
        watermelon.material.emissionColor = { 0.5, 1, 0.2, 1 };
        watermelon.material.emissionStrength = 0;

        watermelon.position = { -2,0,0 };

        object.add(mirror);
        object.add(tomato);
        object.add(watermelon);

        //object.position = { 5, 0, 5 };
        object.position = { 0, 0, -2 };
        scene.add(object);
    }

    Renderer renderer;
    Camera camera;

    // to face -z
    camera.yaw = glm::three_over_two_pi<float>();

    camera.position = { -3.46753, 2.47487, -3.21886 };
    camera.pitch = -0.533686;
    camera.yaw = 6.39403;

    sf::Vector2i mPosPrev{ sf::Mouse::getPosition() };
    sf::Vector2i mPosCur{ sf::Mouse::getPosition() };

    float camMoveSpd{ 1 };

    int tick{};

    while (!INTERACTIVE_MODE || window->isOpen())
    {
        if (window)
            window->update();

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
            if (sf::Keyboard::isKeyPressed(CLOSE))
                return 0;

            if (sf::Keyboard::isKeyPressed(CAM_DEBUG))
            {
                const auto& c = camera;

                // print pos and euler angles of camera
                std::cout << "camera.position = {" <<
                    c.position.x << ", " <<
                    c.position.y << ", " <<
                    c.position.z <<
                    "};\ncamera.pitch = " << c.pitch << 
                    ";\ncamera.yaw = " << c.yaw << ";\n\n";
            }

            if (sf::Keyboard::isKeyPressed(CAM_SLOW_SPD_KEY))
                camMoveSpd = CAM_SLOW_SPD;
            if (sf::Keyboard::isKeyPressed(CAM_MED_SPD_KEY))
                camMoveSpd = CAM_MED_SPD;
            if (sf::Keyboard::isKeyPressed(CAM_FAST_SPD_KEY))
                camMoveSpd = CAM_FAST_SPD;

            // mouse input for angling/pointing camera
            mPosPrev = mPosCur;
            mPosCur = sf::Mouse::getPosition();

            const sf::Vector2i mouseMove{ mPosCur - mPosPrev };

            camera.yaw   +=  (mouseMove.x * MOUSE_SENSITIVITY) / 256.f;
            camera.pitch += -(mouseMove.y * MOUSE_SENSITIVITY) / 256.f;    // subtract so controls aren't inverted
        }

        /// update
        tick++;

        // update scene
        //auto& first = scene.geometry[0];
        //auto& second = scene.geometry[1];
        //
        //first.position.z += sinf(tick / 32.f) / 16.f;
        //
        //second.position.x = cosf(tick / 4.f) * 4;
        //second.position.y = sinf(tick / 4.f) * 4;

        //for (auto& o : scene.geometry)
        //{}

        // update camera
        camera.pitch = std::clamp(camera.pitch, -glm::half_pi<float>(), glm::half_pi<float>());

        camera.updateViewMat();

        /// render scene to image
        renderer.render(scene, camera, image);

        if (INTERACTIVE_MODE)
        {
            window->clear();
            window->setBuffer(image);
            window->display();
        }
        else
        {
            const float renderTime{ timer.getElapsedTime().asSeconds()};

            std::cout << "Render time: " << renderTime << '\n';

            image.saveToFile("render " + std::to_string(renderTime) + "s");

            return 0;
        }
    }

    return 0;
}