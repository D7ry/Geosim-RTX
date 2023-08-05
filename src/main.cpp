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

#include <memory>

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
 
    std::shared_ptr<Material> greenMat = std::make_shared<Dielectric>();
    greenMat->albedo = { 0,1,0 };
    greenMat->roughness = 1;
    greenMat->emissionStrength = .5;
    greenMat->emissionColor = { 0,1,1 };

    std::shared_ptr<Material> redMat = std::make_shared<Dielectric>();
    redMat->albedo = { 1,0,0 };
    redMat->roughness = 0.5;

    std::shared_ptr<Dielectric> mirrorMat = std::make_shared<Dielectric>();
    mirrorMat->albedo = { 1,1,1 };
    mirrorMat->roughness = 0;
    mirrorMat->ior = 1.5;

    std::shared_ptr<Material> floorMat = std::make_shared<Dielectric>();
    floorMat->albedo = { 0.6f, 0.6f, 1.f };
    floorMat->roughness = .1;

    std::shared_ptr<Material> whiteMat = std::make_shared<Dielectric>();
    whiteMat->albedo = { 1,1,1 };
    whiteMat->roughness = 0.9;

    std::shared_ptr<Material> lightMat = std::make_shared<Dielectric>();
    lightMat->albedo = { 1,1,1 };
    lightMat->roughness = 1;
    lightMat->emissionStrength = 1;
    lightMat->emissionColor = { 1,.9,.8 };

    // create scene
    {
        Geometry snowmanObject;    // scene will have one object

        // create geometry of primitives
        Sphere head;
        head.position = { 0.f, 1.f, 0.f };
        head.radius = .5;

        Sphere middle;
        middle.position = { 0.f, 0.f, 0.f };
        middle.radius = .7;

        Sphere bottom;
        bottom.position = { 0.f, -1.1f, 0.f };
        bottom.radius = .9;

        // assign materials
        head.material = whiteMat;
        middle.material = whiteMat;
        bottom.material = whiteMat;

        // add primitives to object
        snowmanObject.add(head);
        snowmanObject.add(middle);
        snowmanObject.add(bottom);

        // add one instance of obj to scene
        snowmanObject.position = { 0, 1, -4 };
        scene.add(snowmanObject);

        // add another, positioned elsewhere
        snowmanObject.position = { 3, 2, -4 };
        //scene.add(snowmanObject);
    }

    {
        Geometry floorObject;
        Sphere floor;

        floor.radius = 1000;
        floor.material = floorMat;

        floorObject.add(floor);
        floorObject.position = { 0, -1001, 0 };

        scene.add(floorObject);
    }

    {
        Geometry lightObject;
        Sphere light;

        light.radius = 1000;
        light.material = lightMat;

        lightObject.add(light);
        lightObject.position = { 0, 1234, 0 };

        //scene.add(lightObject);
    }

    {
        Geometry object;

        Sphere mirror;
        mirror.position = { 0,0,0 };

        Sphere tomato;
        tomato.position = { 2,0,0 };

        Sphere watermelon;
        watermelon.position = { -2,0,0 };

        mirror.material = mirrorMat;
        tomato.material = redMat;
        watermelon.material = greenMat;

        object.add(mirror);
        object.add(tomato);
        object.add(watermelon);

        object.position = { 0, 0, -2 };
        scene.add(object);
    }

    Renderer renderer;
    Camera camera;

    // to face -z
    camera.yaw = glm::three_over_two_pi<float>();

    sf::Vector2i mPosPrev{ sf::Mouse::getPosition() };
    sf::Vector2i mPosCur{ sf::Mouse::getPosition() };

    float camMoveSpd{ CAM_MED_SPD };

    camera.position = { -1.89711, 2.31398, 0.773041 };
    camera.pitch = -0.573735;
    camera.yaw = -13.6802;

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
        globalTick++;

        // update scene
        // ...

        // update camera
        camera.pitch = std::clamp(camera.pitch, -glm::half_pi<float>(), glm::half_pi<float>());

        camera.updateViewMat();

        /// render scene to image
        renderer.render(scene, camera, image);

        if constexpr (INTERACTIVE_MODE)
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