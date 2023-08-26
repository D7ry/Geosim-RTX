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
static constexpr sf::Keyboard::Key RESPAWN{ sf::Keyboard::Key::X };

static constexpr sf::Keyboard::Key ENABLE_ACCUMULATION{ sf::Keyboard::Key::Equal };
static constexpr sf::Keyboard::Key DISABLE_ACCUMULATION{ sf::Keyboard::Key::Dash };

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
 
    std::shared_ptr<Dielectric> greenMat = std::make_shared<Dielectric>();
    greenMat->albedo = { 0,1,0 };
    greenMat->roughness = 1;
    greenMat->emissionStrength = .5;
    greenMat->emissionColor = { 0,1,1 };

    std::shared_ptr<Dielectric> redMat = std::make_shared<Dielectric>();
    redMat->albedo = { 1,0,0 };
    redMat->roughness = 0.5;

    std::shared_ptr<Dielectric> evilMat = std::make_shared<Dielectric>();
    evilMat->albedo = { .7,0,0 };
    evilMat->roughness = 0.7;
    evilMat->emissionStrength = .2;
    evilMat->emissionColor = { 1,0,0 };

    std::shared_ptr<Dielectric> mirrorMat = std::make_shared<Dielectric>();
    mirrorMat->albedo = { 1,1,1 };
    mirrorMat->roughness = 0;
    mirrorMat->ior = 1.5;

    std::shared_ptr<Dielectric> glassMat = std::make_shared<Dielectric>();
    glassMat->albedo = { 1,1,1 };
    glassMat->roughness = 0;
    glassMat->ior = 1.5;
    glassMat->opacity = 0;

    std::shared_ptr<Dielectric> floorMat = std::make_shared<Dielectric>();
    floorMat->albedo = { 0.6f, 0.6f, 1.f };
    floorMat->roughness = .1;

    std::shared_ptr<Dielectric> whiteMat = std::make_shared<Dielectric>();
    whiteMat->albedo = { 1,1,1 };
    whiteMat->roughness = 0.9;

    std::shared_ptr<Dielectric> lightMat = std::make_shared<Dielectric>();
    lightMat->albedo = { 1,1,1 };
    lightMat->roughness = .5;
    lightMat->emissionStrength = .5;
    lightMat->emissionColor = { 1,.9,.8 };

    /// create scene
    // snowman
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
        head.material =   lightMat;
        middle.material = lightMat;
        bottom.material = lightMat;

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

    // floor
    {
        Geometry floorObject;
        Sphere floor;

        floor.radius = 1000;
        floor.material = floorMat;

        floorObject.add(floor);
        floorObject.position = { 0, -1001, 0 };

        scene.add(floorObject);
    }

    // sun
    {
        Geometry lightObject;
        Sphere light;

        light.radius = 1000;
        light.material = lightMat;

        lightObject.add(light);
        lightObject.position = { 0, 1234, 0 };

        //scene.add(lightObject);
    }

    // 3 balls
    {
        Geometry object;

        Sphere mirror;
        mirror.position = { 0,0,0 };

        Sphere tomato;
        tomato.position = { 2,0,0 };

        Sphere watermelon;
        watermelon.position = { -2,0,0 };

        mirror.material = glassMat;
        tomato.material = redMat;
        watermelon.material = greenMat;

        object.add(mirror);
        object.add(tomato);
        object.add(watermelon);

        object.position = { 0, 0, -2 };
        scene.add(object);
    }

    // plane
    {
        Geometry planeObject;
        Plane plane;

        plane.material = mirrorMat;

        planeObject.add(plane);
        planeObject.position = { 10,10,0 };

        scene.add(planeObject);
    }

    // triangle
    {
        Geometry triangleObject;
        Triangle triangle;

        triangle.material = greenMat;
        triangle.vertices[0] = { 0, 0, 0 };
        triangle.vertices[1] = { 0.5, 1, 0 };
        triangle.vertices[2] = { 1, 0, 0 };

        triangleObject.add(triangle);
        triangleObject.position = { 5,3,-3 };

        scene.add(triangleObject);
    }

    // red thing
    Renderer renderer;
    {
        Geometry evilObject;
        Sphere s;

        s.material = evilMat;
        s.radius = 5;

        evilObject.add(s);
        evilObject.position = { -5,-10, 3 };

        scene.add(evilObject);
    }
    Camera camera;

    // to face -z
    camera.yaw = glm::three_over_two_pi<float>();

    camera.position = { 5.07275, 4.13277, -5.65936 };
    camera.pitch = -0.498047;
    camera.yaw = 2.15184;

    sf::Vector2i mPosPrev{ sf::Mouse::getPosition() };
    sf::Vector2i mPosCur{ sf::Mouse::getPosition() };

    float camMoveSpd{ CAM_MED_SPD };

    int tick{};

    if constexpr (INTERACTIVE_MODE)
        std::cout << "Controls:\n"
        << "Move: WASD LShift Space\n"
        << "Change Speed: 123\n"
        << "Accumulate Off/On: -/+\n"
        << "Respawn: x\n"
        << "Close: Esc\n"
        << "Have Fun! (Moving while accumulating is cool)\n";

    while (!INTERACTIVE_MODE || window->isOpen())
    {
        if (window)
            window->update();

        // handle input
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

            if (sf::Keyboard::isKeyPressed(RESPAWN))
                camera.position = { 0.f, 0.f, 0.f };

            if (sf::Keyboard::isKeyPressed(ENABLE_ACCUMULATION))
                renderer.accumulate = true;
            if (sf::Keyboard::isKeyPressed(DISABLE_ACCUMULATION))
            {
                renderer.accumulate = false;
                renderer.resetAccumulator();
            }
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
        const float minPitch{ -glm::half_pi<float>() + 0.01f };
        const float maxPitch{  glm::half_pi<float>() - 0.01f };
        camera.pitch = std::clamp(camera.pitch, minPitch, maxPitch);

        camera.updateViewMat();

        if constexpr (!INTERACTIVE_MODE)
            std::cout << "Starting render... (be patient this might take a while)\n";

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