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
#include <random>

float prng(float min, float max) {
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_real_distribution<float> distr(min, max); // Define the range

    return distr(eng); // Generate and return a random float
}

// settings and controls
static constexpr sf::Keyboard::Key FORWARD{sf::Keyboard::Key::W};
static constexpr sf::Keyboard::Key BACKWARD{sf::Keyboard::Key::S};
static constexpr sf::Keyboard::Key LEFT{sf::Keyboard::Key::A};
static constexpr sf::Keyboard::Key RIGHT{sf::Keyboard::Key::D};
static constexpr sf::Keyboard::Key UP{sf::Keyboard::Key::Space};
static constexpr sf::Keyboard::Key DOWN{sf::Keyboard::Key::LShift};
static constexpr sf::Keyboard::Key CLOSE{sf::Keyboard::Key::Escape};
static constexpr sf::Keyboard::Key RESPAWN{sf::Keyboard::Key::X};

static constexpr sf::Keyboard::Key INC_PITCH{sf::Keyboard::Key::Up};
static constexpr sf::Keyboard::Key DEC_PITCH{sf::Keyboard::Key::Down};
static constexpr sf::Keyboard::Key INC_YAW{sf::Keyboard::Key::Right};
static constexpr sf::Keyboard::Key DEC_YAW{sf::Keyboard::Key::Left};

static constexpr sf::Keyboard::Key ENABLE_ACCUMULATION{sf::Keyboard::Key::Equal
};
static constexpr sf::Keyboard::Key DISABLE_ACCUMULATION{sf::Keyboard::Key::Dash
};

static constexpr sf::Keyboard::Key CAM_DEBUG{sf::Keyboard::Key::C};

static constexpr sf::Keyboard::Key CAM_SLOW_SPD_KEY{sf::Keyboard::Key::Num1};
static constexpr sf::Keyboard::Key CAM_MED_SPD_KEY{sf::Keyboard::Key::Num2};
static constexpr sf::Keyboard::Key CAM_FAST_SPD_KEY{sf::Keyboard::Key::Num3};

static constexpr float CAM_SLOW_SPD{0.01f};
static constexpr float CAM_MED_SPD{0.10f};
static constexpr float CAM_FAST_SPD{1.00f};

static constexpr float MOUSE_SENSITIVITY{1.5f};

#if CUDA
#include "Rotor.h"
#include "gpu.h"
#include "cuda_scenes.h"
#endif

int main() {
    // if (!Math::hyperbolicUnitTests())
    //{
    //     std::cout << "unit tests failed";
    //     return 0;
    // }

    // if (!Math::sphereUnitTests())
    //{
    //     std::cout << "unit tests failed";
    //     return 0;
    // }

    sf::Clock timer;
    timer.restart();

    Window* window = nullptr;
    const int window_width = 1280;
    const int window_height = 720;
    const int window_scale = 1;

    if (INTERACTIVE_MODE) {
        window = new Window(
            window_width, window_height, window_scale, "Rodent-Raytracer RTX"
        );
    }

    Image image{window_width, window_height};

#if CUDA
    RendererCUDA::init();
    CUDAStruct::Scene scene;

    // Rotor rotor;
#else
    Scene scene;
#endif

    std::shared_ptr<Dielectric> greenMat = std::make_shared<Dielectric>();
    greenMat->albedo = {0, 1, 0};
    greenMat->roughness = 1;
    greenMat->emissionStrength = .5;
    greenMat->emissionColor = {0, 1, 1};

    std::shared_ptr<Dielectric> redMat = std::make_shared<Dielectric>();
    redMat->albedo = {1, 0, 0};
    redMat->roughness = 0.5;

    std::shared_ptr<Dielectric> evilMat = std::make_shared<Dielectric>();
    evilMat->albedo = {.7, 0, 0};
    evilMat->roughness = 0.7;
    evilMat->emissionStrength = .2;
    evilMat->emissionColor = {1, 0, 0};

    std::shared_ptr<Dielectric> mirrorMat = std::make_shared<Dielectric>();
    mirrorMat->albedo = {1, 1, 1};
    mirrorMat->roughness = 0;
    mirrorMat->ior = 1.5;

    std::shared_ptr<Dielectric> glassMat = std::make_shared<Dielectric>();
    glassMat->albedo = {1, 1, 1};
    glassMat->roughness = 0;
    glassMat->ior = 1.5;
    glassMat->opacity = 0;

    std::shared_ptr<Dielectric> floorMat = std::make_shared<Dielectric>();
    floorMat->albedo = {0.6f, 0.6f, 1.f};
    floorMat->roughness = .1;

    std::shared_ptr<Dielectric> whiteMat = std::make_shared<Dielectric>();
    whiteMat->albedo = {1, 1, 1};
    whiteMat->roughness = 0.9;

    std::shared_ptr<Dielectric> lightMat = std::make_shared<Dielectric>();
    lightMat->albedo = {1, 1, 1};
    lightMat->roughness = .5;
    lightMat->emissionStrength = .5;
    lightMat->emissionColor = {1, .9, .8};

    /// create scene
    // snowman
    if (true) {
#if CUDA

#else
        Geometry snowmanObject; // scene will have one object

        // create geometry of primitives
        Sphere head;
        head.position = {0.f, 1.f, 0.f};
        head.radius = 1.2;

        Sphere middle;
        middle.position = {0.f, 0.f, 0.f};
        middle.radius = 1.5;

        Sphere bottom;
        bottom.position = {0.f, -1.1f, 0.f};
        bottom.radius = 1.8;

        // assign materials
        head.material = lightMat;
        middle.material = lightMat;
        bottom.material = lightMat;

        head.material = lightMat;
        middle.material = greenMat;
        bottom.material = redMat;

        // add primitives to object
        snowmanObject.add(head);
        snowmanObject.add(middle);
        snowmanObject.add(bottom);

        // add one instance of obj to scene
        snowmanObject.position = {-2, 0, 0};
        scene.add(snowmanObject);

#endif
    }

    // 3 balls
    if (true) {
#if CUDA
        using Geometry = CUDAStruct::Geometry;
        using Sphere = CUDAStruct::SpherePrimitive;

        const bool solar_system = true;
        const bool jonathan_balls = false;
        const bool random_objects = false; // FIXME: fix it

        if (solar_system) { // solar system
            CUDAScenes::solar_system(&scene);
        }

        if (jonathan_balls) { // Jonathan balls

            Geometry object;

            Sphere mirror;
            mirror.position = {-2, 0, 0};

            Sphere tomato;
            tomato.position = {0, -2, 0};

            tomato.radius = 1;

            Sphere watermelon;
            watermelon.position = {2, 0, 0};

            Sphere watermelon2;
            watermelon2.position = {4, 0, 0};

            mirror.mat_albedo = {1, 1, 1};
            mirror.mat_roughness = 0;
            mirror.mat_emissionColor = {0, 1, 1};
            mirror.mat_emissionStrength = 0.5;

            tomato.mat_albedo = {1, 0, 0};
            tomato.mat_roughness = 0.5;
            tomato.mat_emissionColor = {1, 0, 1};
            tomato.mat_emissionStrength = 0.5;

            watermelon.mat_albedo = {0, 1, 0};
            watermelon.mat_roughness = 1;
            watermelon.mat_emissionColor = {1, 1, 0};
            watermelon.mat_emissionStrength = 0.5;

            watermelon2.mat_albedo = {0, 1, 0};
            watermelon2.mat_roughness = 1;
            watermelon2.mat_emissionColor = {1, 1, 1};
            watermelon2.mat_emissionStrength = 0.5;

            object.add(mirror);
            object.add(tomato);
            object.add(watermelon);
            object.add(watermelon2);

            object.position = {0, 0, -1.5};
            scene.add(object);
        }

        if (random_objects) {
            for (int i = 0; i < 3; i++) {
                Geometry randomObject;
                for (int j = 0; j < 3; j++) {

                    Sphere randomSphere;
                    randomSphere.position
                        = {prng(-2, 2), prng(-2, 2), prng(-1, 1)};
                    randomSphere.radius = prng(0.2, 0.5);

                    bool inside_another_sphere = false;
                    // check if sphere is inside of other sphere
                    for (int p = 0; p < scene.num_geometries; p++) {
                        Geometry& _object = scene.geometries[p];
                        for (int k = 0; k < _object.num_spheres; k++) {
                            Sphere otherSphere = _object.spheres[k];
                            float distance = glm::distance(
                                randomSphere.position, otherSphere.position
                            );
                            if (distance < randomSphere.radius
                                               + otherSphere.radius + 1) {
                                inside_another_sphere = true;
                                break;
                            }
                        }
                    }
                    if (inside_another_sphere) {
                        j -= 1; // try again
                        continue;
                    }

                    randomSphere.mat_albedo
                        = {rand() % 70 / 100.f,
                           rand() % 70 / 100.f,
                           rand() % 70 / 100.f};
                    randomSphere.mat_roughness = rand() % 100 / 100.f;
                    randomSphere.mat_emissionColor
                        = {rand() % 100 / 100.f,
                           rand() % 100 / 100.f,
                           rand() % 100 / 100.f};
                    randomSphere.mat_emissionStrength = rand() % 50 / 100.f;
                    randomObject.add(randomSphere);
                }
                randomObject.position
                    = {rand() % 10 - 5, rand() % 10 - 5, rand() % 10 - 5};

                randomObject.position = {0, 0, -1.5};
                scene.add(randomObject);
            }
        }

#else
        Geometry object;

        Sphere mirror;
        mirror.position = {-2, 0, 0};

        Sphere tomato;
        tomato.position = {0, 0, 0};

        Sphere watermelon;
        watermelon.position = {2, 0, 0};
        Sphere watermelon2;
        watermelon2.position = {4, 0, 0};

        mirror.material = glassMat;
        tomato.material = redMat;
        watermelon.material = greenMat;
        watermelon2.material = greenMat;

        object.add(mirror);
        object.add(tomato);
        object.add(watermelon);
        object.add(watermelon2);

        object.position = {0, 0, -1.5};
        scene.add(object);
#endif
    }

    // blue unit sphere
    if (true) {
#if CUDA
#else
        Geometry object;

        Sphere s;
        s.position = {0, 0, 0};
        s.material = floorMat;

        object.add(s);

        object.position = {0, 1, -1.8};
        scene.add(object);

        object.position = {0, 1, 1};
        scene.add(object);
#endif
    }

    Renderer renderer;
    Camera camera;

    // to face -z
    camera.yaw = glm::three_over_two_pi<float>();

    camera.position = {0, 0, 0};
    camera.pitch = 0;

    camera.positionHyp = {0, 0, 0, 1};
    hypCamPosX = 0;
    hypCamPosY = 0;
    hypCamPosZ = 0;
    hypCamPosW = 1;

    sf::Vector2i mPosPrev{sf::Mouse::getPosition()};
    sf::Vector2i mPosCur{sf::Mouse::getPosition()};

    float camMoveSpd{CAM_MED_SPD};

    int tick{};

    if constexpr (INTERACTIVE_MODE)
        std::cout << "Controls:\n"
                  << "Move: WASD LShift Space\n"
                  << "Change Speed: 123\n"
                  << "Accumulate Off/On: -/+\n"
                  << "Respawn: x\n"
                  << "Close: Esc\n"
                  << "Have Fun! (Moving while accumulating is cool)\n";

    // main render loop
    while (!INTERACTIVE_MODE || window->isOpen()) {
        if (window)
            window->update();

        // handle input
        {
            // keyboard input for moving camera pos
            if (EUCLIDEAN) {
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
                if (sf::Keyboard::isKeyPressed(RESPAWN))
                    camera.position = {0.f, 0.f, 0.f};
            } else {
                auto directionToSphericalAngles = [](const glm::vec4& direction
                                                  ) {
                    // Extract the components of the direction vector
                    float dx = direction.x;
                    float dy = direction.y;
                    float dz = direction.z;
                    float dw = direction.w;

                    // Compute the pitch angle
                    float pitch = std::atan2(dz, std::sqrt(dx * dx + dy * dy));

                    // Compute the yaw angle
                    float yaw = std::atan2(dy, dx);

                    return std::make_pair(pitch, yaw);
                };

                if (sf::Keyboard::isKeyPressed(FORWARD)) {
                    const glm::vec4 nextPos{Math::hypGeoFlowPos(
                        camera.positionHyp, camera.forwardDirHyp, camMoveSpd
                    )};
                    const glm::vec4 nextDir{Math::hypGeoFlowDir(
                        camera.positionHyp, camera.forwardDirHyp, camMoveSpd
                    )};
                    camera.positionHyp = nextPos;
                    // camera.forwardDirHyp = nextDir;

                    const auto pitchYaw{directionToSphericalAngles(nextDir)};
                    // camera.pitch = pitchYaw.first;
                    // camera.yaw = pitchYaw.second;

                    // std::cout << "after: \n";
                    // std::cout << "p: " << camera.pitch
                    //     << " y: " << camera.yaw << '\n';
                }
                if (sf::Keyboard::isKeyPressed(BACKWARD)) {
                    const glm::vec4 nextPos{Math::hypGeoFlowPos(
                        camera.positionHyp, -camera.forwardDirHyp, camMoveSpd
                    )};
                    const glm::vec4 nextDir{Math::hypGeoFlowDir(
                        camera.positionHyp, -camera.forwardDirHyp, camMoveSpd
                    )};
                    camera.positionHyp = nextPos;
                    // camera.forwardDirHyp = -nextDir;

                    const auto pitchYaw{directionToSphericalAngles(nextDir)};
                    // camera.pitch = pitchYaw.first;
                    // camera.yaw = pitchYaw.second;
                }

                if (sf::Keyboard::isKeyPressed(RIGHT)) {
                    const glm::vec4 nextPos{Math::hypGeoFlowPos(
                        camera.positionHyp, camera.rightDirHyp, camMoveSpd
                    )};
                    const glm::vec4 nextDir{Math::hypGeoFlowDir(
                        camera.positionHyp, camera.rightDirHyp, camMoveSpd
                    )};
                    camera.positionHyp = nextPos;
                    // camera.rightDirHyp = nextDir;

                    const auto pitchYaw{directionToSphericalAngles(nextDir)};
                    // camera.pitch = pitchYaw.first;
                    // camera.yaw = pitchYaw.second;
                }
                if (sf::Keyboard::isKeyPressed(LEFT)) {
                    const glm::vec4 nextPos{Math::hypGeoFlowPos(
                        camera.positionHyp, -camera.rightDirHyp, camMoveSpd
                    )};
                    const glm::vec4 nextDir{Math::hypGeoFlowDir(
                        camera.positionHyp, -camera.rightDirHyp, camMoveSpd
                    )};
                    camera.positionHyp = nextPos;
                    // camera.rightDirHyp = -nextDir;

                    const auto pitchYaw{directionToSphericalAngles(nextDir)};
                    // camera.pitch = pitchYaw.first;
                    // camera.yaw = pitchYaw.second;
                }
                if (sf::Keyboard::isKeyPressed(UP)) {
                    const glm::vec4 nextPos{Math::hypGeoFlowPos(
                        camera.positionHyp, camera.upDirHyp, camMoveSpd
                    )};
                    const glm::vec4 nextDir{Math::hypGeoFlowDir(
                        camera.positionHyp, camera.upDirHyp, camMoveSpd
                    )};
                    camera.positionHyp = nextPos;
                    // camera.upDirHyp = nextDir;

                    const auto pitchYaw{directionToSphericalAngles(nextDir)};
                    // camera.pitch = pitchYaw.first;
                    // camera.yaw = pitchYaw.second;
                }
                if (sf::Keyboard::isKeyPressed(DOWN)) {
                    const glm::vec4 nextPos{Math::hypGeoFlowPos(
                        camera.positionHyp, -camera.upDirHyp, camMoveSpd
                    )};
                    const glm::vec4 nextDir{Math::hypGeoFlowDir(
                        camera.positionHyp, -camera.upDirHyp, camMoveSpd
                    )};
                    camera.positionHyp = nextPos;
                    // camera.upDirHyp = -nextDir;

                    const auto pitchYaw{directionToSphericalAngles(nextDir)};
                    // camera.pitch = pitchYaw.first;
                    // camera.yaw = pitchYaw.second;
                }
                if (sf::Keyboard::isKeyPressed(RESPAWN)) {
                    camera.positionHyp = {0.f, 0.f, 0.f, 1.f};
                    // camera.yaw = glm::three_over_two_pi<float>();
                    // camera.pitch = 0;
                }

                // std::cout << "p: " << camera.pitch
                //     << " y: " << camera.yaw << '\n';

                camera.positionHyp = Math::correctH3Point(camera.positionHyp);

                // reset camera position if invalid
                {
                    const float x{camera.positionHyp.x};
                    const float y{camera.positionHyp.y};
                    const float z{camera.positionHyp.z};
                    const float w{camera.positionHyp.w};
                    if (std::isinf(x) || std::isnan(x) || std::isinf(y)
                        || std::isnan(y) || std::isinf(z) || std::isnan(z)
                        || std::isinf(w) || std::isnan(w))
                        camera.positionHyp = {0, 0, 0, 1};
                }
                hypCamPosX = camera.positionHyp.x;
                hypCamPosY = camera.positionHyp.y;
                hypCamPosZ = camera.positionHyp.z;
                hypCamPosW = camera.positionHyp.w;
            }

            if (sf::Keyboard::isKeyPressed(CLOSE))
                return 0;
            if (sf::Keyboard::isKeyPressed(ENABLE_ACCUMULATION))
                renderer.accumulate = true;
            if (sf::Keyboard::isKeyPressed(DISABLE_ACCUMULATION)) {
                renderer.accumulate = false;
                renderer.resetAccumulator();
            }
            if (sf::Keyboard::isKeyPressed(CAM_DEBUG)) {
                const auto& c = camera;

                // print pos and euler angles of camera

                if (EUCLIDEAN) {
                    std::cout << "camera.position = {" << c.position.x << ", "
                              << c.position.y << ", " << c.position.z
                              << "};\ncamera.pitch = " << c.pitch
                              << ";\ncamera.yaw = " << c.yaw << ";\n\n";
                } else {
                    std::cout << "camera.positionHyp = {" << c.positionHyp.x
                              << ", " << c.positionHyp.y << ", "
                              << c.positionHyp.z << ", " << c.positionHyp.w
                              << ", "
                              << "};\n"
                              <<

                        "hypCamPosX = " << hypCamPosX << ";\n"
                              << "hypCamPosY = " << hypCamPosY << ";\n"
                              << "hypCamPosZ = " << hypCamPosZ << ";\n"
                              << "hypCamPosW = " << hypCamPosW
                              << ";\ncamera.pitch = " << c.pitch
                              << ";\ncamera.yaw = " << c.yaw << ";\n\n";
                }
            }

            if (sf::Keyboard::isKeyPressed(CAM_SLOW_SPD_KEY))
                camMoveSpd = CAM_SLOW_SPD;
            if (sf::Keyboard::isKeyPressed(CAM_MED_SPD_KEY))
                camMoveSpd = CAM_MED_SPD;
            if (sf::Keyboard::isKeyPressed(CAM_FAST_SPD_KEY))
                camMoveSpd = CAM_FAST_SPD;

            if (sf::Keyboard::isKeyPressed(INC_PITCH))
                camera.pitch += 1 / 8.f;
            if (sf::Keyboard::isKeyPressed(DEC_PITCH))
                camera.pitch -= 1 / 8.f;
            if (sf::Keyboard::isKeyPressed(INC_YAW))
                camera.yaw += 1 / 8.f;
            if (sf::Keyboard::isKeyPressed(DEC_YAW))
                camera.yaw -= 1 / 8.f;

            // mouse input for angling/pointing camera
            mPosPrev = mPosCur;
            mPosCur = sf::Mouse::getPosition();

            const sf::Vector2i mouseMove{mPosCur - mPosPrev};

            camera.yaw += (mouseMove.x * MOUSE_SENSITIVITY) / 256.f;
            camera.pitch += -(mouseMove.y * MOUSE_SENSITIVITY)
                            / 256.f; // subtract so controls aren't inverted
        }

        // std::cout << "isInH3: " << Math::isH3Point(p) << " hypCamPos = {" <<
        // p.x
        //           << ", " << p.y << ", " << p.z << ", " << p.w << "}\n";

        // std::cout << "hyperbolic error: " << hyperbolicErrorAcc << '\n';

        //<< " hypCamDir = {"
        //<< p.x << ", "
        //<< p.y << ", "
        //<< p.z << ", "
        //<< p.w << "}\n";

        /// update
        tick++;
        globalTick++;

        // update scene
        // ...

        // update camera
        const float minPitch{-glm::half_pi<float>() + 0.01f};
        const float maxPitch{glm::half_pi<float>() - 0.01f};
        camera.pitch = glm::clamp(camera.pitch, minPitch, maxPitch);

        camera.updateViewMat();

        if constexpr (!INTERACTIVE_MODE)
            std::cout
                << "Starting render... (be patient this might take a while)\n";

            /// render scene to image
#if CUDA
        float deltaTime = timer.restart().asSeconds();
        scene.tick(deltaTime);
        // rotor.tick(deltaTime);
        RendererCUDA::render(&scene, &camera, &image);
#else
        renderer.render(scene, camera, image);
#endif

        if constexpr (INTERACTIVE_MODE) {
            window->clear();
            window->setBuffer(image);
            window->display();
        } else {
            const float renderTime{timer.getElapsedTime().asSeconds()};

            std::cout << "Render time: " << renderTime << '\n';

            image.saveToFile("render " + std::to_string(renderTime) + "s");

            return 0;
        }
#if CUDA_ONCE
        return 0;
#endif
    }

#if CUDA
    RendererCUDA::cleanup();
#endif

    return 0;
}
