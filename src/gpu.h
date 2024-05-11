namespace CudaPlayground {
    void play();
}


class Scene;
class Camera;
class Image;

namespace RendererCUDA {
    void render(const Scene* scene, const Camera* camera, Image* image);
} // namespace Renderer
