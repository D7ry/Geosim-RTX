# Geosim-RTX

CUDA-based real-time ray marcher for physically-based rendering of geometries in H3 space.

## Results

![solar system](results/solar_system.gif)

## Build

### Requirements

- `CUDA 11.8` or above && `nvcc`
- `CMake`
- `make`
- `gcc`
- `glm`

```bash
mkdir build
cd build
cmake ..
make
```


## Reference

[Ray-marching Thurston geometries](https://arxiv.org/abs/2010.15801)

[Non-euclidean virtual reality I: explorations of ℍ3](https://arxiv.org/abs/1702.04004)

