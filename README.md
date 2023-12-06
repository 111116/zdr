# zdr

Simple raytracing differentiable renderer (w.r.t. material parameters). Images and parameters are both PyTorch tensors.

Currently only supports single object, collocated direct lighting (point light at position of the camera).

## Dependency

[LuisaCompute](https://github.com/LuisaGroup/LuisaCompute), which requires C++20 and Nvidia driver R535+.

## Usage

See `example.py`
