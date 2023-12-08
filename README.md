# zdr

Simple raytracing differentiable renderer (w.r.t. material parameters). Images and parameters are both PyTorch tensors.

Currently only supports single object, collocated direct lighting (point light at position of the camera).

Note: when optimized textures aren't spatially varying, or contentrate on a tiny portion of the texture buffer, the atomic_fetch_add will become extremely slow. Need to add duplicate buffer for that.

UPD: polymorphic material eval not implemented yet.

Current state:

    - emission=0 -> material Microfacet(d, 0.04, r)

    - emission>0 -> light (material=None)

## Dependency

[LuisaCompute](https://github.com/LuisaGroup/LuisaCompute), which requires C++20, Python>=3.10 and Nvidia driver R535+.

## Usage

See `example.py`
