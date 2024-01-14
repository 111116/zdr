import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf
from .integrator import derive_render_kernel, derive_render_backward_kernel
from .interaction import read_bsdf, write_bsdf_grad, surface_interact
from .onb import *


@luisa.func
def collocated_estimator(ray, sampler, heap, accel, light_count, env_count, material_buffer, texture_res):
    """Compute radiance for a ray using collocated estimator.
    A point light source is placed at the camera position.
    All other light sources / envmaps in the scene are ignored.
    """
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return float3(0.0)
    it = surface_interact(hit, heap, accel)
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return float3(0.0)
    mat = read_bsdf(it.uv, material_buffer, texture_res)
    diffuse = mat.xyz
    roughness = mat.w
    specular = 0.04
    onb = make_onb(it.ns)
    wo_local = onb.to_local(-ray.get_dir())
    beta = ggx_brdf(wo_local, wo_local, diffuse, specular, roughness)
    intensity = float3(1.0)
    li = intensity * (1/hit.ray_t)**2
    return beta * li


@luisa.func
def collocated_estimator_backward(ray, sampler, heap, accel, light_count, env_count,
                               d_material_buffer, material_buffer, texture_res, le_grad):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return
    it = surface_interact(hit, heap, accel)
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return
    mat = read_bsdf(it.uv, material_buffer, texture_res)
    with autodiff():
        requires_grad(mat)
        diffuse = mat.xyz
        roughness = mat.w
        specular = 0.04
        onb = make_onb(it.ns)
        wo_local = onb.to_local(-ray.get_dir())
        beta = ggx_brdf(wo_local, wo_local, diffuse, specular, roughness)
        intensity = float3(1.0)
        li = intensity * (1/hit.ray_t)**2
        le = beta * li
        backward(le, le_grad)
        mat_grad = grad(mat)
    write_bsdf_grad(it.uv, mat_grad, d_material_buffer, texture_res)

render_collocated_kernel = derive_render_kernel(collocated_estimator)
render_collocated_backward_kernel = derive_render_backward_kernel(collocated_estimator_backward)