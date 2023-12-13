import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf
from .integrator import derive_render_kernel, derive_render_backward_kernel
from .interaction import read_bsdf, write_bsdf_grad, surface_interact
from .light import sample_light

@luisa.func
def direct_estimator(ray, sampler, heap, accel, light_count, material_buffer, texture_res):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return float3(0.0)
    it = surface_interact(hit, heap)
    # backfacing geometry
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return float3(0.0)
    emission = heap.buffer_read(float3, 23333, hit.inst)
    if any(emission > float3(0.0)):
        return emission

    radiance = float3(0.0)
    light = sample_light(it.p, light_count, heap, sampler) # (wi, dist, pdf, eval)
    shadow_ray = luisa.make_ray(it.p, light.wi, 1e-4, light.dist)
    occluded = accel.trace_any(shadow_ray, -1)
    cos_wi_light = dot(light.wi, it.ns)
    # there could still be light from back face due to imperfect occlusion
    if not occluded and cos_wi_light > 0.0:
        mat = read_bsdf(it.uv, material_buffer, texture_res)
        diffuse = mat.xyz
        roughness = mat.w
        specular = 0.04
        bsdf = ggx_brdf(-ray.get_dir(), light.wi, it.ns, diffuse, specular, roughness)
        # mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
        mis_weight = 1.0
        radiance += bsdf * cos_wi_light * mis_weight * light.eval / max(light.pdf, 1e-4)
    return radiance


@luisa.func
def direct_estimator_backward(ray, sampler, heap, accel, light_count,
                               d_material_buffer, material_buffer, texture_res, le_grad):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return
    it = surface_interact(hit, heap)
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return

    light = sample_light(it.p, light_count, heap, sampler) # (wi, dist, pdf, eval)
    shadow_ray = luisa.make_ray(it.p, light.wi, 1e-4, light.dist)
    occluded = accel.trace_any(shadow_ray, -1)
    cos_wi_light = dot(light.wi, it.ns)
    if not occluded:
        mat = read_bsdf(it.uv, material_buffer, texture_res)
        with autodiff():
            requires_grad(mat)
            diffuse = mat.xyz
            roughness = mat.w
            specular = 0.04
            bsdf = ggx_brdf(-ray.get_dir(), light.wi, it.ns, diffuse, specular, roughness)
            # mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
            mis_weight = 1.0
            le = bsdf * cos_wi_light * mis_weight * light.eval / max(light.pdf, 1e-4)
            backward(le, le_grad)
            mat_grad = grad(mat)
    write_bsdf_grad(it.uv, mat_grad, d_material_buffer, texture_res)

render_direct_kernel = derive_render_kernel(direct_estimator)
render_direct_backward_kernel = derive_render_backward_kernel(direct_estimator_backward)