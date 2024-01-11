import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf, ggx_sample, ggx_sample_pdf
from .integrator import derive_render_kernel, derive_render_backward_kernel
from .interaction import read_bsdf, write_bsdf_grad, surface_interact
from .light import sample_light, sample_light_pdf
from .onb import *
from .envmap import direction_to_uv, uv_to_direction

# MIS off: Only draw light samples. Good for small lights.
# MIS on: Draw light & bsdf samples, at cost of ~2.6x computation.
#         Useful for large lights / glossy surfaces.
use_MIS = True

@luisa.func
def balanced_heuristic(pdf_a, pdf_b):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)

@luisa.func
def direct_estimator(ray, sampler, heap, accel, light_count, material_buffer, texture_res):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        envmap_uv = direction_to_uv(ray.get_dir())
        return heap.texture2d_sample(23332, envmap_uv).xyz
        return float3(0.0)
    it = surface_interact(hit, heap, accel)
    # backfacing geometry
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return float3(0.0)
    emission = heap.buffer_read(float3, 23333, hit.inst)
    if hit.inst > 0: # treat everything as light other than the first object
    # if any(emission > float3(0.0)):
        return emission

    # fetch texture
    mat = read_bsdf(it.uv, material_buffer, texture_res)
    diffuse = mat.xyz
    roughness = mat.w
    specular = 0.04

    # direct light sample
    radiance = float3(0.0)
    # light = sample_light(it.p, light_count, heap, accel, sampler) # (wi, dist, pdf, eval)
    # shadow_ray = luisa.make_ray(it.p, light.wi, 1e-4, light.dist)
    # occluded = accel.trace_any(shadow_ray, -1)
    onb = make_onb(it.ns)
    wo_local = onb.to_local(-ray.get_dir())
    # wi_light_local = onb.to_local(light.wi)
    # # Discard light from back face due to imperfect occlusion / non-manifolds
    # if not occluded and wi_light_local.z > 0.0:
    #     bsdf = ggx_brdf(wo_local, wi_light_local, diffuse, specular, roughness)
    #     if use_MIS:
    #         pdf_bsdf = ggx_sample_pdf(wo_local, wi_light_local, diffuse, specular, roughness)
    #         mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
    #     else:
    #         mis_weight = 1.0
    #     radiance += bsdf * mis_weight * light.eval / max(light.pdf, 1e-4)

    if use_MIS:
        # pdf sample (next bounce)
        wi_local = ggx_sample(wo_local, diffuse, specular, roughness, sampler)
        pdf_bsdf = ggx_sample_pdf(wo_local, wi_local, diffuse, specular, roughness)
        wi = onb.to_world(wi_local)
        if dot(wi, it.ng) < 1e-4 or wi_local.z < 1e-4:
            return radiance
        ray = luisa.make_ray(luisa.offset_ray_origin(it.p, it.ng), wi, 0.0, 1e30)
        beta = ggx_brdf(wo_local, wi_local, diffuse, specular, roughness) / pdf_bsdf
        # Just repeat tracing...
        origin = it.p
        hit = accel.trace_closest(ray, -1)
        if hit.miss():
            # return radiance
            emission = heap.texture2d_sample(23332, direction_to_uv(ray.get_dir())).xyz
        else:
            it = surface_interact(hit, heap, accel)
            # backfacing geometry
            if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
                return radiance
            emission = heap.buffer_read(float3, 23333, hit.inst)
        if any(emission > float3(0.0)):
            pdf_light = sample_light_pdf(origin, light_count, heap, accel, hit.inst, hit.prim, it.p)
            mis_weight = balanced_heuristic(pdf_bsdf, pdf_light)
            mis_weight = 1.0 # TODO
            return radiance + beta * mis_weight * emission

    return radiance


@luisa.func
def direct_estimator_backward(ray, sampler, heap, accel, light_count,
                               d_material_buffer, material_buffer, texture_res, le_grad):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return
    it = surface_interact(hit, heap, accel)
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return
    emission = heap.buffer_read(float3, 23333, hit.inst)
    if hit.inst > 0: # treat everything as light other than the first object
    # if any(emission > float3(0.0)):
        return

    # direct light sample
    light = sample_light(it.p, light_count, heap, accel, sampler) # (wi, dist, pdf, eval)
    shadow_ray = luisa.make_ray(it.p, light.wi, 1e-4, light.dist)
    occluded = accel.trace_any(shadow_ray, -1)
    cos_wi_light = dot(light.wi, it.ns)
    onb = make_onb(it.ns)
    if not occluded and cos_wi_light > 0.0:
        mat = read_bsdf(it.uv, material_buffer, texture_res)
        with autodiff():
            # differentiate w.r.t. material
            requires_grad(mat)
            diffuse = mat.xyz
            roughness = mat.w
            specular = 0.04
            bsdf = ggx_brdf(onb.to_local(-ray.get_dir()), onb.to_local(light.wi), diffuse, specular, roughness)
            mis_weight = 1.0
            le = bsdf * mis_weight * light.eval / max(light.pdf, 1e-4)
            backward(le, le_grad)
            mat_grad = grad(mat)
    write_bsdf_grad(it.uv, mat_grad, d_material_buffer, texture_res)

render_direct_kernel = derive_render_kernel(direct_estimator)
render_direct_backward_kernel = derive_render_backward_kernel(direct_estimator_backward)