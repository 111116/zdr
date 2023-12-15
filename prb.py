import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf, ggx_sample, ggx_sample_pdf
from .integrator import derive_render_kernel, derive_render_backward_kernel
from .interaction import read_bsdf, write_bsdf_grad, surface_interact
from .light import sample_light, sample_light_pdf
from .onb import *

@luisa.func
def balanced_heuristic(pdf_a, pdf_b):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)

max_depth = 5
rr_depth = 2

@luisa.func
def path_estimator(ray, sampler, heap, accel, light_count, material_buffer, texture_res):
    radiance = float3(0.0) # path accumulated Le
    beta = float3(1.0) # path throughput
    pdf_bsdf = 1e30 # current angular sample density
    for depth in range(max_depth):
        # ray intersection
        hit = accel.trace_closest(ray, -1)
        if hit.miss():
            break
        it = surface_interact(hit, heap)
        # discard backfacing geometry
        if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
            return float3(0.0)

        # hit light?
        emission = heap.buffer_read(float3, 23333, hit.inst)
        if any(emission > float3(0.0)):
            pdf_light = sample_light_pdf(ray.get_origin(), light_count, heap, hit.inst, hit.prim, it.p)
            mis_weight = balanced_heuristic(pdf_bsdf, pdf_light)
            radiance += beta * mis_weight * emission
            break

        # fetch material texture
        mat = read_bsdf(it.uv, material_buffer, texture_res)
        diffuse = mat.xyz
        roughness = mat.w
        specular = 0.04

        onb = make_onb(it.ns)
        wo_local = onb.to_local(-ray.get_dir())
        # draw light sample
        light = sample_light(it.p, light_count, heap, sampler) # (wi, dist, pdf, eval)
        shadow_ray = luisa.make_ray(it.p, light.wi, 1e-4, light.dist)
        occluded = accel.trace_any(shadow_ray, -1)
        wi_light_local = onb.to_local(light.wi)
        # Discard light from back face due to imperfect occlusion / non-manifolds
        if not occluded and wi_light_local.z >= 1e-4:
            bsdf = ggx_brdf(wo_local, wi_light_local, diffuse, specular, roughness)
            pdf_bsdf = ggx_sample_pdf(wo_local, wi_light_local, diffuse, specular, roughness)
            mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
            radiance += beta * bsdf * mis_weight * light.eval / max(light.pdf, 1e-4)

        # draw bsdf sample
        wi_local = ggx_sample(wo_local, diffuse, specular, roughness, sampler)
        pdf_bsdf = ggx_sample_pdf(wo_local, wi_local, diffuse, specular, roughness)
        wi = onb.to_world(wi_local)
        # discard bounce that points into surface
        if dot(wi, it.ng) < 1e-4 or wi_local.z < 1e-4:
            break
        ray = luisa.make_ray(luisa.offset_ray_origin(it.p, it.ng), wi, 0.0, 1e30)
        beta *= ggx_brdf(wo_local, wi_local, diffuse, specular, roughness) / pdf_bsdf

        # Russian roulette
        if depth >= rr_depth:
            l = dot(float3(0.212671, 0.715160, 0.072169), beta)
            if l == 0.0:
                break
            q = max(l, 0.05)
            r = sampler.next()
            if r >= q:
                break
            beta /= q

    return radiance


@luisa.func
def path_estimator_backward(ray, sampler, heap, accel, light_count,
                            d_material_buffer, material_buffer, texture_res, le_grad):
    pass


render_path_kernel = derive_render_kernel(path_estimator)
render_path_backward_kernel = derive_render_backward_kernel(path_estimator_backward)