import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf
from .integrator import read_bsdf, write_bsdf_grad, derive_render_kernel, derive_render_backward_kernel

point_light_array = luisa.array([luisa.struct(p=float3(5,4,1), intensity=float3(10.0))])
point_light_count = len(point_light_array)
# env_prob = 0.3 # no environment light yet


# make struct in kernel doesn't work now, so workaround:
LightSampleStruct = luisa.StructType(wi=float3, dist=float, pdf=float, eval=float3)

@luisa.func
def sample_light(origin, sampler):
    # point lights only
    u = sampler.next()
    n = point_light_count
    inst = point_light_array[clamp(int(u * n), 0, n-1)]
    sqr_dist = length_squared(inst.p - origin)
    wi_light = normalize(inst.p - origin)
    pdf = sqr_dist / n
    t = LightSampleStruct()
    t.wi = wi_light
    t.dist = 0.9999*sqrt(sqr_dist)
    t.pdf = pdf
    t.eval = inst.intensity
    return t

@luisa.func
def direct_estimator(ray, sampler, v_buffer, vt_buffer, vn_buffer, triangle_buffer,
                                 accel, material_buffer, texture_res):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return float3(0.0)
    i0 = triangle_buffer.read(hit.prim * 3 + 0)
    i1 = triangle_buffer.read(hit.prim * 3 + 1)
    i2 = triangle_buffer.read(hit.prim * 3 + 2)
    p0 = v_buffer.read(i0)
    p1 = v_buffer.read(i1)
    p2 = v_buffer.read(i2)
    pt0 = vt_buffer.read(i0)
    pt1 = vt_buffer.read(i1)
    pt2 = vt_buffer.read(i2)
    pn0 = vn_buffer.read(i0)
    pn1 = vn_buffer.read(i1)
    pn2 = vn_buffer.read(i2)
    p = hit.interpolate(p0, p1, p2)
    uv = hit.interpolate(pt0, pt1, pt2)
    ns = hit.interpolate(pn0, pn1, pn2)
    ng = normalize(cross(p1 - p0, p2 - p0))
    if dot(-ray.get_dir(), ng) < 1e-4 or dot(-ray.get_dir(), ns) < 1e-4:
        return float3(0.0)

    radiance = float3(0.0)
    light = sample_light(p, sampler) # (wi, dist, pdf, eval)
    shadow_ray = luisa.make_ray(p, light.wi, 1e-4, light.dist)
    occluded = accel.trace_any(shadow_ray, -1)
    cos_wi_light = dot(light.wi, ns)
    if not occluded:
        mat = read_bsdf(uv, material_buffer, texture_res)
        diffuse = mat.xyz
        roughness = mat.w
        specular = 0.04
        bsdf = ggx_brdf(-ray.get_dir(), -ray.get_dir(), ns, diffuse, specular, roughness)
        # mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
        mis_weight = 1.0
        radiance += bsdf * cos_wi_light * mis_weight * light.eval / max(light.pdf, 1e-4)
    return radiance


@luisa.func
def direct_estimator_backward(ray, sampler, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel,
                               d_material_buffer, material_buffer, texture_res, le_grad):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return
    i0 = triangle_buffer.read(hit.prim * 3 + 0)
    i1 = triangle_buffer.read(hit.prim * 3 + 1)
    i2 = triangle_buffer.read(hit.prim * 3 + 2)
    p0 = v_buffer.read(i0)
    p1 = v_buffer.read(i1)
    p2 = v_buffer.read(i2)
    pt0 = vt_buffer.read(i0)
    pt1 = vt_buffer.read(i1)
    pt2 = vt_buffer.read(i2)
    pn0 = vn_buffer.read(i0)
    pn1 = vn_buffer.read(i1)
    pn2 = vn_buffer.read(i2)
    p = hit.interpolate(p0, p1, p2)
    uv = hit.interpolate(pt0, pt1, pt2)
    ns = hit.interpolate(pn0, pn1, pn2)
    ng = normalize(cross(p1 - p0, p2 - p0))
    if dot(-ray.get_dir(), ng) < 1e-4 or dot(-ray.get_dir(), ns) < 1e-4:
        return

    light = sample_light(p, sampler) # (wi, dist, pdf, eval)
    shadow_ray = luisa.make_ray(p, light.wi, 1e-4, light.dist)
    occluded = accel.trace_any(shadow_ray, -1)
    cos_wi_light = dot(light.wi, ns)
    if not occluded:
        mat = read_bsdf(uv, material_buffer, texture_res)
        with autodiff():
            requires_grad(mat)
            diffuse = mat.xyz
            roughness = mat.w
            specular = 0.04
            bsdf = ggx_brdf(-ray.get_dir(), -ray.get_dir(), ns, diffuse, specular, roughness)
            # mis_weight = balanced_heuristic(light.pdf, pdf_bsdf)
            mis_weight = 1.0
            le = bsdf * cos_wi_light * mis_weight * light.eval / max(light.pdf, 1e-4)
            backward(le, le_grad)
            mat_grad = grad(mat)
    write_bsdf_grad(uv, mat_grad, d_material_buffer, texture_res)

render_direct_kernel = derive_render_kernel(direct_estimator)
render_direct_backward_kernel = derive_render_backward_kernel(direct_estimator_backward)