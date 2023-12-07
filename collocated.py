import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf
from .integrator import read_bsdf, write_bsdf_grad, derive_render_kernel, derive_render_backward_kernel

@luisa.func
def collocated_estimator(ray, sampler, v_buffer, vt_buffer, vn_buffer, triangle_buffer,
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
    # return float3(uv, 0.5)
    mat = read_bsdf(uv, material_buffer, texture_res)
    diffuse = mat.xyz
    roughness = mat.w
    specular = 0.04
    beta = ggx_brdf(-ray.get_dir(), -ray.get_dir(), ns, diffuse, specular, roughness)
    intensity = float3(1.0)
    li = intensity * (1/hit.ray_t)**2
    return beta * li


@luisa.func
def collocated_estimator_backward(ray, sampler, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel,
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
    # return float3(uv, 0.5)
    mat = read_bsdf(uv, material_buffer, texture_res)
    with autodiff():
        requires_grad(mat)
        diffuse = mat.xyz
        roughness = mat.w
        specular = 0.04
        beta = ggx_brdf(-ray.get_dir(), -ray.get_dir(), ns, diffuse, specular, roughness)
        intensity = float3(1.0)
        li = intensity * (1/hit.ray_t)**2
        le = beta * li
        backward(le, le_grad)
        mat_grad = grad(mat)
    write_bsdf_grad(uv, mat_grad, d_material_buffer, texture_res)

render_collocated_kernel = derive_render_kernel(collocated_estimator)
render_collocated_backward_kernel = derive_render_backward_kernel(collocated_estimator_backward)