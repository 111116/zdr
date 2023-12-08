import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf
from .integrator import read_bsdf, write_bsdf_grad, derive_render_kernel, derive_render_backward_kernel
from .vertex import Vertex

Interaction = luisa.StructType(p=float3, uv=float2, ns=float3, ng=float3)

@luisa.func
def surface_interact(hit, heap):
    i0 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 0)
    i1 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 1)
    i2 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 2)
    p0 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i0)
    p1 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i1)
    p2 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i2)
    it = Interaction()
    it.p = hit.interpolate(p0.v(), p1.v(), p2.v())
    it.uv = hit.interpolate(p0.vt(), p1.vt(), p2.vt())
    it.ns = hit.interpolate(p0.vn(), p1.vn(), p2.vn())
    it.ng = normalize(cross(p1.v() - p0.v(), p2.v() - p0.v()))
    return it


@luisa.func
def collocated_estimator(ray, sampler, heap, accel, material_buffer, texture_res):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return float3(0.0)
    it = surface_interact(hit, heap)
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return float3(0.0)
    mat = read_bsdf(it.uv, material_buffer, texture_res)
    diffuse = mat.xyz
    roughness = mat.w
    specular = 0.04
    beta = ggx_brdf(-ray.get_dir(), -ray.get_dir(), it.ns, diffuse, specular, roughness)
    intensity = float3(1.0)
    li = intensity * (1/hit.ray_t)**2
    return beta * li


@luisa.func
def collocated_estimator_backward(ray, sampler, heap, accel,
                               d_material_buffer, material_buffer, texture_res, le_grad):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return
    it = surface_interact(hit, heap)
    if dot(-ray.get_dir(), it.ng) < 1e-4 or dot(-ray.get_dir(), it.ns) < 1e-4:
        return
    mat = read_bsdf(it.uv, material_buffer, texture_res)
    with autodiff():
        requires_grad(mat)
        diffuse = mat.xyz
        roughness = mat.w
        specular = 0.04
        beta = ggx_brdf(-ray.get_dir(), -ray.get_dir(), it.ns, diffuse, specular, roughness)
        intensity = float3(1.0)
        li = intensity * (1/hit.ray_t)**2
        le = beta * li
        backward(le, le_grad)
        mat_grad = grad(mat)
    write_bsdf_grad(it.uv, mat_grad, d_material_buffer, texture_res)

render_collocated_kernel = derive_render_kernel(collocated_estimator)
render_collocated_backward_kernel = derive_render_backward_kernel(collocated_estimator_backward)