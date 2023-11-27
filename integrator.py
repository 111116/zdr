import luisa
from luisa.mathtypes import *
from luisa.autodiff import requires_grad, autodiff, backward, grad
from .microfacet import ggx_brdf
from .camera import generate_ray

@luisa.func
def get_uv_coord(uv: float2, texture_res: int2):
    p = float2(uv.x, 1.0-uv.y) * float2(texture_res-1)
    ip = int2(p)
    off = p - float2(ip)
    # TODO boundary check
    nearest = int2(p+0.499)
    return nearest.x + texture_res.x * nearest.y

@luisa.func
def read_bsdf(uv: float2, material_buffer, texture_res):
    coord = get_uv_coord(uv, texture_res)
    return float4(
        material_buffer.read(coord * 4 + 0),
        material_buffer.read(coord * 4 + 1),
        material_buffer.read(coord * 4 + 2),
        material_buffer.read(coord * 4 + 3))

@luisa.func
def write_bsdf_grad(uv: float2, dmat, d_material_buffer, texture_res):
    coord = get_uv_coord(uv, texture_res)
    _ = d_material_buffer.atomic_fetch_add(coord * 4 + 0, dmat.x)
    _ = d_material_buffer.atomic_fetch_add(coord * 4 + 1, dmat.y)
    _ = d_material_buffer.atomic_fetch_add(coord * 4 + 2, dmat.z)
    _ = d_material_buffer.atomic_fetch_add(coord * 4 + 3, dmat.w)

@luisa.func
def direct_collocated(ray, v_buffer, vt_buffer, vn_buffer, triangle_buffer,
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
def direct_collocated_backward(ray, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel,
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
    

@luisa.func
def render_kernel(image, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel, 
                  material_buffer, texture_res, camera, spp, seed):
    resolution = dispatch_size().xy
    coord = dispatch_id().xy
    # TODO spp
    sampler = luisa.util.make_random_sampler3d(int3(int2(coord), seed))
    pixel = 2.0 / resolution * (float2(coord) + sampler.next2f()) - 1.0
    ray = generate_ray(camera, pixel)
    radiance = direct_collocated(ray, v_buffer, vt_buffer, vn_buffer, triangle_buffer,
                                 accel, material_buffer, texture_res)
    if any(isnan(radiance)):
        radiance = float3(0.0)
    image.write(coord.x + coord.y * resolution.x, float4(radiance, 1.0))

@luisa.func
def render_backward_kernel(d_image, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel, 
                    d_material_buffer, material_buffer, texture_res, camera, spp, seed):
    resolution = dispatch_size().xy
    coord = dispatch_id().xy
    # TODO spp
    sampler = luisa.util.make_random_sampler3d(int3(int2(coord), seed))
    pixel = 2.0 / resolution * (float2(coord) + sampler.next2f()) - 1.0
    ray = generate_ray(camera, pixel)
    le_grad = d_image.read(coord.x + coord.y * resolution.x).xyz
    if any(isnan(le_grad)):
        le_grad = float3(0.0)
    direct_collocated_backward(ray, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel,
                               d_material_buffer, material_buffer, texture_res, le_grad)
