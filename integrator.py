import luisa
from luisa.mathtypes import *
from .camera import generate_ray, tent_warp

bilinear = True

@luisa.func
def read_single_bsdf(coord: int2, material_buffer, texture_res):
    # Address mode: CLAMP
    coord = clamp(coord, int2(0), texture_res-1)
    idx = coord.x + texture_res.x * coord.y
    return float4(
        material_buffer.read(idx * 4 + 0),
        material_buffer.read(idx * 4 + 1),
        material_buffer.read(idx * 4 + 2),
        material_buffer.read(idx * 4 + 3))

@luisa.func
def read_bsdf(uv: float2, material_buffer, texture_res):
    if not bilinear:
        p = float2(uv.x, 1.0-uv.y) * float2(texture_res-1)
        nearest = int2(p+0.4999)
        return read_single_bsdf(nearest, material_buffer, texture_res)
    else:
        p = float2(uv.x, 1.0-uv.y) * float2(texture_res-1)
        ip = int2(p)
        off = p - float2(ip)
        c00 = read_single_bsdf(ip + int2(0,0), material_buffer, texture_res)
        c01 = read_single_bsdf(ip + int2(0,1), material_buffer, texture_res)
        c10 = read_single_bsdf(ip + int2(1,0), material_buffer, texture_res)
        c11 = read_single_bsdf(ip + int2(1,1), material_buffer, texture_res)
        return lerp(lerp(c00, c01, off.y), lerp(c10, c11, off.y), off.x)

@luisa.func
def write_single_bsdf_grad(coord: int2, dmat, d_material_buffer, texture_res):
    # Address mode: CLAMP
    coord = clamp(coord, int2(0), texture_res-1)
    idx = coord.x + texture_res.x * coord.y
    _ = d_material_buffer.atomic_fetch_add(idx * 4 + 0, dmat.x)
    _ = d_material_buffer.atomic_fetch_add(idx * 4 + 1, dmat.y)
    _ = d_material_buffer.atomic_fetch_add(idx * 4 + 2, dmat.z)
    _ = d_material_buffer.atomic_fetch_add(idx * 4 + 3, dmat.w)

@luisa.func
def write_bsdf_grad(uv: float2, dmat, d_material_buffer, texture_res):
    if not bilinear:
        p = float2(uv.x, 1.0-uv.y) * float2(texture_res-1)
        nearest = int2(p+0.4999)
        write_single_bsdf_grad(nearest, dmat, d_material_buffer, texture_res)
    else:
        p = float2(uv.x, 1.0-uv.y) * float2(texture_res-1)
        ip = int2(p)
        off = p - float2(ip)
        k00 = (1-off.x) * (1-off.y)
        k01 = (1-off.x) * off.y
        k10 = off.x * (1-off.y)
        k11 = off.x * off.y
        write_single_bsdf_grad(ip + int2(0,0), k00 * dmat, d_material_buffer, texture_res)
        write_single_bsdf_grad(ip + int2(0,1), k01 * dmat, d_material_buffer, texture_res)
        write_single_bsdf_grad(ip + int2(1,0), k10 * dmat, d_material_buffer, texture_res)
        write_single_bsdf_grad(ip + int2(1,1), k11 * dmat, d_material_buffer, texture_res)


def derive_render_kernel(integrator_func):
    @luisa.func
    def _kernel(image, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel, 
                material_buffer, texture_res, camera, spp, seed, use_tent_filter):
        resolution = dispatch_size().xy
        coord = dispatch_id().xy
        s = float3(0.0)
        for it in range(spp):
            sampler = luisa.util.make_random_sampler3d(int3(int2(coord), seed^(it*5087)))
            pixel_offset = sampler.next2f()
            if use_tent_filter:
                pixel_offset = tent_warp(pixel_offset, 1.0) + float2(0.5)
            pixel = 2.0 / resolution * (float2(coord) + pixel_offset) - 1.0
            ray = generate_ray(camera, pixel)
            radiance = integrator_func(ray, sampler, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel,
                                       material_buffer, texture_res)
            if not any(isnan(radiance)):
                s += radiance
        image.write(coord.x + coord.y * resolution.x, float4(s/spp, 1.0))
    return _kernel

def derive_render_backward_kernel(integrator_backward_func):
    @luisa.func
    def _kernel(d_image, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel, 
                d_material_buffer, material_buffer, texture_res, camera, spp, seed, use_tent_filter):
        resolution = dispatch_size().xy
        coord = dispatch_id().xy
        le_grad = d_image.read(coord.x + coord.y * resolution.x).xyz / spp
        if any(isnan(le_grad)):
            le_grad = float3(0.0)
        for it in range(spp):
            sampler = luisa.util.make_random_sampler3d(int3(int2(coord), seed^(it*5087)))
            pixel_offset = sampler.next2f()
            if use_tent_filter:
                pixel_offset = tent_warp(pixel_offset, 1.0) + float2(0.5)
            pixel = 2.0 / resolution * (float2(coord) + pixel_offset) - 1.0
            ray = generate_ray(camera, pixel)
            integrator_backward_func(ray, sampler, v_buffer, vt_buffer, vn_buffer, triangle_buffer, accel,
                                     d_material_buffer, material_buffer, texture_res, le_grad)
    return _kernel
