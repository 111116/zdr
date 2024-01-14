import luisa
from luisa.mathtypes import *
from .camera import generate_ray, tent_warp
from .vertex import Vertex

@luisa.func
def compute_dpduv(p0, p1, p2, pt0, pt1, pt2):
    dpde = float3x3(p1-p0, p2-p0, float3(0))
    duvde = float2x2(pt1-pt0, pt2-pt0)
    # TODO Check for degenerate UV coordinates
    # if abs(determinant(dUVde)) < 1e-6:
        # pass
    deduv = inverse(duvde)
    dpduv = dpde * float3x3(deduv)
    dpduv[1] *= -1 # inverted v
    return dpduv # [dpdu dpdv 0]

@luisa.func
def trace_duvdxy(ray, ray_dx, ray_dy, heap, accel):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return float4(0.0)
    i0 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 0)
    i1 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 1)
    i2 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 2)
    v0 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i0)
    v1 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i1)
    v2 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i2)
    p0 = v0.v()
    p1 = v1.v()
    p2 = v2.v()
    pt0 = v0.vt()
    pt1 = v1.vt()
    pt2 = v2.vt()
    p = hit.interpolate(p0, p1, p2)
    dpduv = compute_dpduv(p0, p1, p2, pt0, pt1, pt2)
    # use offset rays to compute finite difference of p
    ng = normalize(cross(p1 - p0, p2 - p0))
    t_dx = dot(p-ray_dx.get_origin(), ng) / dot(ray_dx.get_dir(), ng)
    t_dy = dot(p-ray_dy.get_origin(), ng) / dot(ray_dy.get_dir(), ng)
    p_dx = ray_dx.get_origin() + t_dx * ray_dx.get_dir()
    p_dy = ray_dy.get_origin() + t_dy * ray_dy.get_dir()
    dpdx = p_dx - p
    dpdy = p_dy - p
    iaTa = inverse(float2x2(transpose(dpduv) * dpduv))
    iaTaaT = float3x3(iaTa) * transpose(dpduv)
    duvdx = (iaTaaT * dpdx).xy
    duvdy = (iaTaaT * dpdy).xy
    return float4(duvdx, duvdy)

    # --------- Mitsuba3 implementation ---------
    # dpdu = dpduv[0]
    # dpdv = dpduv[1]
    # a00 = dot(dpdu, dpdu)
    # a01 = dot(dpdu, dpdv)
    # a11 = dot(dpdv, dpdv)
    # inv_det = 1/(a00*a11+a01*a01)

    # b0x = dot(dpdu, dpdx)
    # b1x = dot(dpdv, dpdx)
    # b0y = dot(dpdu, dpdy)
    # b1y = dot(dpdv, dpdy)

    # # Set the UV partials to zero if dpdu and/or dpdv == 0
    # inv_det = 0.0 if isinf(inv_det) else inv_det

    # duvdx = float2(a11 * b0x - a01 * b1x,
    #                a00 * b1x - a01 * b0x) * inv_det

    # duvdy = float2(a11 * b0y - a01 * b1y,
    #                a00 * b1y - a01 * b0y) * inv_det
    # --------- ----------------------- ---------


@luisa.func
def render_uvgrad_kernel(image, heap, accel, light_count, env_count,
                         material_buffer, texture_res, camera, spp, seed, use_tent_filter):
    resolution = dispatch_size().xy
    coord = dispatch_id().xy
    s = float4(0.0)
    for it in range(spp):
        sampler = luisa.util.make_random_sampler3d(int3(int2(coord), seed^(it*5087)))
        pixel_offset = sampler.next2f()
        if use_tent_filter:
            pixel_offset = tent_warp(pixel_offset, 1.0) + float2(0.5)
        pixel = 2.0 / resolution * (float2(coord) + pixel_offset) - 1.0
        pixel_dx = 2.0 / resolution * (float2(coord) + pixel_offset + float2(1, 0)) - 1.0
        pixel_dy = 2.0 / resolution * (float2(coord) + pixel_offset + float2(0, 1)) - 1.0
        ray = generate_ray(camera, pixel)
        ray_dx = generate_ray(camera, pixel_dx)
        ray_dy = generate_ray(camera, pixel_dy)
        uvgrad = trace_duvdxy(ray, ray_dx, ray_dy, heap, accel)
        if not any(isnan(uvgrad)):
            s += uvgrad
    image.write(coord.x + coord.y * resolution.x, s/spp)
