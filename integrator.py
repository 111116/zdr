import luisa
from luisa.mathtypes import *
from .camera import generate_ray, tent_warp
from .pmj02bn import *
from .corrmj import *


def derive_render_kernel(integrator_func):
    @luisa.func
    def _kernel(image, heap, accel, light_count, env_count,
                material_buffer, texture_res, camera, spp, seed, use_tent_filter):
        resolution = dispatch_size().xy
        coord = dispatch_id().xy
        s = float3(0.0)
        for it in range(spp):
            # sampler = make_corrmj_sampler(int2(coord), seed, spp, it)
            sampler = make_pmj02bn_sampler(int2(coord), seed, spp, it)
            # sampler = luisa.util.make_random_sampler3d(int3(int2(coord), seed^(it*987654347)))
            pixel_offset = sampler.next2f()
            if use_tent_filter:
                pixel_offset = tent_warp(pixel_offset, 1.0) + float2(0.5)
            pixel = 2.0 / resolution * (float2(coord) + pixel_offset) - 1.0
            ray = generate_ray(camera, pixel)
            radiance = integrator_func(ray, sampler, heap, accel, light_count, env_count,
                                       material_buffer, texture_res)
            if not any(isnan(radiance)):
                s += clamp(radiance, 0.0, 100000.0)
        image.write(coord.x + coord.y * resolution.x, float4(s/spp, 1.0))
    return _kernel

def derive_render_backward_kernel(integrator_backward_func):
    @luisa.func
    def _kernel(d_image, heap, accel, light_count, env_count,
                d_material_buffer, material_buffer, texture_res, camera, spp, seed, use_tent_filter):
        resolution = dispatch_size().xy
        coord = dispatch_id().xy
        le_grad = d_image.read(coord.x + coord.y * resolution.x).xyz / spp
        if any(isnan(le_grad)):
            le_grad = float3(0.0)
        for it in range(spp):
            # sampler = make_corrmj_sampler(int2(coord), seed, spp, it)
            sampler = make_pmj02bn_sampler(int2(coord), seed, spp, it)
            # sampler = luisa.util.make_random_sampler3d(int3(int2(coord), seed^(it*987654347)))
            pixel_offset = sampler.next2f()
            if use_tent_filter:
                pixel_offset = tent_warp(pixel_offset, 1.0) + float2(0.5)
            pixel = 2.0 / resolution * (float2(coord) + pixel_offset) - 1.0
            ray = generate_ray(camera, pixel)
            integrator_backward_func(ray, sampler, heap, accel, light_count, env_count,
                                     d_material_buffer, material_buffer, texture_res, le_grad)
    return _kernel
