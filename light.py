import luisa
from luisa.mathtypes import *
from .vertex import Vertex

point_light_array = luisa.array([luisa.struct(p=float3(-0.2, 5, -3), intensity=float3(10.0))])
# point_light_count = len(point_light_array) # TODO
point_light_count = 0
# env_prob = 0.3 # no environment light yet

# make struct in kernel doesn't work now, so workaround:
LightSampleStruct = luisa.StructType(wi=float3, dist=float, pdf=float, eval=float3)


@luisa.func
def sample_uniform_triangle(u: float2):
    uv = make_float2(0.5 * u.x, -0.5 * u.x + u.y) \
         if u.x < u.y else \
         make_float2(-0.5 * u.y + u.x, 0.5 * u.y)
    return make_float3(uv, 1.0 - uv.x - uv.y)

@luisa.func
def sample_light(origin, light_count, heap, sampler):
    u = sampler.next()
    mesh_light_count = light_count
    n = point_light_count + mesh_light_count
    # TODO clean up code & put point light source from scene
    idx = clamp(int(u * n), 0, n-1)
    if idx < point_light_count:
        # sample from point light sources
        pointlight = point_light_array[idx]
        sqr_dist = length_squared(pointlight.p - origin)
        wi_light = normalize(pointlight.p - origin)
        pdf = sqr_dist / n
        t = LightSampleStruct()
        t.wi = wi_light
        t.dist = 0.9999*sqrt(sqr_dist)
        t.pdf = pdf
        t.eval = pointlight.intensity
        return t
    else:
        # sample from mesh light sources
        idx -= point_light_count
        inst = heap.buffer_read(int, 23334, idx)
        trig_count = heap.buffer_read(int, 23335, inst)
        prim = clamp(int(sampler.next() * trig_count), 0, trig_count-1)
        # fetch sampled primitive (triangle)
        i0 = heap.buffer_read(int, inst * 2, prim * 3 + 0)
        i1 = heap.buffer_read(int, inst * 2, prim * 3 + 1)
        i2 = heap.buffer_read(int, inst * 2, prim * 3 + 2)
        p0 = heap.buffer_read(Vertex, inst * 2 + 1, i0).v()
        p1 = heap.buffer_read(Vertex, inst * 2 + 1, i1).v()
        p2 = heap.buffer_read(Vertex, inst * 2 + 1, i2).v()
        # apply transform TODO
        # transform = accel.instance_transform(inst)
        # p0 = (transform * float4(p0, 1.0)).xyz
        # p1 = (transform * float4(p1, 1.0)).xyz
        # p2 = (transform * float4(p2, 1.0)).xyz
        abc = sample_uniform_triangle(sampler.next2f())
        p = abc.x * p0 + abc.y * p1 + abc.z * p2 # point on light
        emission = heap.buffer_read(float3, 23333, inst)
        # calculating pdf (avoid calling mesh_light_sampled_pdf to save some redundant computation)
        wi_light = normalize(p - origin)
        c = cross(p1 - p0, p2 - p0)
        light_normal = normalize(c)
        cos_light = -dot(light_normal, wi_light)
        emission = emission if cos_light > 1e-4 else float3(0)
        sqr_dist = length_squared(p - origin)
        area = length(c) / 2
        pdf = sqr_dist / (n * trig_count * area * cos_light)
        # return as struct
        t = LightSampleStruct()
        t.wi = wi_light
        t.dist = 0.9999*sqrt(sqr_dist)
        t.pdf = pdf
        t.eval = emission
        return t
