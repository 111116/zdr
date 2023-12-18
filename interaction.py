import luisa
from luisa.mathtypes import *
from .vertex import Vertex


Interaction = luisa.StructType(p=float3, uv=float2, ns=float3, ng=float3)

@luisa.func
def surface_interact(hit, heap, accel):
    # read vertex info from bindless array
    i0 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 0)
    i1 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 1)
    i2 = heap.buffer_read(int, hit.inst * 2, hit.prim * 3 + 2)
    v0 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i0)
    v1 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i1)
    v2 = heap.buffer_read(Vertex, hit.inst * 2 + 1, i2)
    # apply transform 
    transform = accel.instance_transform(hit.inst)
    p0 = (transform * float4(v0.v(), 1.0)).xyz
    p1 = (transform * float4(v1.v(), 1.0)).xyz
    p2 = (transform * float4(v2.v(), 1.0)).xyz
    # compute hit point
    it = Interaction()
    it.p = hit.interpolate(p0, p1, p2)
    it.uv = hit.interpolate(v0.vt(), v1.vt(), v2.vt())
    ns0 = hit.interpolate(v0.vn(), v1.vn(), v2.vn())
    it.ns = normalize(inverse(transpose(make_float3x3(transform))) * ns0)
    it.ng = normalize(cross(p1 - p0, p2 - p0))
    return it


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
