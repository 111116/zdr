import torch
import luisa
from luisa.mathtypes import *
import numpy as np
import microfacet
from load_obj import read_obj, concat_triangles
from recompute_normal import recompute_normal

@luisa.func
def generate_ray(p):
    fov = 40 / 180 * 3.1415926
    origin = float3(1.0, 0.5, 0.0)
    target = float3(0.0, 0.0, 0.0)
    up = float3(0.0, 1.0, 0.0)
    forward = normalize(target - origin)
    right = normalize(cross(forward, up))
    up_perp = cross(right, forward)
    
    p = p * tan(0.5 * fov)
    direction = normalize(p.x * right - p.y * up_perp + forward)
    return luisa.make_ray(origin, direction, 0.0, 1e30)

@luisa.func
def get_uv_coord(uv: float2):
    p = float2(uv.x, 1.0-uv.y) * float2(texture_resolution-1)
    ip = int2(p)
    off = p - float2(ip)
    # TODO boundary check
    nearest = int2(p+0.499)
    return nearest.x + texture_resolution.x * nearest.y

@luisa.func
def read_bsdf(uv: float2):
    coord = get_uv_coord(uv)
    return float4(
        material_buffer.read(coord * 4 + 0),
        material_buffer.read(coord * 4 + 1),
        material_buffer.read(coord * 4 + 2),
        material_buffer.read(coord * 4 + 3))

@luisa.func
def write_bsdf_grad(uv: float2, dmat):
    coord = get_uv_coord(uv)
    material_grad_buffer.atomic_fetch_add(coord * 4 + 0, dmat.x)
    material_grad_buffer.atomic_fetch_add(coord * 4 + 1, dmat.y)
    material_grad_buffer.atomic_fetch_add(coord * 4 + 2, dmat.z)
    material_grad_buffer.atomic_fetch_add(coord * 4 + 3, dmat.w)

@luisa.func
def direct_collocated(ray):
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
    mat = read_bsdf(uv)
    diffuse = mat.xyz
    roughness = mat.w
    specular = 0.04
    beta = microfacet.ggx_brdf(-ray.get_dir(), -ray.get_dir(), ns, diffuse, specular, roughness)
    intensity = float3(1.0)
    li = intensity * (1/hit.ray_t)**2
    return beta * li


@luisa.func
def direct_collocated_backward(ray, le_grad):
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
    mat = read_bsdf(uv)
    with autodiff():
        requires_grad(mat)
        diffuse = mat.xyz
        roughness = mat.w
        specular = 0.04
        beta = microfacet.ggx_brdf(-ray.get_dir(), -ray.get_dir(), ns, diffuse, specular, roughness)
        intensity = float3(1.0)
        li = intensity * (1/hit.ray_t)**2
        le = beta * li
        backward(le, le_grad)
        mat_grad = grad(mat)
    

@luisa.func
def render(image, resolution, frame_id):
    coord = dispatch_id().xy
    sampler = luisa.util.make_random_sampler3d(int3(int2(coord), frame_id))
    pixel = 2.0 / resolution * (float2(coord) + sampler.next2f()) - 1.0
    ray = generate_ray(pixel)
    radiance = direct_collocated(ray)
    if any(isnan(radiance)):
        radiance = float3(0.0)
    image.write(coord, float4(radiance, 1.0))

@luisa.func
def render_backward(d_image, resolution, frame_id):
    coord = dispatch_id().xy
    sampler = luisa.util.make_random_sampler3d(int3(int2(coord), frame_id))
    pixel = 2.0 / resolution * (float2(coord) + sampler.next2f()) - 1.0
    ray = generate_ray(pixel)
    le_grad = d_image.read(coord).xyz
    if any(isnan(le_grad)):
        le_grad = float3(0.0)
    direct_collocated_backward(ray, le_grad)


# load shape from obj file
file_path = 'assets/sphere.obj'
positions, tex_coords, normals, faces = read_obj(file_path)
# upload shapes
luisa.init()

def float3list_to_padded_tensor(l):
    a = torch.tensor(l, dtype=torch.float32, device='cuda')
    assert a.dim()==2 and a.shape[1]==3
    n = a.shape[0]
    b = torch.empty((n, 1), dtype=torch.float32, device='cuda')
    w = torch.hstack((a,b))
    return w.as_strided(size=(n,3), stride=(4,1))

v_buffer = luisa.Buffer.from_dlpack(float3list_to_padded_tensor(positions))
vt_buffer = luisa.Buffer.from_dlpack(torch.tensor(tex_coords, dtype=torch.float32, device='cuda'))
vn_buffer = luisa.Buffer.from_dlpack(float3list_to_padded_tensor(normals))
triangle_buffer = luisa.buffer(concat_triangles(faces))
recompute_normal(v_buffer, vn_buffer, triangle_buffer)
accel = luisa.Accel()
accel.add(v_buffer, triangle_buffer)
accel.update()


# read material maps
def np_from_image(file, n_channels):
    arr = luisa.lcapi.load_ldr_image(file)
    assert len(arr.shape) == 3 and arr.shape[2] == 4
    return arr[..., 0:n_channels]
# diffuse_arr = np_from_image('assets/wood-01-1k/diffuse.jpg', 3)
# roughness_arr = np_from_image('assets/wood-01-1k/roughness.jpg', 1)
diffuse_arr = np_from_image('assets/wood_olive/wood_olive_wood_olive_basecolor.png', 3)
roughness_arr = np_from_image('assets/wood_olive/wood_olive_wood_olive_roughness.png', 1)
arr = np.concatenate((diffuse_arr, roughness_arr), axis=2)
# row, column, 4 floats (diffuse + roughness)
texture_resolution = int2(*arr.shape[0:2])
arr = ((arr.astype('float32')/255)**2.2).flatten()
material_buffer = luisa.buffer(arr)



# render image
res = 1024, 1024
image = luisa.Image2D(*res, 4, float)
luisa.synchronize()
render(image, int2(*res), 0, dispatch_size=res)
luisa.synchronize()
image.to_image("a.png") # Note: compatibility of image needs improvement