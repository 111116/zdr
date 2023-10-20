import luisa

@luisa.func
def generate_ray(p):
    fov = 60 / 180 * 3.1415926
    origin = float3(-0.01, 0.995, 5.0)
    target = float3(0.0, 0.0, 0.0)
    up = float3(0.0, 1.0, 0.0)
    forward = normalize(target - origin)
    right = normalize(cross(forward, up))
    up1 = cross(right, forward)
    
    p = p * tan(0.5 * fov)
    direction = normalize(p.x * right + p.y * up + forward)
    return make_ray(origin, direction, 0.0, 1e30)

@luisa.func
def direct_collocated(ray, accel, heap):
    hit = accel.trace_closest(ray, -1)
    if hit.miss():
        return float3(0.0)
    i0 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 0)
    i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
    i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
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
    mat = read_bsdf(uv)
    li = intensity * (1/hit.ray_t)**2
    return eval_bsdf_collocated(mat, ns, -ray.get_dir()) * li


@luisa.func
def render(image_buffer, accel, heap, resolution):
    coord = dispatch_id().xy
    sampler = luisa.RandomSampler(int3(coord, frame_id)) # builtin RNG; the sobol sampler can be used instead to improve convergence
    # generate ray from camera
    pixel = 2.0 / resolution * (float2(coord) + sampler.next2f()) - 1.0
    ray = generate_ray(pixel)
    radiance = direct_collocated(ray, accel, heap)
    if any(isnan(radiance)):
        radiance = float3(0.0)
    image.write(coord, float4(radiance, 1.0))


from load_obj import read_obj, concat_triangles
file_path = 'sphere.obj'
positions, tex_coords, normals, faces = read_obj(file_path)
# upload shapes
luisa.init()
v_buffer = luisa.buffer([luisa.float3(*x) for x in positions])
vt_buffer = luisa.buffer([luisa.float2(*x) for x in tex_coords])
vn_buffer = luisa.buffer([luisa.float3(*x) for x in normals])
triangle_buffer = luisa.buffer(concat_triangles(faces))
accel = luisa.Accel()
accel.add(v_buffer, triangle_buffer)
accel.update()
luisa.synchronize()

res = 1024, 1024

