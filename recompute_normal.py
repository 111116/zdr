import luisa
from luisa.atomic import _atomic_access_call

@luisa.func
def clear_normals(vertex_buffer):
    i = dispatch_id().x
    vert = vertex_buffer.read(i)
    vert.set_vn(luisa.float3(0.0))
    vertex_buffer.write(i, vert)

@luisa.func
def scatter_face_normal(vertex_buffer, triangle_buffer):
    i0 = triangle_buffer.read(dispatch_id().x * 3 + 0)
    i1 = triangle_buffer.read(dispatch_id().x * 3 + 1)
    i2 = triangle_buffer.read(dispatch_id().x * 3 + 2)
    p0 = vertex_buffer.read(i0)
    p1 = vertex_buffer.read(i1)
    p2 = vertex_buffer.read(i2)
    e1 = p1.v() - p0.v()
    e2 = p2.v() - p0.v()
    n = cross(e1, e2)
    # nested_level = 2 (vn, x/y/z)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i0, 2, 2, 0, n.x)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i0, 2, 2, 1, n.y)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i0, 2, 2, 2, n.z)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i1, 2, 2, 0, n.x)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i1, 2, 2, 1, n.y)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i1, 2, 2, 2, n.z)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i2, 2, 2, 0, n.x)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i2, 2, 2, 1, n.y)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vertex_buffer, i2, 2, 2, 2, n.z)

@luisa.func
def normalize_normals(vertex_buffer):
    i = dispatch_id().x
    vert = vertex_buffer.read(i)
    n = vert.vn()
    vert.set_vn(normalize(n))
    vertex_buffer.write(i, vert)

# def recompute_normal(v_buffer, vn_buffer, triangle_buffer):
#     assert triangle_buffer.size%3==0
#     clear_normals(vn_buffer, dispatch_size=vn_buffer.size)
#     scatter_face_normal(v_buffer, vn_buffer, triangle_buffer, dispatch_size=triangle_buffer.size//3)
#     normalize_normals(vn_buffer, dispatch_size=vn_buffer.size)
    
def recompute_normal(vertex_buffer, triangle_buffer):
    assert triangle_buffer.size%3==0
    clear_normals(vertex_buffer, dispatch_size=vertex_buffer.size)
    scatter_face_normal(vertex_buffer, triangle_buffer, dispatch_size=triangle_buffer.size//3)
    normalize_normals(vertex_buffer, dispatch_size=vertex_buffer.size)
    