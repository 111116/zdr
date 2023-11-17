import luisa
from luisa.atomic import _atomic_access_call

@luisa.func
def clear_normals(vn_buffer):
    i = dispatch_id().x
    vn_buffer.write(i, luisa.float3(0.0))

@luisa.func
def scatter_face_normal(v_buffer, vn_buffer, triangle_buffer):
    i0 = triangle_buffer.read(dispatch_id().x * 3 + 0)
    i1 = triangle_buffer.read(dispatch_id().x * 3 + 1)
    i2 = triangle_buffer.read(dispatch_id().x * 3 + 2)
    v0 = v_buffer.read(i0)
    v1 = v_buffer.read(i1)
    v2 = v_buffer.read(i2)
    e1 = v1 - v0
    e2 = v2 - v0
    n = cross(e1, e2)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i0, 1, 0, n.x)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i0, 1, 1, n.y)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i0, 1, 2, n.z)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i1, 1, 0, n.x)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i1, 1, 1, n.y)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i1, 1, 2, n.z)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i2, 1, 0, n.x)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i2, 1, 1, n.y)
    _ = _atomic_access_call(float, "ATOMIC_FETCH_ADD", vn_buffer, i2, 1, 2, n.z)

@luisa.func
def normalize_normals(vn_buffer):
    i = dispatch_id().x
    n = vn_buffer.read(i)
    vn_buffer.write(i, normalize(n))

def recompute_normal(v_buffer, vn_buffer, triangle_buffer):
    assert triangle_buffer.size%3==0
    clear_normals(vn_buffer, dispatch_size=vn_buffer.size)
    scatter_face_normal(v_buffer, vn_buffer, triangle_buffer, dispatch_size=triangle_buffer.size//3)
    normalize_normals(vn_buffer, dispatch_size=vn_buffer.size)
    