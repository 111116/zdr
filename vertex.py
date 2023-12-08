from luisa import StructType, ArrayType, func, float2, float3

# compact storing vertex position, normal & texture coords
Vertex = StructType(alignment=16, _v=ArrayType(3, float), _vt=ArrayType(2, float), _vn=ArrayType(3, float))

@func
def v(self):
    return float3(self._v[0], self._v[1], self._v[2])

@func
def vt(self):
    return float2(self._vt[0], self._vt[1])

@func
def vn(self):
    return float3(self._vn[0], self._vn[1], self._vn[2])

@func
def set_v(self, val: float3):
    self._v[0] = val.x
    self._v[1] = val.y
    self._v[2] = val.z

@func
def set_vt(self, val: float2):
    self._vt[0] = val.x
    self._vt[1] = val.y

@func
def set_vn(self, val: float3):
    self._vn[0] = val.x
    self._vn[1] = val.y
    self._vn[2] = val.z


# read-only methods
Vertex.add_method(v)
Vertex.add_method(vt)
Vertex.add_method(vn)

# setter methods
Vertex.add_method(set_v)
Vertex.add_method(set_vt)
Vertex.add_method(set_vn)
