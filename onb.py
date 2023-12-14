# local frame (orthonormal basis)

import luisa
from luisa import float3, StructType


Onb = StructType(tangent=float3, binormal=float3, normal=float3)

@luisa.func
def to_world(self, v: float3):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal

@luisa.func
def to_local(self, v: float3):
    return float3(dot(v, self.tangent), dot(v, self.binormal), dot(v, self.normal))

Onb.add_method(to_world, "to_world")
Onb.add_method(to_local, "to_local")

@luisa.func
def make_onb(normal: float3):
    binormal = normalize(float3(-normal.y, normal.x, 0.0) if abs(normal.x) > abs(normal.z) else float3(0.0, -normal.z, normal.y))
    tangent = normalize(cross(binormal, normal))
    result = Onb()
    result.tangent = tangent
    result.binormal = binormal
    result.normal = normal
    return result

__all__ = ['Onb', 'make_onb']