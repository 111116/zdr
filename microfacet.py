import luisa
from luisa import float2, float3
from math import pi


@luisa.func
def ggx_distribution(h, alpha):
    # isotropic Trowbridge-Reitz distribution
    alpha2 = alpha * alpha
    nh = max(0.00001, h.z)
    return alpha2 / (pi * ((nh * nh * (alpha2 - 1) + 1)) ** 2)

@luisa.func
def fresnel_schlick(cos_theta, specular):
    return specular + (1 - specular) * (1 - cos_theta) ** 5

@luisa.func
def smith_geometry(v, alpha):
    alpha2 = alpha * alpha
    nv = max(0.00001, v.z)
    return 2 / (1 + sqrt(1 + alpha2 * (1 - nv * nv) / (nv * nv)))

@luisa.func
def ggx_brdf(wi, wo, diffuse, specular, roughness):
    alpha = roughness ** 2
    h = normalize(wi + wo)
    d = ggx_distribution(h, alpha)
    f = fresnel_schlick(clamp(dot(wo,h), 0.00001, 1.0), specular)
    g = smith_geometry(wi, alpha) * smith_geometry(wo, alpha)
    return ((d * f * g) / (4 * max(0.00001, wi.z) * max(0.00001, wo.z)) + diffuse/pi) * wo.z

