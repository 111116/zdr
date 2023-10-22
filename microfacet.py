import luisa
from math import pi

@luisa.func
def ggx_distribution(n, h, alpha):
    alpha2 = alpha * alpha
    nh = max(0.00001, dot(n,h))
    return alpha2 / (pi * ((nh * nh * (alpha2 - 1) + 1)) ** 2)

@luisa.func
def fresnel_schlick(cos_theta, specular):
    return specular + (1 - specular) * (1 - cos_theta) ** 5

@luisa.func
def smith_geometry(n, v, alpha):
    nv = max(0.00001, dot(n,v))
    alpha2 = alpha * alpha
    return 2 / (1 + sqrt(1 + alpha2 * (1 - nv * nv) / (nv * nv)))

@luisa.func
def ggx_brdf(wi, wo, n, diffuse, specular, roughness):
    alpha = roughness ** 2
    h = normalize(wi + wo)
    d = ggx_distribution(n, h, alpha)
    f = fresnel_schlick(clamp(dot(wo,h), 0.00001, 1.0), specular)
    g = smith_geometry(n, wi, alpha) * smith_geometry(n, wo, alpha)
    # if not (1 - dot(wo,h)) ** 5 > -0.0001:
    #     return luisa.float3(1.0, 0.0, 1.0)
    # if not f>0:
    #     return luisa.float3(0.0,0.0,1.0)
    return ((d * f * g) / (4 * max(0.00001, dot(n,wi)) * max(0.00001, dot(n,wo))) + diffuse) * dot(wo,h)
