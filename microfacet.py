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
def ggx_brdf(wo, wi, diffuse, specular, roughness):
    alpha = roughness ** 2
    h = normalize(wi + wo)
    d = ggx_distribution(h, alpha)
    f = fresnel_schlick(clamp(dot(wo,h), 0.00001, 1.0), specular)
    g = smith_geometry(wi, alpha) * smith_geometry(wo, alpha)
    return ((d * f * g) / (4 * max(0.00001, wi.z) * max(0.00001, wo.z)) + diffuse/pi) * wi.z


@luisa.func
def cosine_sample_hemisphere(u: float2):
    r = sqrt(u.x)
    phi = 2 * pi * u.y
    return float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x))

@luisa.func
def reflect(wo, n):
    return -wo + 2 * dot(wo, n) * n

@luisa.func
def ggx_sample(wo, diffuse, specular, roughness, sampler):
    alpha = roughness ** 2
    u = sampler.next2f()
    # sample diffuse
    diffuse_sample_wi = cosine_sample_hemisphere(u)
    diffuse_sample_pdf = diffuse_sample_wi.z / pi
    # sample gloss
    wm = sample_wm(wo, alpha, u)
    glossy_sample_wi = reflect(wo, wm)
    glossy_sample_pdf = pdf_wm(wo, wm, alpha) / (4 * abs(dot(wo, wm)))
    # importance sampling
    # w1 = ggx_brdf(diffuse_sample_wi, wo, diffuse, specular, roughness) / diffuse_sample_pdf
    # w2 = ggx_brdf(glossy_sample_wi, wo, diffuse, specular, roughness) / glossy_sample_pdf
    # TODO only sampling glossy lobe now
    return diffuse_sample_wi

@luisa.func
def ggx_sample_pdf(wo, wi, diffuse, specular, roughness):
    return wi.z / pi
    alpha = roughness ** 2
    wm = normalize(wi + wo)
    # TODO only sampling glossy lobe now
    return pdf_wm(wo, wm, alpha) / (4 * abs(dot(wo, wm)))


@luisa.func
def SampleUniformDiskPolar(u: float2):
    r = sqrt(u.x)
    theta = 2 * pi * u.y
    return float2(r * cos(theta), r * sin(theta))

@luisa.func
def pdf_wm(w: float3, wm: float3, alpha: float):
    return ggx_distribution(wm, alpha) * smith_geometry(w, alpha) * abs(w.z) * abs(dot(w, wm))

@luisa.func
def sample_wm(w: float3, alpha: float, u: float2) -> float3:
    """sample visible normal from Trowbridge-Reitz distribution (pbrt-v4)
    """
    # Transform _w_ to hemispherical configuration
    wh = normalize(float3(alpha * w.xy, w.z))
    if wh.z < 0:
        wh = -wh
    # Find orthonormal basis for visible normal sampling
    T1 = normalize(cross(float3(0,0,1), wh)) if wh.z < 0.9999 else float3(1,0,0)
    T2 = cross(wh, T1)
    # Generate uniformly distributed points on the unit disk
    p = SampleUniformDiskPolar(u)
    # Warp hemispherical projection for visible normal sampling
    h = sqrt(1 - p.x**2)
    p.y = lerp((1+wh.z)/2, h, p.y)
    # Reproject to hemisphere and transform normal to ellipsoid configuration
    pz = sqrt(max(0.0, 1.0 - length_squared(p)))
    nh = p.x * T1 + p.y * T2 + pz * wh
    # nh.z==0 should be rare
    return normalize(float3(alpha * nh.xy, max(1e-6, nh.z)))


