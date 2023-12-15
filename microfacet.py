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
def ggx_sample(wo, diffuse, specular, roughness, sampler):
    if sampler.next() < 0.5:
        # sample diffuse lobe with p=0.5
        return cosine_sample_hemisphere(sampler.next2f())
    else:
        # sample glossy lobe with p=0.5
        alpha = roughness ** 2
        wm = sample_wm(wo, alpha, sampler.next2f())
        return reflect(-wo, wm)

@luisa.func
def ggx_sample_pdf(wo, wi, diffuse, specular, roughness):
    alpha = roughness ** 2
    wm = normalize(wi + wo)
    # mixed pdf
    diffuse_pdf = wi.z / pi
    glossy_pdf = pdf_wm(wo, wm, alpha) / (4 * abs(dot(wo, wm)))
    return 0.5 * diffuse_pdf + 0.5 * glossy_pdf


@luisa.func
def SampleUniformDiskPolar(u: float2):
    r = sqrt(u.x)
    theta = 2 * pi * u.y
    return float2(r * cos(theta), r * sin(theta))

@luisa.func
def pdf_wm(w: float3, wm: float3, alpha: float):
    return smith_geometry(w, alpha) / abs(w.z) * ggx_distribution(wm, alpha) * abs(dot(w, wm))

@luisa.func
def sample_wm(w: float3, alpha: float, u: float2) -> float3:
    """sample visible normal from Trowbridge-Reitz distribution (pbrt-v4)
    """
    # Transform _w_ to hemispherical configuration
    wh = normalize(float3(alpha * w.xy, w.z))
    if wh.z < 0:
        wh = -wh
    # Find orthonormal basis for visible normal sampling
    T1 = normalize(cross(float3(0,0,1), wh)) if wh.z < 0.99999 else float3(1,0,0)
    T2 = cross(wh, T1)
    # Generate uniformly distributed points on the unit disk
    p = SampleUniformDiskPolar(u)
    # Warp hemispherical projection for visible normal sampling
    h = sqrt(1 - p.x**2)
    p.y = lerp(h, p.y, (1+wh.z)/2)
    # Reproject to hemisphere and transform normal to ellipsoid configuration
    pz = sqrt(max(0.0, 1.0 - length_squared(p)))
    nh = p.x * T1 + p.y * T2 + pz * wh
    # nh.z==0 should be rare
    wm = normalize(float3(alpha * nh.xy, max(1e-6, nh.z)))
    return wm


