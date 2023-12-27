import luisa
import torch
from luisa.mathtypes import *

@luisa.func
def _permutation_element(i, l, w, p):
    while True:
        i ^= p
        i *= 0xe170893d
        i ^= p >> 16
        i ^= (i & w) >> 4
        i ^= p >> 8
        i *= 0x0929eb3f
        i ^= p >> 23
        i ^= (i & w) >> 1
        i *= 1 | p >> 27
        i *= 0x6935fa69
        i ^= (i & w) >> 11
        i *= 0x74dcb303
        i ^= (i & w) >> 2
        i *= 0x9e501cc3
        i ^= (i & w) >> 2
        i *= 0xc860a3df
        i &= w
        i ^= i >> 5
        if i < l:
            break
    return (i + p) % l

@luisa.func
def xxhash32_int4(p: int4):
    PRIME32_2 = 2246822519
    PRIME32_3 = 3266489917
    PRIME32_4 = 668265263
    PRIME32_5 = 374761393
    h32 = p.w + PRIME32_5 + p.x * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)))
    h32 += p.y * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)))
    h32 += p.z * PRIME32_3
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)))
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15))
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13))
    return h32 ^ (h32 >> 16)

one_minus_epsilon = float.fromhex("0x1.fffffep-1")

CorrMJSampler = luisa.StructType(
    sample_index=int,
    dimension=int,
    permutation_seed=int,
    state=int,
    spp=int,
    w=int,
    res=int,
    resw=int
)

@luisa.func
def make_corrmj_sampler(pixel, seed, spp, sample_index):
    w = spp - 1
    w |= w >> 1
    w |= w >> 2
    w |= w >> 4
    w |= w >> 8
    w |= w >> 16
    res = int(sqrt(spp+0.4))
    resw = res - 1
    resw |= resw >> 1
    resw |= resw >> 2
    resw |= resw >> 4
    resw |= resw >> 8
    resw |= resw >> 16

    t = CorrMJSampler()
    t.sample_index = sample_index
    t.dimension = 0
    t.permutation_seed = xxhash32_int4(int4(pixel, seed, 0))
    t.state = xxhash32_int4(int4(pixel, seed, sample_index))
    t.spp = spp
    t.w = w
    t.res = res
    t.resw = resw
    return t


@luisa.func
def _next_lcg(sampler):
    lcg_a = 1664525
    lcg_c = 1013904223
    sampler.state = lcg_a * sampler.state + lcg_c
    return float(sampler.state & 0x00ffffff) * (1.0 / 0x01000000)

@luisa.func
def generate_1d(sampler):
    perm_seed = sampler.permutation_seed + sampler.dimension
    index = _permutation_element(sampler.sample_index, sampler.spp, sampler.w, (perm_seed * 0x45fbe943) & 0x70ffffff)
    delta = _next_lcg(sampler)
    u = (index + delta) / sampler.spp
    sampler.dimension += 1
    return clamp(u, 0.0, one_minus_epsilon)

@luisa.func
def generate_2d(sampler):
    perm_seed = sampler.permutation_seed + sampler.dimension
    index = _permutation_element(sampler.sample_index, sampler.spp, sampler.w, (perm_seed * 0x51633e2d) & 0x70ffffff)
    y = index // sampler.res
    x = index % sampler.res
    sx = _permutation_element(x, sampler.res, sampler.resw, (perm_seed * 0x68bc21eb) & 0x70ffffff)
    sy = _permutation_element(y, sampler.res, sampler.resw, (perm_seed * 0x02e5be93) & 0x70ffffff)
    dx = _next_lcg(sampler)
    dy = _next_lcg(sampler)
    u = (float2(x,y) + float2(sy + dx, sx + dy) / sampler.res) / sampler.res
    sampler.dimension += 2
    return clamp(u, 0.0, one_minus_epsilon)

CorrMJSampler.add_method(generate_1d, "next")
CorrMJSampler.add_method(generate_2d, "next2f")

__all__ = ['CorrMJSampler', 'make_corrmj_sampler']