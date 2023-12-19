from .bluenoise import *
from .pmj02tables import *
import luisa
import torch
from luisa.mathtypes import *


luisa.init() # doesn't do anything if it's already initialized
PMJ02bnSamples_torch = (torch.tensor(PMJ02bnSamples)/2**32).to(torch.float32).cuda()
assert PMJ02bnSamples_torch.shape == (nPMJ02bnSets, nPMJ02bnSamples, 2)
PMJ02bnSamples_buffer = luisa.Buffer.from_dlpack(PMJ02bnSamples_torch.reshape(-1,2))
assert PMJ02bnSamples_buffer.dtype == luisa.float2

BlueNoiseTextures_torch = (torch.tensor(BlueNoiseTextures)/2**16).to(torch.float32).cuda()
assert BlueNoiseTextures_torch.shape == (NumBlueNoiseTextures, BlueNoiseResolution, BlueNoiseResolution)
BlueNoiseTextures_buffer = luisa.Buffer.from_dlpack(BlueNoiseTextures_torch.flatten())


@luisa.func
def _blue_noise(tex_id: int, p: int2) -> float:
    tex_index = tex_id % NumBlueNoiseTextures
    coord = p % BlueNoiseResolution
    i = tex_index * (BlueNoiseResolution + coord.x) * BlueNoiseResolution + coord.y
    return BlueNoiseTextures_buffer.read(i)

@luisa.func
def _pmj02bn_sample(set_id: int, sample_id: int) -> float2:
        set_index = set_id % nPMJ02bnSets
        i = set_index * nPMJ02bnSamples + sample_id
        return PMJ02bnSamples_buffer.read(i)

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
    # return (i + p) % l
    #FIXME this might theoretically produce negative results?
    return ((i + p) % l + l) % l

@luisa.func
def xxhash32_int4(ps: int4):
    p = make_uint4(ps) # workaround for py signed int...
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

PMJ02bnSampler = luisa.StructType(
    pixel=int2,
    sample_index=int,
    dimension=int,
    seed=int,
    spp=int,
    w=int
)

@luisa.func
def make_pmj02bn_sampler(pixel, seed, spp, sample_index):
    w = spp - 1
    w |= w >> 1
    w |= w >> 2
    w |= w >> 4
    w |= w >> 8
    w |= w >> 16
    t = PMJ02bnSampler()
    t.pixel = pixel
    t.sample_index = sample_index
    t.dimension = 0
    t.seed = seed
    t.spp = spp
    t.w = w
    return t

@luisa.func
def generate_1d(sampler):
    hash32 = xxhash32_int4(int4(sampler.pixel, sampler.dimension, sampler.seed))
    index = _permutation_element(sampler.sample_index, sampler.spp, sampler.w, hash32)
    delta = _blue_noise(sampler.dimension, sampler.pixel ^ sampler.seed)
    # xor seed to change blue noise for each pixel when changing seed
    u = (index + delta) / sampler.spp
    sampler.dimension += 1
    return clamp(u, 0.0, one_minus_epsilon)

@luisa.func
def generate_2d(sampler):
    index = sampler.sample_index
    pmj_instance = sampler.dimension // 2
    if pmj_instance >= nPMJ02bnSets:
        hash32 = xxhash32_int4(int4(sampler.pixel, sampler.dimension, sampler.seed))
        index = _permutation_element(sampler.sample_index, sampler.spp, sampler.w, hash32)
    u = _pmj02bn_sample(pmj_instance, index) + make_float2(
        _blue_noise(sampler.dimension, sampler.pixel ^ sampler.seed),
        _blue_noise(sampler.dimension+1, sampler.pixel ^ sampler.seed)
    )
    sampler.dimension += 2
    return fract(u)

PMJ02bnSampler.add_method(generate_1d, "next")
PMJ02bnSampler.add_method(generate_2d, "next2f")

__all__ = ['PMJ02bnSampler', 'make_pmj02bn_sampler']