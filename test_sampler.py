import luisa
from matplotlib import pyplot as plt
import torch
from pmj02bn import *
from corrmj import *
from luisa.mathtypes import *
from math import pi

luisa.init()

n = 32
b = luisa.buffer([luisa.float2(0) for _ in range(n)])


@luisa.func
def SampleUniformDiskPolar(u: float2):
    r = sqrt(u.x)
    theta = 2 * pi * u.y
    return float2(r * cos(theta), r * sin(theta))

@luisa.func
def f():
    pixel = int2(24,345)
    seed = 0
    for i in range(n):
#         sampler = make_corrmj_sampler(pixel, seed, n, i)
        sampler = make_pmj02bn_sampler(pixel, seed, n, i)
        u = sampler.next2f()
        u = sampler.next2f()
        h = sampler.next()
#         u = sampler.next2f()
#         p = SampleUniformDiskPolar(u)
        b.write(i, u)
#         b.write(i, float2(h,0.1))

f(dispatch_size=1)

a = torch.from_dlpack(b).cpu()
plt.scatter(a[:,0], a[:,1])


plt.xlim(0, 1)
plt.ylim(0, 1)
ax = plt.gca()

# c = plt.Circle((0,0),1, fill=False)
# ax.add_patch(c)

ax.set_aspect('equal', adjustable='box')