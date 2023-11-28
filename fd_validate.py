import torch
import random
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import imageio

import sys
sys.path.append('..')
from zdr import Scene

def load_material(diffuse_file, roughness_file):
    from torchvision.transforms import ToTensor
    diffuse_img = ToTensor()(Image.open(diffuse_file)).cuda()
    roughness_img = ToTensor()(Image.open(roughness_file)).cuda()
    assert roughness_img.shape[0] == 1
    mat = torch.vstack((diffuse_img, roughness_img)).permute((1,2,0))**2.2
    return mat.contiguous()

obj_file = 'assets/sphere.obj'
scene = Scene(obj_file, use_face_normal=True)
diffuse_file = 'assets/wood_olive/wood_olive_wood_olive_basecolor.png'
roughness_file = 'assets/wood_olive/wood_olive_wood_olive_roughness.png'
material_GT = load_material(diffuse_file, roughness_file)


def compare_fdad(scene, material, texidx, imgidx, res):
    """Compare AD results with finite difference (FD)

    Prints FD and AD calculation of the gradient dI(imgidx)/dt(texidx).
    Each line is increasing spp, using 5 different seeds.
    
    Args:
        material (tensor)
        texidx (tuple of 3 ints): y, x, c
        imgidx (tuple of 3 ints): y, x, c
        res (tuple of 2 ints): w, h
        spp (int): samples per pixel
    
    """
    # finite difference of forward rendering
    def fd_grad(scene, material, texidx, imgidx, res, spp, seed, fd_eps):
        # two-sided difference, more accurate
        m0 = material.clone()
        m0[texidx] -= fd_eps
        I0 = scene.render(m0, res=res, spp=spp, seed=seed)
        m1 = material.clone()
        m1[texidx] += fd_eps
        I1 = scene.render(m1, res=res, spp=spp, seed=seed)
        # use the same seed for correlation to reduce variance
        return (I1[imgidx] - I0[imgidx]).item() / (2*fd_eps)

    # Automatic differentiation (backward)
    def ad_grad(scene, material, texidx, imgidx, res, spp, seed):
        material.requires_grad_()
        material.grad = None
        I = scene.render(material, res=res, spp=spp, seed=seed)
        I[imgidx].backward()
        return material.grad[texidx].item()
    
    # finite difference step size
    fd_eps = 0.01
    if material[texidx] < fd_eps or material[texidx] > 1-fd_eps:
        raise RuntimeError("material too close to boundary, can not FD")

    print("Increasing spps, each 5 different seeds")
    print("FD:")
    for spp in 1,2,4,8,16,32,64,128,256,512,1024,2048,4096:
        for seed in 0,12345,853402567, 19260817, 948377263:
            print("%0.6f" % fd_grad(scene, material, texidx=texidx, imgidx=imgidx, res=res, spp=spp, seed=seed, fd_eps=fd_eps), end=' ')
        print()
    # print(fd_grad(scene, material, texidx=texidx, imgidx=imgidx, res=res, spp=128, seed=0, fd_eps=fd_eps))
    
    print("AD:")
    for spp in 1,2,4,8,16,32,64,128,256,512,1024,2048,4096:
        for seed in 0,12345,853402567, 19260817, 948377263:
            print("%0.6f" % ad_grad(scene, material, texidx=texidx, imgidx=imgidx, res=res, spp=spp, seed=seed), end=' ')
        print()
    # print(ad_grad(scene, material, texidx=texidx, imgidx=imgidx, res=res, spp=128, seed=0))
    print("good if values in last row of AD and FD are similar")
    
def importance_sample_tensor(values: torch.tensor):
    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))
    values_flat = values.flatten()
    sampled_indices = torch.multinomial(values_flat, num_samples=1).item()
    return unravel_index(sampled_indices, values.shape)


# mandate differentiating roughness with a probability 
try_roughness = random.random() < 0.5
print()
print("Selecting pair: ", end='')
while True:
    print("#", end='')
    # randomly find points of interest in image
    material_GT.requires_grad_()
    material_GT.grad = None
    I_GT = scene.render(material_GT, res=(1024,1024), spp=128) # seed defaults to 0
    # Don't select the alpha channel (c=3) because the derivatives will all be zero
    imgidx = importance_sample_tensor(I_GT[...,0:3])

    # randomly find points of interest in material
    I_GT[imgidx].backward()
    texidx = importance_sample_tensor(material_GT.grad.abs())
    if texidx[-1]==3 or not try_roughness:
        print()
        break

print("Image index:", imgidx)
print("Pixel brightness:", I_GT[imgidx].item())
print("Texture index:", texidx)
print("Texel gradient:", material_GT.grad[texidx].item())
print()

# compare ad fd
compare_fdad(scene, material_GT, texidx, imgidx, res=(1024,1024))