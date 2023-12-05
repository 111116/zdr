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
I_GT = scene.render(material_GT, res=(1024,1024), spp=128) # seed defaults to 0
Image.fromarray((I_GT[...,0:3].clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/gt.png')
duvdxy = scene.render_duvdxy(material_GT, res=(1024,1024), spp=128) # seed defaults to 0
Image.fromarray(((duvdxy[...,0:3]*1000+0.5).clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/duvdx_dudy.png')
pixel_independency = torch.det(duvdxy.reshape(1024,1024,2,2)*1024).abs()
Image.fromarray((pixel_independency.clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/pixel_independency.png')
pixel_independency1 = sum(torch.det(scene.render_duvdxy(material_GT,
                                                        res=(1024,1024),
                                                        spp=1,
                                                        seed=random.randint(0, 2147483647)
                                                        ).reshape(1024,1024,2,2)*1024).abs() for _ in range(128))/128
Image.fromarray((pixel_independency1.clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/pixel_independency1.png')

# scene.render_kernel = render_uvgrad_kernel


# ======== Optimization using differentiable rendering ========

# gradient descent
material = torch.rand((1024,1024,4), device='cuda')
material.requires_grad_()
optimizer = torch.optim.Adam([material], lr=0.01)
for it in tqdm(range(1000)):
    optimizer.zero_grad()
    I = scene.render(material, res=(1024,1024), spp=4, seed=random.randint(0, 2147483647))
    ((I-I_GT)**2).sum().backward()
    optimizer.step()
    material.data.clamp_(min=1e-3, max=1)

I = scene.render(material, res=(1024,1024), spp=16)
Image.fromarray((I[...,0:3].clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/a.png')
Image.fromarray((material[...,0:3].clamp(min=0, max=1)**0.454*255).detach().cpu().numpy().astype("uint8")).save("results/d.png")
Image.fromarray((material[...,3].clamp(min=0, max=1)**0.454*255).detach().cpu().numpy().astype("uint8")).save("results/dr.png")
print("loss", ((I-I_GT)**2).sum().item())

