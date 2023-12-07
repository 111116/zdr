import torch
import random
from PIL import Image
from tqdm import tqdm
import imageio

import sys
sys.path.append('..')
from zdr import Scene
import luisa

def load_material(diffuse_file, roughness_file):
    from torchvision.transforms import ToTensor
    diffuse_img = ToTensor()(Image.open(diffuse_file)).cuda()
    roughness_img = ToTensor()(Image.open(roughness_file)).cuda()
    assert roughness_img.shape[0] == 1
    mat = torch.vstack((diffuse_img, roughness_img)).permute((1,2,0))**2.2
    return mat.contiguous()

obj_file = 'assets/the-valentini-torso_high.obj'
scene = Scene(obj_file, use_face_normal=True)
scene.camera = luisa.struct(
    fov = 50 / 180 * 3.1415926,
    origin = luisa.float3(5.0, 3.0, 4.0),
    target = luisa.float3(0.0, 2.0, 0.0),
    up = luisa.float3(0.0, 1.0, 0.0)
)
diffuse_file = 'assets/wood_olive/wood_olive_wood_olive_basecolor.png'
roughness_file = 'assets/wood_olive/wood_olive_wood_olive_roughness.png'
material_GT = load_material(diffuse_file, roughness_file)
ImgRes = 1024, 1024
print("Image resolution:", ImgRes)
I_GT = scene.render(material_GT, res=ImgRes, spp=128) # seed defaults to 0
Image.fromarray((I_GT[...,0:3].clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/gt.png')

# duv/dxy (screen space to texture space jacobian)
duvdxy = scene.render_duvdxy(material_GT, res=ImgRes, spp=128) # seed defaults to 0
Image.fromarray(((duvdxy[...,0:3]*1000+0.5).clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/duvdx_dudy.png')

quit()
# ======== Optimization using differentiable rendering ========
# Note that this is just an example, where scene.camera remains unchanged.

# gradient descent
TexRes = 1024, 1024
print("Texture resolution:", TexRes)
material = torch.rand((*TexRes,4), device='cuda')
material.requires_grad_()
optimizer = torch.optim.Adam([material], lr=0.01)
for it in tqdm(range(500)):
    I = scene.render(material, res=ImgRes, spp=4, seed=random.randint(0, 2147483647))
    ((I-I_GT)**2).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    material.data.clamp_(min=1e-3, max=1)

# Rendered image of reconstruction
I = scene.render(material, res=ImgRes, spp=16)
print("MSE", ((I-I_GT)**2).mean().item())
Image.fromarray((I[...,0:3].clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/a.png')

# Reconstructed texture
Image.fromarray((material[...,0:3].clamp(min=0, max=1)**0.454*255).detach().cpu().numpy().astype("uint8")).save("results/d.png")
Image.fromarray((material[...,3].clamp(min=0, max=1)**0.454*255).detach().cpu().numpy().astype("uint8")).save("results/dr.png")

