import torch
import random
from PIL import Image
from tqdm import tqdm
import imageio
import luisa

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
scene = Scene([(obj_file, None, 0), ('assets/lightstage/l00.obj', None, 100)], integrator='direct')
diffuse_file = 'assets/wood_olive/wood_olive_wood_olive_basecolor_midres.png'
roughness_file = 'assets/wood_olive/wood_olive_wood_olive_roughness_midres.png'
material_GT = load_material(diffuse_file, roughness_file)

ImgRes = 512, 512
spp = 256
TexRes = material_GT.shape[0:2]
print("Image resolution:", ImgRes)
print("Texture resolution:", TexRes)
print("spp:", spp)


print("Forward + backward:")
# Benchmark forward + backward
material = material_GT.clone().requires_grad_()
for it in tqdm(range(1000)):
    I = scene.render(material, res=ImgRes, spp=spp, seed=random.randint(0, 2147483647))
    I.sum().backward()
