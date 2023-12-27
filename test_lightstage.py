from math import cos, sin, pi, acos
import numpy as np

import torch
import random
from PIL import Image
from tqdm import tqdm
import imageio
import time
import sys
sys.path.append('..')
from zdr import Scene, float3, Camera
import luisa


def load_material(diffuse_file, roughness_file):
    from torchvision.transforms import ToTensor
    diffuse_img = ToTensor()(Image.open(diffuse_file)).cuda()[0:3]
    roughness_img = ToTensor()(Image.open(roughness_file)).cuda()[0:1]
    mat = torch.vstack((diffuse_img, roughness_img)).permute((1,2,0))**2.2
    return mat.contiguous()


def rotate_mat(theta, phi, offset):
    pitch = np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta),  cos(theta), 0, 0],
        [0,0,1,0],
        [0,0,0,1],
    ])
    yaw = np.array([
        [cos(phi), 0, -sin(phi), 0],
        [0, 1, 0, 0],
        [sin(phi), 0,  cos(phi), 0],
        [0,0,0,1],
    ])
    translate = np.array([
        [1,0,0,offset[0]],
        [0,1,0,offset[1]],
        [0,0,1,offset[2]],
        [0,0,0,1],
    ])
    m = yaw @ pitch @ translate
    return luisa.float4x4(*m.transpose().flatten())

models = []
models.append(('assets/bunnyuv.obj', rotate_mat(0,-0.4,float3(0,0,0)), None))
# for i in range(23):
#     models.append((f'assets/lightstage/l{i:02}.obj', None, 50))
# models.append(('assets/quad.obj', rotate_mat(0, 0, (0,10,0)), 50))
nlight = 30
for i in range(nlight):
    if i==29:
        models.append(('assets/quad.obj', rotate_mat(acos((i+0.5)/nlight*2-1), pi*2*0.618*(i+1), float3(0)), 50))

sphere_camera1 = Camera(
    fov = 50 / 180 * 3.1415926,
    origin = float3(0,0.5,2),
    target = float3(0,0,0),
    up = float3(0.0, 1.0, 0.0)
)


scene = Scene(models, integrator='direct')
scene.camera = sphere_camera1

# scene.update_lights([0,200,0,0,0,0])

diffuse_file = 'assets/wood_olive/wood_olive_wood_olive_basecolor.png'
roughness_file = 'assets/wood_olive/wood_olive_wood_olive_roughness.png'
material_GT = load_material(diffuse_file, roughness_file)
ImgRes = 1024, 1024
print("Image resolution:", ImgRes)


I_GT = scene.render(material_GT, res=ImgRes, spp=1024) # seed defaults to 0
Image.fromarray((I_GT[...,0:3].clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('results/gt.png')
quit()