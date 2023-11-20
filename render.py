import torch
import luisa
from luisa.mathtypes import *
import numpy as np
from PIL import Image
import weakref

from load_obj import read_obj, concat_triangles
from recompute_normal import recompute_normal
from uvgrad import render_uvgrad_kernel
from integrator import render_kernel, render_backward_kernel

luisa.init('cuda')


def float3list_to_padded_tensor(l):
    a = torch.tensor(l, dtype=torch.float32, device='cuda')
    assert a.dim()==2 and a.shape[1]==3
    n = a.shape[0]
    b = torch.empty((n, 1), dtype=torch.float32, device='cuda')
    w = torch.hstack((a,b))
    return w.as_strided(size=(n,3), stride=(4,1))

class Scene:
    def __init__(self, obj_file, use_face_normal=False):
        # load geometry from obj file
        # TODO recompute normal if not availble
        positions, tex_coords, normals, faces = read_obj(obj_file)
        self.v_buffer = luisa.Buffer.from_dlpack(float3list_to_padded_tensor(positions))
        self.vt_buffer = luisa.Buffer.from_dlpack(torch.tensor(tex_coords, dtype=torch.float32, device='cuda'))
        self.vn_buffer = luisa.Buffer.from_dlpack(float3list_to_padded_tensor(normals))
        self.triangle_buffer = luisa.buffer(concat_triangles(faces))
        if use_face_normal:
            recompute_normal(self.v_buffer, self.vn_buffer, self.triangle_buffer)
        self.accel = luisa.Accel()
        self.accel.add(self.v_buffer, self.triangle_buffer)
        self.accel.update()
        self.camera = luisa.struct(
            fov = 40 / 180 * 3.1415926,
            origin = float3(1.0, 0.5, 0.0),
            target = float3(0.0, 0.0, 0.0),
            up = float3(0.0, 1.0, 0.0)
        )
        self.render_kernel = render_kernel

    def render_forward(self, material, res, spp, seed):
        assert material.ndim == 3 and material.shape[2] == 4
        texture_res = material.shape[0:2]
        material_buffer = luisa.Buffer.from_dlpack(material.detach().flatten())
        image = torch.empty((res[1], res[0], 4), dtype=torch.float32, device='cuda')
        image_buffer = luisa.Buffer.from_dlpack(image.reshape((res[0]*res[1], 4)))
        self.render_kernel(image_buffer,
            self.v_buffer, self.vt_buffer, self.vn_buffer, self.triangle_buffer,
            self.accel, material_buffer, int2(*texture_res), self.camera,
            spp, seed, dispatch_size=res)
        luisa.synchronize()
        return image
    
    def render_backward(self, grad_output, material, res, spp, seed):
        assert material.ndim == 3 and material.shape[2] == 4
        texture_res = material.shape[0:2]
        material_buffer = luisa.Buffer.from_dlpack(material.detach().flatten())
        # Can't use empty_like here because d_material have to be contiguous
        d_material = torch.empty(material.size(), dtype=material.dtype, device=material.device)
        d_material_buffer = luisa.Buffer.from_dlpack(d_material.flatten())
        d_image = luisa.Buffer.from_dlpack(grad_output.reshape((res[0]*res[1], 4)).contiguous())
        render_backward_kernel(d_image,
            self.v_buffer, self.vt_buffer, self.vn_buffer, self.triangle_buffer, self.accel,
            d_material_buffer, material_buffer, int2(*texture_res), self.camera,
            spp, seed+1, dispatch_size=res)
        luisa.synchronize()
        return d_material, None, None, None, None

    class RenderOperator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, material, self, *args):
            ctx.save_for_backward(material)
            ctx.scene = weakref.ref(self)
            ctx.args = args
            return self.render_forward(material, *args)
        
        @staticmethod
        def backward(ctx, grad_output):
            material, = ctx.saved_tensors
            return ctx.scene().render_backward(grad_output, material, *ctx.args)
            
    def render(self, material, *, res, spp, seed):
        return Scene.RenderOperator.apply(material, self, res, spp, seed)


if __name__ == "__main__":

    def load_material(diffuse_file, roughness_file):
        from torchvision.transforms import ToTensor
        diffuse_img = ToTensor()(Image.open(diffuse_file)).cuda()
        roughness_img = ToTensor()(Image.open(roughness_file)).cuda()
        assert roughness_img.shape[0] == 1
        mat = torch.vstack((diffuse_img, roughness_img)).permute((1,2,0))**2.2
        return mat
    obj_file = 'assets/sphere.obj'
    scene = Scene(obj_file, use_face_normal=True)
    diffuse_file = 'assets/wood_olive/wood_olive_wood_olive_basecolor.png'
    roughness_file = 'assets/wood_olive/wood_olive_wood_olive_roughness.png'
    material = load_material(diffuse_file, roughness_file)
    material.requires_grad_()
    # scene.render_kernel = render_uvgrad_kernel
    I = scene.render(material, res=(1024,1024), spp=1, seed=0)
    Image.fromarray((I[...,0:3].clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('a.png')
    I.sum().backward()
    Image.fromarray((material.grad[...,0:3].clamp(min=0, max=1)**0.454*255).cpu().numpy().astype("uint8")).save("d.png")
    Image.fromarray((material.grad[...,3].clamp(min=0, max=1)**0.454*255).cpu().numpy().astype("uint8")).save("dr.png")
