import torch
import luisa
from luisa.mathtypes import *
import numpy as np
from load_obj import read_obj, concat_triangles
from recompute_normal import recompute_normal
from uvgrad import render_uvgrad_kernel
from integrator import render_kernel, render_backward_kernel
import weakref

luisa.init()


def float3list_to_padded_tensor(l):
    a = torch.tensor(l, dtype=torch.float32, device='cuda')
    assert a.dim()==2 and a.shape[1]==3
    n = a.shape[0]
    b = torch.empty((n, 1), dtype=torch.float32, device='cuda')
    w = torch.hstack((a,b))
    return w.as_strided(size=(n,3), stride=(4,1))

def np_from_image(file, n_channels):
    arr = luisa.lcapi.load_ldr_image(file)
    assert arr.ndim == 3 and arr.shape[2] == 4
    return arr[..., 0:n_channels]


class Scene:
    def __init__(self, obj_file, diffuse_file, roughness_file, use_face_normal=False):
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
        # load material
        diffuse_arr = np_from_image(diffuse_file, 3)
        roughness_arr = np_from_image(roughness_file, 1)
        arr = np.concatenate((diffuse_arr, roughness_arr), axis=2)
        self.texture_res = arr.shape[0:2]
        arr = ((arr.astype('float32')/255)**2.2).flatten()
        # row, column, 4 floats (diffuse + roughness)
        self.material_buffer = luisa.buffer(arr)
        self.d_material_buffer = luisa.Buffer(size=self.material_buffer.size, dtype=self.material_buffer.dtype)
        self.material = torch.from_dlpack(self.material_buffer).reshape((*self.texture_res, -1))
        self.d_material = torch.from_dlpack(self.d_material_buffer).reshape((*self.texture_res, -1))
        # set camera
        self.camera = luisa.struct(
            fov = 40 / 180 * 3.1415926,
            origin = float3(1.0, 0.5, 0.0),
            target = float3(0.0, 0.0, 0.0),
            up = float3(0.0, 1.0, 0.0)
        )
        self.render_kernel = render_kernel

    def render_forward(self, res, spp, seed):
        image = luisa.Buffer(res[0]*res[1], dtype=float4)
        material_buffer = luisa.Buffer.from_dlpack(self.material.flatten())
        self.render_kernel(image,
            self.v_buffer, self.vt_buffer, self.vn_buffer, self.triangle_buffer,
            self.accel, material_buffer, int2(*self.texture_res), self.camera,
            spp, seed, dispatch_size=res)
        luisa.synchronize()
        return torch.from_dlpack(image).reshape((res[1],res[0],-1))
        # TODO fix Segmentation fault (core dumped) at end of program
    
    def render_backward(self, grad_output, res, spp, seed):
        d_image = luisa.Buffer.from_dlpack(grad_output.reshape((res[0]*res[1], -1)))
        material_buffer = luisa.Buffer.from_dlpack(self.material.flatten())
        assert d_image.size == res[0]*res[1] and d_image.dtype == float4
        render_backward_kernel(d_image,
            self.v_buffer, self.vt_buffer, self.vn_buffer, self.triangle_buffer, self.accel,
            self.d_material_buffer, material_buffer, int2(*self.texture_res), self.camera,
            spp, seed+1, dispatch_size=res)
        luisa.synchronize()

    class RenderOperator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dummy, self, *args):
            ctx.scene = weakref.ref(self)
            ctx.args = args
            return self.render_forward(*args)
        
        @staticmethod
        def backward(ctx, grad_output):
            ctx.scene().render_backward(grad_output, *ctx.args)
            return tuple([None]*(len(ctx.args)+2))
            
    def render(self, res, spp, seed):
        dummy = torch.zeros(1, requires_grad=True)
        # Torch requires that first argument be differentiable tensor
        return Scene.RenderOperator.apply(dummy, self, res, spp, seed)


if __name__ == "__main__":
    obj_file = 'assets/sphere.obj'
    diffuse_file = 'assets/wood_olive/wood_olive_wood_olive_basecolor.png'
    roughness_file = 'assets/wood_olive/wood_olive_wood_olive_roughness.png'
    scene = Scene(obj_file, diffuse_file, roughness_file, use_face_normal=True)
    # scene.render_kernel = render_uvgrad_kernel
    I = scene.render(res=(1024,1024), spp=1, seed=0)
    from PIL import Image
    Image.fromarray((I[...,0:3].clamp(min=0,max=1)**0.454*255).to(torch.uint8).cpu().numpy()).save('a.png')
    scene.d_material.zero_()
    (I**2/2).sum().backward()
    Image.fromarray((scene.d_material[...,0:3].clamp(min=0, max=1)**0.454*255).cpu().numpy().astype("uint8")).save("d.png")
