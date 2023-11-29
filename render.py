"""Simple Differentiable Renderer using Luisa.

Currently only supports collocated direct lighting.

"""
import torch
import luisa
from luisa.mathtypes import *
import weakref
# import gc

from .load_obj import read_obj, concat_triangles
from .recompute_normal import recompute_normal
from .uvgrad import render_uvgrad_kernel
from .integrator import render_kernel, render_backward_kernel

# Using CUDA for interaction with PyTorch
luisa.init('cuda')


def float3list_to_padded_tensor(l):
    a = torch.tensor(l, dtype=torch.float32, device='cuda')
    assert a.dim()==2 and a.shape[1]==3
    n = a.shape[0]
    b = torch.empty((n, 1), dtype=torch.float32, device='cuda')
    w = torch.hstack((a,b))
    return w.as_strided(size=(n,3), stride=(4,1))

class Scene:
    """A class representing a 3D scene for differentiable rendering.

    The Scene class is responsible for setting up the 3D environment, 
    which includes loading the geometry from an OBJ file, setting up 
    materials, and configuring the camera. It provides functionalities 
    for forward and backward rendering using the differentiable rendering 
    technique.

    Attributes:
        camera (struct): Camera settings including FOV, origin, target, and up vector.
                         Do NOT change order of fields!
            fov (float): The field of view of the camera in radians
            origin (float3): The position of the camera in the 3D world space.
            target (float3): The point in the 3D space that the camera is looking at.
            up (float3): The up vector of the camera, typically (0,1,0).

        use_tent_filter (bool): Will use tent reconstruction filter if True (default);
                                Box filter if set to False.

    Args:
        obj_file (str): Path to the OBJ file containing the 3D model geometry.
        use_face_normal (bool, optional): Flag to determine whether to use face normals. 
                                          Will recompute vertex normals from faces if True.
                                          Defaults to False.

    """
    def __init__(self, obj_file, use_face_normal=False):
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
        self.use_tent_filter = True

    def render_forward(self, material, res, spp, seed, integrator_kernel=render_kernel):
        assert material.ndim == 3 and material.shape[2] == 4
        texture_res = material.shape[0:2]
        material_buffer = luisa.Buffer.from_dlpack(material.detach().flatten())
        image = torch.empty((res[1], res[0], 4), dtype=torch.float32, device='cuda')
        image_buffer = luisa.Buffer.from_dlpack(image.reshape((res[0]*res[1], 4)))
        torch.cuda.synchronize() # make sure material is ready
        integrator_kernel(image_buffer,
            self.v_buffer, self.vt_buffer, self.vn_buffer, self.triangle_buffer,
            self.accel, material_buffer, int2(*texture_res), self.camera,
            spp, seed, self.use_tent_filter, dispatch_size=res)
        luisa.synchronize()
        return image
    
    def render_backward(self, grad_output, d_material, material, res, spp, seed, integrator_kernel=render_backward_kernel):
        assert material.ndim == 3 and material.shape[2] == 4
        texture_res = material.shape[0:2]
        material_buffer = luisa.Buffer.from_dlpack(material.flatten())
        d_material_buffer = luisa.Buffer.from_dlpack(d_material.flatten())
        d_image = luisa.Buffer.from_dlpack(grad_output.reshape((res[0]*res[1], 4)).contiguous())

        # def namestr(obj, namespace):
        #     return [name for name in namespace if namespace[name] is obj]
        # referrers = gc.get_referrers(grad_output)
        # for referrer in referrers:
        #     print("- ", namestr(referrer, globals()), namestr(referrer, locals()))

        # print("REF", len(gc.get_referrers(grad_output)), len(gc.get_referrers(d_image)))
        torch.cuda.synchronize() # make sure grad_output is ready
        integrator_kernel(d_image,
            self.v_buffer, self.vt_buffer, self.vn_buffer, self.triangle_buffer, self.accel,
            d_material_buffer, material_buffer, int2(*texture_res), self.camera,
            spp, seed+1, self.use_tent_filter, dispatch_size=res)
        # print("REF", len(gc.get_referrers(grad_output)), len(gc.get_referrers(d_image)))
        luisa.synchronize()
        return d_material, None, None, None, None

    class RenderOperator(torch.autograd.Function):
        @staticmethod
        def forward(ctx, material, self, *args):
            ctx.save_for_backward(material)
            ctx.scene = weakref.ref(self)
            ctx.args = args
            ctx.camera = self.camera
            return self.render_forward(material.detach(), *args)

        @staticmethod
        def backward(ctx, grad_output):
            # print("a", len(gc.get_referrers(grad_output)))
            # scene.camera may be modified between forward and backward.
            # workaround: save and load camera state
            camera_bak = ctx.scene().camera
            ctx.scene().camera = ctx.camera
            material, = ctx.saved_tensors
            mat_grad = torch.zeros(material.size(), dtype=material.dtype, device=material.device)
            grads = ctx.scene().render_backward(grad_output, mat_grad, material.detach(), *ctx.args)
            ctx.scene().camera = camera_bak
            return grads
            
    def render(self, material, *, res, spp, seed=0):
        """Renders the scene with the given material and rendering parameters.

        This operation is differentiable w.r.t. material.

        Args:
            material (torch.Tensor): A tensor of shape (wu,hu,c) # TODO
            res (tuple): Resolution of the rendered image as a tuple (width, height).
            spp (int): Samples per pixel for rendering. Higher values increase quality at the
                       cost of more computation.
            seed (int): Seed for random number in rendering.

        Returns:
            torch.Tensor: A tensor (height, width, 4) representing the rendered image.

        """
        return Scene.RenderOperator.apply(material, self, res, spp, seed)

    def render_duvdxy(self, material, *, res, spp, seed=0):
        """Computes gradient texture coord w.r.t. screen-space coords

        Args:
            material (torch.Tensor): A tensor of shape (wu,hu,c) # TODO
            res (tuple): Resolution of the rendered image as a tuple (width, height).
            spp (int): Samples per pixel for rendering. Higher values increase quality at the
                       cost of more computation.
            seed (int): Seed for random number in rendering.

        Returns:
            torch.Tensor: A tensor (height, width, 4) storing (dudx,dvdx,dudy,dvdy) in each channel

        """
        return self.render_forward(material.detach(), res, spp, seed, integrator_kernel=render_uvgrad_kernel)
