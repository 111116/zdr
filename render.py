"""Simple Differentiable Renderer using Luisa.

Currently only supports collocated direct lighting.

"""
import torch
import numpy as np
import math
import luisa
from luisa.mathtypes import *
import weakref
import imageio

# import gc

from .load_obj import read_obj, concat_triangles
from .recompute_normal import recompute_normal
from .uvgrad import render_uvgrad_kernel
from .collocated import render_collocated_kernel, render_collocated_backward_kernel
from .direct import render_direct_kernel, render_direct_backward_kernel
from .prb import render_path_kernel, render_path_backward_kernel
from .vertex import Vertex
from .envmap import load_envmap

# Using CUDA for interaction with PyTorch
luisa.init('cuda')

Camera = luisa.StructType(fov=float, origin=float3, target=float3, up=float3)


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

    """
    def __init__(self, models, integrator='direct'):
        self.load_geometry(models)
        self.camera = Camera(
            fov = 40 / 180 * 3.1415926,
            origin = float3(1.0, 0.5, 0.0),
            target = float3(0.0, 0.0, 0.0),
            up = float3(0.0, 1.0, 0.0)
        )
        # if integrator != 'collocated' and self.light_count == 0:
        #     raise RuntimeError("No light source detected!")
        integrators = {
            "path": (render_path_kernel, render_path_backward_kernel),
            "direct": (render_direct_kernel, render_direct_backward_kernel),
            "collocated": (render_collocated_kernel, render_collocated_backward_kernel),
        }
        self.forward_kernel, self.backward_kernel = integrators[integrator]
        self.use_tent_filter = True

    def load_geometry(self, models):
        self.accel = luisa.Accel()
        self.heap = luisa.BindlessArray()
        self.loaded_obj = {}
        inst_metadata = [] # information for each model, e.g. emission, bsdf type
        light_insts = [] # instances that should be sampled as a light source
        inst_trig_count = [] # number of triangles in each mesh instance

        for idx, model in enumerate(models):
            obj_file, transform, emission = model
            if transform is None:
                transform = luisa.float4x4(1.0)
            if emission is None:
                emission = float3(0.0)
            elif type(emission) in {int, float}:
                emission = float3(emission)
            if emission.x>0 or emission.y>0 or emission.z>0:
                light_insts.append(idx)
            inst_metadata.append(emission)

            def get_obj_geometry(obj_file):
                if obj_file not in self.loaded_obj:
                    vertices, faces = read_obj(obj_file)
                    vertex_buffer = luisa.Buffer(dtype=Vertex, size=len(vertices))
                    vertex_buffer.copy_from_array(np.array([(*v,*t,*n) for v,t,n in vertices], dtype=np.float32))
                    triangles = concat_triangles(faces)
                    triangle_buffer = luisa.buffer(triangles)
                    # recompute if vertex normal isn't available
                    if math.isnan(vertices[0][2][0]):
                        print("computing vertex normal vectors from faces...")
                        recompute_normal(vertex_buffer, triangle_buffer)
                    self.loaded_obj[obj_file] = (vertex_buffer, triangle_buffer)
                return self.loaded_obj[obj_file]

            vertex_buffer, triangle_buffer = get_obj_geometry(obj_file)
            inst_trig_count.append(triangle_buffer.size//3) # number of triangles
            self.accel.add(vertex_buffer, triangle_buffer, transform=transform)
            self.heap.emplace(idx*2+0, triangle_buffer)
            self.heap.emplace(idx*2+1, vertex_buffer)

        self.inst_count = idx + 1
        if self.inst_count > 10000:
            raise RuntimeError('exceeding maximum number of mesh instances')
        # light info
        self.env_count = 0
        self.light_count = len(light_insts)
        self.emissions = inst_metadata
        self.inst_metadata_buffer = luisa.buffer(inst_metadata)
        self.light_insts_buffer = luisa.buffer(light_insts + [0]*(self.inst_count - self.light_count))
        # put auxiliary buffers into bindless array
        self.heap.emplace(23333, self.inst_metadata_buffer)
        if self.light_count > 0:
            self.heap.emplace(23334, self.light_insts_buffer)
        self.heap.emplace(23335, luisa.buffer(inst_trig_count))
        self.accel.update()
        self.heap.update()
    
    def update_lights(self, emissions):
        """Rewrite emission values of each mesh in scene

        This function is primarily used for dynamically switching lights
        on and off in a scene like a light stage.

        Args:
            emissions (list of float3): emission color of each object

        """
        self.emissions = emissions # for record
        assert len(emissions) == self.inst_count
        emissions = [float3(0) if x is None else float3(x) if type(x) in {int, float} else x for x in emissions]
        # update emission of each object in inst_metadata & rewrite light_insts
        inst_metadata = emissions
        self.inst_metadata_buffer.copy_from(inst_metadata)
        light_insts = [i for i,x in enumerate(emissions) if x.x>0 or x.y>0 or x.z>0]
        self.light_count = len(light_insts)
        self.light_insts_buffer.copy_from(light_insts + [0]*(self.inst_count - self.light_count))
    
    def add_envmap(self, filename, compensate_mis=True):
        img = np.asarray(imageio.imread(filename))
        if img.shape[2] == 3:
            # convert RGB to RGBA
            img = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
        load_envmap(self.heap, img, compensate_mis=compensate_mis)
        self.env_count = 1


    def render_forward(self, material, res, spp, seed, kernel=None):
        assert material.ndim == 3 and material.shape[2] == 4
        texture_res = material.shape[0:2]
        material_buffer = luisa.Buffer.from_dlpack(material.detach().flatten())
        image = torch.empty((res[1], res[0], 4), dtype=torch.float32, device='cuda')
        image_buffer = luisa.Buffer.from_dlpack(image.reshape((res[0]*res[1], 4)))
        torch.cuda.synchronize() # make sure material is ready
        if kernel is None:
            kernel = self.forward_kernel
        kernel(image_buffer,
            self.heap, self.accel, self.light_count, self.env_count,
            material_buffer, int2(*texture_res), self.camera,
            spp, seed, self.use_tent_filter, dispatch_size=res)
        luisa.synchronize()
        return image
    
    
    def render_backward(self, grad_output, d_material, material, res, spp, seed, kernel=None):
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
        if kernel is None:
            kernel = self.backward_kernel
        kernel(d_image,
            self.heap, self.accel, self.light_count, self.env_count,
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
            ctx.emissions = self.emissions
            return self.render_forward(material.detach(), *args)

        @staticmethod
        def backward(ctx, grad_output):
            # print("a", len(gc.get_referrers(grad_output)))
            # scene.camera may be modified between forward and backward.
            # workaround: save and load camera state
            camera_bak = ctx.scene().camera
            ctx.scene().camera = ctx.camera
            ctx.scene().update_lights(ctx.emissions)
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
        return self.render_forward(material.detach(), res, spp, seed, kernel=render_uvgrad_kernel)
