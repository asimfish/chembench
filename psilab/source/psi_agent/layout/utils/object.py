import os
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

from base_utils.object import OmniObject
from pxr import Usd, UsdGeom

from .sdf import compute_sdf_from_obj, compute_sdf_from_obj_surface
from .transform_utils import farthest_point_sampling, get_bott_up_point, random_point

def load_and_prepare_mesh(obj_path, up_axis):
    if not os.path.exists(obj_path):
        return None
    mesh = trimesh.load(obj_path)
    print("1111111111111111111111111111111111111111111")
    print(mesh.bounds)
    if 'z' in up_axis:
        align_rotation = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    elif 'y' in up_axis:
        align_rotation = R.from_euler('xyz', [-90, 180, 0], degrees=True).as_matrix()
    elif 'x' in up_axis:
        align_rotation = R.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()
    else:
        align_rotation = R.from_euler('xyz', [-90, 180, 0], degrees=True).as_matrix()
    

    align_transform = np.eye(4)
    align_transform[:3, :3] = align_rotation
    mesh.apply_transform(align_transform)
    return mesh



def setup_sdf(mesh):
    _, sdf_voxels = compute_sdf_from_obj_surface(mesh)
    # create callable sdf function with interpolation

    min_corner = mesh.bounds[0]
    max_corner = mesh.bounds[1]

    x = np.linspace(min_corner[0], max_corner[0], sdf_voxels.shape[0])
    y = np.linspace(min_corner[1], max_corner[1], sdf_voxels.shape[1])
    z = np.linspace(min_corner[2], max_corner[2], sdf_voxels.shape[2])
    sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
    return sdf_func


class LayoutObject(OmniObject):
    def __init__(self, obj_info, use_sdf=False, N_collision_points=60, **kwargs):
        super().__init__(name=obj_info['object_id'], **kwargs)

        obj_dir = obj_info["obj_path"]
        up_aixs = obj_info["upAxis"]
        print("maobo scale =  obj_info ")
        # maobo update begin 
        scale =  obj_info["scale"]
        print(scale)
        # maobo update end
        if len(up_aixs) ==0:
            up_aixs = ['y']
            
    

        # self.mesh = load_and_prepare_mesh(obj_dir, up_aixs)
        print("get stage usd.stage.open() ++++++++++++++++++++++++")
        print(obj_dir)
        stage = Usd.Stage.Open(obj_dir)
       
        # 遍历所有 Mesh   Albert  get usd  mesh size  begin 
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                vertices = mesh.GetPointsAttr().Get()
                # 获取面索引
                faces = mesh.GetFaceVertexIndicesAttr().Get()
                vertices_np = np.array(vertices)
                min_bounds = vertices_np.min(axis=0)  # (xmin, ymin, zmin)
                max_bounds = vertices_np.max(axis=0)  # (xmax, ymax, zmax)
                # 计算尺寸（长宽高）
                size = max_bounds - min_bounds  # (width, height, depth)
                 
                self.size = size * scale 
                print(f"Mesh-----: {prim.GetPath()}, Size: {self.size}") 
                self.up_axis = up_aixs[0]     
        # 遍历所有 Mesh   Albert  get usd  mesh size  end 

        # if use_sdf:
        #     self.sdf = setup_sdf(self.mesh)
        
        # if self.mesh is not None:
            # mesh_points, _ =  trimesh.sample.sample_surface(self.mesh, 2000) # 表面采样
            # if mesh_points.shape[0] > N_collision_points:
            #     self.collision_points = farthest_point_sampling(mesh_points, N_collision_points) # 碰撞检测点

            #  Albert 已经注释掉了  不影响使用的  
            # self.anchor_points = {}
            # self.anchor_points['top'] = get_bott_up_point(mesh_points, 1.5,descending=False)
            # self.anchor_points['buttom'] = get_bott_up_point(mesh_points, 1.5,descending=True)
            


            # self.anchor_points['top'] = random_point(self.anchor_points['top'], 3)[np.newaxis, :]
            # self.anchor_points['buttom'] = random_point(self.anchor_points['buttom'], 3)[np.newaxis, :]


       