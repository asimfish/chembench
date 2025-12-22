import cv2
import copy
import time
import trimesh
import pyrender
import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from .data_utils import create_point_cloud_from_depth_image, mask2bbox
from .transforms import calculate_rotation_matrix


def rotate_180_along_axis(target_affine, rot_axis='z'):
    if rot_axis == 'z':
        R_180 = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    elif rot_axis == 'y':
        R_180 = np.array([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]])
    elif rot_axis == 'x':
        R_180 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    else:
        assert False, "Invalid rotation axis. Please choose from 'x', 'y', 'z'."

    target_rotation = target_affine[:3, :3]
    target_rotation_2 = np.dot(target_rotation, R_180)

    target_affine_2 = np.eye(4)
    target_affine_2[:3, :3] = target_rotation_2
    target_affine_2[:3, 3] = target_affine[:3, 3] 
    return target_affine_2




def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)

def create_arrow(scale=0.1):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)





def create_arrow_o3d(origin=[0, 0, 0], end=None, vec=None, length=0.15, color=(1,0,0)):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    assert(not np.all(end == origin))
    if end is not None:
        vec = end - origin


    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.004, 
        cone_radius=0.012, 
        cylinder_height=length, 
        cone_height=0.03)


    vec = np.array(vec) / np.linalg.norm(vec)

    R = calculate_rotation_matrix([0,0,1], vec)
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3,3] = origin
    mesh.transform(pose)

    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_pts_o3d(depth, cam_info, color=None, depth_range=None, mask=None):
    cloud = create_point_cloud_from_depth_image(depth, cam_info)
    if depth_range is not None:
        depth_mask = (depth > depth_range[0]) & (depth < depth_range[1])
        mask = depth_mask & mask if mask is not None else depth_mask


    if mask is not None:
        cloud = cloud[mask]
        if color is not None:
            color = color[mask]/255.0
    else:
        cloud = cloud.reshape(-1,3)
        if color is not None:
            color = color.reshape(-1,3)/255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def create_pts_origin_o3d(pts, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if color is not None:
        color = color.reshape(-1,3)/255.0
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd



def create_3d_coordinate_frame_o3d(transformation_matrix, axis_length):
    """
    绘制3D坐标轴
    :param transformation_matrix: 4x4 矩阵，包含了坐标轴的朝向和原点在世界坐标系下的坐标
    :param axis_length: 坐标轴的长度
    """
    # 创建坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)

    # 应用变换
    coordinate_frame.transform(transformation_matrix)
    return coordinate_frame

def create_3d_axis_o3d(transformation_matrix, axis_length, axis='x', type='active'):
    """
    绘制3D坐标轴
    :param transformation_matrix: 4x4 矩阵，包含了坐标轴的朝向和原点在世界坐标系下的坐标
    :param axis_length: 坐标轴的长度
    """
    half_length = axis_length + 0.02

    type = 'passive'
   # 设置箭头颜色
    if axis == 'x':
        if type=='active':
            origin = np.array([0,0,0])
            end = np.array([1,0,0]) * half_length
        elif type=='passive':
            end = np.array([0,0,0])
            origin = np.array([-1,0,0]) * half_length
        arrow = create_arrow_o3d(origin=origin, end=end, length=axis_length*2, color=[1, 0, 0])
    elif axis == 'y':
        if type=='active':
            origin = np.array([0,0,0])
            end = np.array([0,1,0]) * half_length
        elif type=='passive':
            end = np.array([0,0,0])
            origin = np.array([0,-1,0]) * half_length
        arrow = create_arrow_o3d(origin=origin, end=end, length=axis_length*2, color=[1, 0, 0])
    elif axis == 'z':
        if type=='active':
            origin = np.array([0,0,0])
            end = np.array([0,0,1]) * half_length
        elif type=='passive':
            end = np.array([0,0,0])
            origin = np.array([0,0,-1]) * half_length
        arrow = create_arrow_o3d(origin=origin, end=end, length=axis_length*2, color=[1, 0, 0])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # 应用变换
    arrow.transform(transformation_matrix)
    return arrow


def create_trimesh_o3d(mesh, pose=None):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()

    if mesh.visual.kind == 'vertex':
        vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # 转换为 [0, 1] 范围
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    elif mesh.visual.kind == 'texture':
        texture_image = mesh.visual.material.image
        mesh_o3d.triangle_uvs = o3d.utility.Vector2dVector(mesh.visual.uv)
        mesh_o3d.textures = [o3d.geometry.Image(np.array(texture_image))]

    if pose is not None:
        mesh_o3d.transform(pose)
    return mesh_o3d

def visual_geometries_o3d(geometries, cam_info, cam_pose=None, gui=True, exist_vis=None):    
    if exist_vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=cam_info['W'], height=cam_info['H'])
    else:
        vis = exist_vis
        vis.clear_geometries()
            
    for geometry in geometries:
        vis.add_geometry(geometry)

    # set camera
    view_control = vis.get_view_control()
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic()

    intrinsics.set_intrinsics(
        width=cam_info['W'], height=cam_info['H'], 
        fx=cam_info['K'][0][0], fy=cam_info['K'][1][1], cx=cam_info['W']//2-0.5, cy=cam_info['H']//2-0.5,
        # near=0.01,  # 或者设为其他较小值
        # far=1000.0  # 远剪裁平面
        )
    param = o3d.cuda.pybind.camera.PinholeCameraParameters()
    param.extrinsic = np.eye(4) if cam_pose is None else cam_pose
    param.intrinsic = intrinsics
    view_control.convert_from_pinhole_camera_parameters(param)
    
    vis.get_render_option().point_size = 0.1  # 设置点的大小
    if gui:
        vis.run()
    # render image
    buffer = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(buffer)
    image = (image * 255).astype(np.uint8)
    time.sleep(0.1)
    if exist_vis is None:
        vis.destroy_window()

    return image


def render_object_depth_map(pts, cam_info, cam_pose=None, gui=False, surface_reconstruction=False):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)

    # cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
    # point_cloud = point_cloud.select_by_index(ind)

    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=10))
    point_cloud.orient_normals_consistent_tangent_plane(100)



    if not surface_reconstruction:
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)

        camera_position = np.array([0.0, 0.0, 0.0]) if cam_pose is None else cam_pose[:3, 3]
        view_directions = points - camera_position
        view_directions /= np.linalg.norm(view_directions, axis=1, keepdims=True)

        dot_products = np.einsum('ij,ij->i', normals, view_directions)

        threshold = 0.0  
        pts_visible_mask = dot_products > threshold

        points = points[pts_visible_mask]


        points_cam = (cam_info['K'] @ points[:, :3].T).T
        depth = points_cam[:, 2]
        x_pixel = np.round(points_cam[:, 0] / depth).astype(int)
        y_pixel = np.round(points_cam[:, 1] / depth).astype(int)
        _valid_mask = (x_pixel >= 0) & (x_pixel < cam_info['W']) & (y_pixel >= 0) & (y_pixel < cam_info['H'])
        x_pixel, y_pixel, depth = x_pixel[_valid_mask], y_pixel[_valid_mask], depth[_valid_mask]

        depth_map = np.zeros((cam_info['H'], cam_info['W']), dtype=np.float32)
        x_map, y_map = np.zeros_like(depth_map), np.zeros_like(depth_map)


        depth_map[y_pixel, x_pixel] = depth
        x_map[y_pixel, x_pixel] = points[:,0][_valid_mask]
        y_map[y_pixel, x_pixel] = points[:,1][_valid_mask]


        obj_mask_origin = depth_map > 0
        obj_mask = cv2.morphologyEx(obj_mask_origin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9,9), np.uint8)) > 0


        depth_map = cv2.inpaint(depth_map, (~obj_mask_origin).astype(np.uint8), inpaintRadius=5, flags=cv2.INPAINT_NS)
        depth_map[~obj_mask] = 0



        return depth_map, obj_mask, pts_visible_mask





    else:
        # 从点云创建网格
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.05)
        mesh.compute_vertex_normals()
        # radii = [0.05, 0.05, 0.04, 0.05]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     point_cloud, o3d.utility.DoubleVector(radii))
        # o3d.visualization.draw_geometries([pcd, rec_mesh])





        # # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
        # exit()

        # # 从点云创建网格
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.05)
        mesh.compute_vertex_normals()
        # radii = [0.05, 0.05, 0.04, 0.05]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     point_cloud, o3d.utility.DoubleVector(radii))
        # o3d.visualization.draw_geometries([pcd, rec_mesh])

        # 设置相机内参
        intrinsics = o3d.camera.PinholeCameraIntrinsic()

        intrinsics.set_intrinsics(width=cam_info['W'], height=cam_info['H'], fx=cam_info['K'][0][0], fy=cam_info['K'][1][1], cx=cam_info['W']//2-0.5, cy=cam_info['H']//2-0.5)
        param = o3d.cuda.pybind.camera.PinholeCameraParameters()
        param.extrinsic = np.eye(4) if cam_pose is None else cam_pose
        param.intrinsic = intrinsics
        # 创建渲染器
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=cam_info['W'], height=cam_info['H'], visible=gui==True)
        vis.add_geometry(mesh)
        view_control = vis.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(param)

        # 渲染深度图
        vis.poll_events()
        vis.update_renderer()
        if gui:
            vis.run()
        depth_image = vis.capture_depth_float_buffer(do_render=True)

        # 转换为numpy数组
        depth_map = np.asarray(depth_image)

        # 关闭渲染器
        vis.destroy_window()
        del vis

    return depth_map


def render_mesh_with_pyrender(meshs, instrinsics, cam_pose=np.eye(4), poses=np.eye(4), H=512, W=512, light_level=1):
    cam_pose = rotate_180_along_axis(cam_pose, 'x')

    scene = pyrender.Scene()

    if not isinstance(meshs, list):
        meshs = [meshs]
        poses = [poses]
    for mesh, pose in zip(meshs, poses):
        scene.add(pyrender.Mesh.from_trimesh(mesh), pose=pose)
    # scene.add(pyrender.Mesh.from_trimesh(mesh), pose=obj_pose)


    camera = pyrender.IntrinsicsCamera(
        fx=instrinsics[0][0],
        fy=instrinsics[1][1],
        cx=instrinsics[0][2],
        cy=instrinsics[1][2]
    )
    scene.add(camera, pose=cam_pose)


    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5*light_level)
    light_pose = cam_pose.copy()
    light_pose[:3, 3] += np.array([0, 0, 0])  # 在相机上方2个单位
    scene.add(light, pose=light_pose)


    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

 
    color, depth = renderer.render(scene)
    renderer.delete()
    return color, depth

def render_mesh_with_pyrender_at_scene(observation, meshs, poses, cam_pose=np.eye(4), extra_mask_visual=None, extra_mask_crop=None, light_level=3):
    image, depth, cam_info = observation['image'], observation['depth'], observation['cam_info']
    obj_rgb, obj_depth = render_mesh_with_pyrender(meshs, cam_info['K'], poses=poses, H=cam_info['H'], W=cam_info['W'], light_level=light_level)
        
    obj_mask = obj_depth > 0

    scene_depth = copy.deepcopy(depth)
    _mask = (scene_depth > 0.1) & (scene_depth < 2)
    scene_depth[~_mask] = 1000
    if extra_mask_visual is not None:
        scene_depth[extra_mask_visual] = 1000

    visible_mask = (obj_depth < scene_depth) & obj_mask
    blocked_mask = (obj_depth >= scene_depth) & obj_mask

    alpha = 0.7
    img = image.copy()
    for c in range(3):  # 假设图像是彩色的
        img[..., c] = np.where(
            visible_mask, 
            obj_rgb[..., c],
            img[..., c])

        img[..., c] = np.where(
            blocked_mask, 
            cv2.addWeighted(obj_rgb[..., c], 1-alpha, img[..., c], alpha, 0),
            img[..., c])


    if extra_mask_crop is not None:
        all_mask = extra_mask_crop | obj_mask
        x0, y0, x1, y1 = mask2bbox(all_mask)
        H, W = cam_info['H'], cam_info['W']
        if (x1 - x0) * (y1 - y0) < 0.5 * W * H:
            x0 = max(0, x0 - 30)
            y0 = max(0, y0 - 30)
            x1 = min(W-1, x1 + 30)
            y1 = min(H-1, y1 + 30)
            img = img[y0:y1, x0:x1]
    return img

def o3d_geometry_to_trimesh(geometry):
    vertices = np.asarray(geometry.vertices)
    faces = np.asarray(geometry.triangles)
    vertex_colors = (np.asarray(geometry.vertex_colors)*255).astype(np.uint8) if geometry.has_vertex_colors() else None

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    return mesh



def get_gripper_mesh(gripper_pose, width=None, o3d2trimesh=True, score=1.0):
    from graspnetAPI.graspnet_eval import GraspGroup
    '''
    Args:
        grasp_pose: (N, 4, 4)
        width: (N,)
    Return:
        gripper_mesh: List[trimesh.Trimesh]
    '''
    N = gripper_pose.shape[0]

    rotation = gripper_pose[:, :3, :3].reshape(N, -1)
    translation = gripper_pose[:, :3, 3]

    width = width[:, np.newaxis] if width is not None else 0.1 * np.ones((N, 1))
    height = 0.02 * np.ones_like(width)
    depth = np.zeros_like(width)
    score = np.ones_like(width) * score
    obj_id = -1 * np.ones_like(width)


    grasp_group = np.concatenate([score, width, height, depth, rotation, translation, obj_id], axis=-1)    
    gripper_mesh = GraspGroup(grasp_group).to_open3d_geometry_list()
    if o3d2trimesh:
        gripper_mesh = [o3d_geometry_to_trimesh(gripper) for gripper in gripper_mesh]
    return gripper_mesh