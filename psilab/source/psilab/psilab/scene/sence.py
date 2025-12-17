# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from collections.abc import Sequence

""" Common Modules  """ 
import torch
import random
import re

""" Omniverse Modules  """ 
import carb
import omni.usd
from pxr import Sdf
from pxr import Gf
""" IsaacSim Modules  """ 
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.utils import clone
from isaacsim.core.cloner import GridCloner

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyMaterialCfg,MassPropertiesCfg
from isaaclab.sim.schemas import modify_physics_material_properties,modify_mass_properties
from isaaclab.utils.math import quat_from_euler_xyz,quat_conjugate,quat_apply,quat_mul
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.sensors import (
    SensorBaseCfg,
    CameraCfg,
    ContactSensorCfg, 
    FrameTransformerCfg,
    Camera,
    TiledCamera,
    TiledCameraCfg
)

from pxr import Sdf, Usd


""" Psilab Modules  """ 
from psilab.scene.sence_cfg import SceneCfg
from psilab.assets.robot.robot_base_cfg import RobotBaseCfg
from psilab.assets.robot.robot_base import RobotBase
from psilab.assets.particle_cloth.particle_cloth_cfg import ParticleClothCfg
from psilab.assets.particle_cloth.particle_cloth import ParticleCloth

from psilab.assets.light.light_cfg import LightBaseCfg
from psilab.assets.light.light import Light
from psilab.assets.articulated_object.articulated_object import ArticulatedObject
from psilab.assets import ArticulatedObjectCfg
from psilab.random.random_cfg import RandomCfg
from psilab.random.visual_material_random_cfg import VisualMaterialRandomCfg
from psilab.random.rigid_physic_material_random_cfg import RigidPhysicMaterialRandomCfg
from psilab.random.position_random_cfg import PositionRandomCfg
from psilab.random.orientation_random_cfg import OrientationRandomCfg
from psilab.random.joint_random_cfg import JointRandomCfg
from psilab.random.mass_random_cfg import MassRandomCfg
from psilab.random.prim_random_cfg import PrimRandomCfg
from psilab.sim.utils import set_attribute_on_usd_prim
from isaaclab.sim import utils

class Scene(InteractiveScene):
    """
    A scene that contains entities added to the simulation.
    """

    def __init__(self, cfg: SceneCfg):
        self.cfg = cfg

        # initiallize robot elements
        self._robots:dict[str, RobotBase] = dict()
        self._cameras:dict[str, Camera] = dict()
        self._tiled_cameras:dict[str, TiledCamera] = dict()
        self._lights: dict[str,Light] = dict()
        self._articulated_objects:  dict[str,ArticulatedObject] = dict()
        self._visualizer:VisualizationMarkers = None # type: ignore
        self._particle_cloths:dict[str,ParticleCloth] = dict()
        super().__init__(self.cfg)
    # TODO: add state of objects created by scene
    def get_state(self, is_relative: bool = False) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Returns the state of the scene entities.

        Args:
            is_relative: If set to True, the state is considered relative to the environment origins.

        Returns:
            A dictionary of the state of the scene entities.
        """
        state = super().get_state(is_relative)

        return state

    def reset(self, env_ids: Sequence[int] | None = None):

        if env_ids is None:
            env_ids = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device) # type: ignore

        super().reset(env_ids)

        # robots
        for robot in self._robots.values():
            robot.reset(env_ids)

        # articulated object
        for articulated_object in self._articulated_objects.values():
            articulated_object.reset(env_ids)


        # cameras
        for camera in self._cameras.values():
            camera.reset(env_ids)
        for tile_camera in self._tiled_cameras.values():
            tile_camera.reset(env_ids)

        # rigid object
        for rigid_name,rigid_object in self._rigid_objects.items():
            pos = self.cfg.rigid_objects_cfg[rigid_name].init_state.pos
            rot = self.cfg.rigid_objects_cfg[rigid_name].init_state.rot
            lin_vel = self.cfg.rigid_objects_cfg[rigid_name].init_state.lin_vel
            ang_vel = self.cfg.rigid_objects_cfg[rigid_name].init_state.ang_vel
            root_state_init = torch.tensor(list(pos)+list(rot)+list(lin_vel)+list(ang_vel),device=self.device).unsqueeze(0).repeat(self.num_envs,1) # type: ignore
            root_state_init[env_ids,:3]+=self.env_origins[env_ids,:]
            # add height offset
            if self.rigid_objects[rigid_name].cfg.enable_height_offset and self.rigid_objects[rigid_name].cfg.height_offset:
                root_state_init[:,2]+= self.rigid_objects[rigid_name].cfg.height_offset * torch.ones_like(root_state_init[:,2])
            #
            rigid_object.write_root_state_to_sim(root_state_init[env_ids,:],env_ids = env_ids)

        # 
        for particle_cloth in self._particle_cloths.values():
            particle_cloth.reset(env_ids)

        # apply random
        self._apply_random(env_ids)

    # TODO: add reset to for objects created by scene
    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        is_relative: bool = False,
    ):
        super().reset_to(state,env_ids,is_relative)
   
    def write_data_to_sim(self):

        super().write_data_to_sim()
        
        for robot in self._robots.values():
            robot.write_data_to_sim()
             
    def update(self, dt: float) -> None:
        
        super().update(dt)

        # update robot
        for robot in self._robots.values():
            robot.update(dt)

            # cameras in robot
            for camera in robot.cameras.values():
                camera.update(dt)
            for tiled_camera in robot.tiled_cameras.values():
                tiled_camera.update(dt)
        
        # update articulated object
        for articulated_object in self._articulated_objects.values():
            articulated_object.update(dt)
        
        # cameras
        for camera in self._cameras.values():
            camera.update(dt, force_recompute=not self.cfg.lazy_sensor_update)
        for tiled_camera in self._tiled_cameras.values():
            tiled_camera.update(dt)

        # update particle cloth
        for particle_cloth in self._particle_cloths.values():
            particle_cloth.update(dt)

    # TODO: add keys of objects created by scene
    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.
        # """
        all_keys = super().keys()
        # add robot and camera
        # for asset_family in [
        #     self._robots,
        #     self._cameras,
        # ]:
        #     all_keys += list(asset_family.keys())
        return all_keys

    def _add_entities_from_cfg(self):

        """
        Add scene entities from the config.
        Overwrite function in base class as config list description will cause ValueError in case class.
        """
        # ********* Super _add_entities_from_cfg
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            # resolve regex
            if hasattr(asset_cfg, "prim_path"):
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            # create asset
            if isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, DeformableObjectCfg):
                self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCfg):
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCollectionCfg):
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                        asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                        self._global_prim_paths += asset_paths
            elif isinstance(asset_cfg, SensorBaseCfg):
                # Update target frame path(s)' regex name space for FrameTransformer
                if isinstance(asset_cfg, FrameTransformerCfg):
                    updated_target_frames = []
                    for target_frame in asset_cfg.target_frames:
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):
                    updated_filter_prim_paths_expr = []
                    for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                        updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                    asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr

                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
            # Author: Feng Yunduo 2025-07-05 start
            # 新增光照-全局
            elif isinstance(asset_cfg, LightBaseCfg):
                if asset_cfg.spawn is not None:
                    self._lights[asset_name]=asset_cfg.class_type(asset_cfg)
            # Author: Feng Yunduo 2025-07-05 End
            
            elif isinstance(asset_cfg, AssetBaseCfg):
                # manually spawn asset
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(
                        asset_cfg.prim_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,
                        orientation=asset_cfg.init_state.rot,
                    )
                # store xform prim view corresponding to this asset
                # all prims in the scene are Xform prims (i.e. have a transform component)
                # if isinstance(asset_cfg, LightBaseCfg):
                #     self._lights[asset_name] = XFormPrim(asset_cfg.prim_path, reset_xform_properties=False)
                # else:
                self._extras[asset_name] = XFormPrim(asset_cfg.prim_path, reset_xform_properties=False)

            # Author: Feng Yunduo 2025-02-08 start
            
            # 新增配置字典类型
            elif isinstance(asset_cfg, VisualizationMarkersCfg):
                self._visualizer = VisualizationMarkers(asset_cfg)
            elif isinstance(asset_cfg, dict):
                for sub_asset_name, sub_asset_cfg in asset_cfg.items():
                    if isinstance(sub_asset_cfg, RobotBaseCfg):
                        # terrains are special entities since they define environment origins
                        self._robots[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                        # ********* Add Camera entities from the robot config ********
                        # normal camera
                        if sub_asset_cfg.cameras is not None: 
                            for camera_name, camera_cfg in sub_asset_cfg.cameras.items():
                                self._robots[sub_asset_name].cameras[camera_name]=Camera(camera_cfg)
                        # Tiled camera
                        if sub_asset_cfg.tiled_cameras is not None: 
                            for camera_name, camera_cfg in sub_asset_cfg.tiled_cameras.items():
                                self._robots[sub_asset_name].tiled_cameras[camera_name]=TiledCamera(camera_cfg)
                    elif isinstance(sub_asset_cfg, ArticulatedObjectCfg):
                        # terrains are special entities since they define environment origins
                        self._articulated_objects[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)                    
                    elif isinstance(sub_asset_cfg,RigidObjectCfg):
                        # add rigid object
                        self._rigid_objects[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)   
                    elif isinstance(sub_asset_cfg, DeformableObjectCfg):
                        self._deformable_objects[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                    elif isinstance(sub_asset_cfg, LightBaseCfg):
                        if sub_asset_cfg.spawn is not None:
                            self._lights[sub_asset_name]=sub_asset_cfg.class_type(sub_asset_cfg)
                        else:
                            self._lights[sub_asset_name]=sub_asset_cfg.class_type(sub_asset_cfg)
                    elif isinstance(sub_asset_cfg, ParticleClothCfg):
                        self._particle_cloths[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg) 
                        # pass
                    elif isinstance(sub_asset_cfg, AssetBaseCfg):
                        # manually spawn asset
                        if sub_asset_cfg.spawn is not None:
                            sub_asset_cfg.spawn.func(
                                sub_asset_cfg.prim_path,
                                sub_asset_cfg.spawn,
                                translation=sub_asset_cfg.init_state.pos,
                                orientation=sub_asset_cfg.init_state.rot,
                            )
                        # store xform prim view corresponding to this asset
                        # all prims in the scene are Xform prims (i.e. have a transform component)
                        self._extras[sub_asset_name] = XFormPrim(sub_asset_cfg.prim_path, reset_xform_properties=False)
                    elif isinstance(sub_asset_cfg, SensorBaseCfg):
                        if isinstance(sub_asset_cfg, TiledCameraCfg):
                            self._tiled_cameras[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                        elif isinstance(sub_asset_cfg, CameraCfg):
                            self._cameras[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                        elif isinstance(sub_asset_cfg,ContactSensorCfg):
                            self._sensors[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
            elif isinstance(asset_cfg, RandomCfg):
                # do nothing with random config
                pass
            # Author: Feng Yunduo 2025-02-08 end
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
 
    def _apply_random(self, env_ids: Sequence[int] | None = None):
        
        # pass
        if self.cfg.random is None:
            return
        # robots
        self._apply_robots_random(env_ids)
        # rigid objects
        self._apply_rigid_objects_random(env_ids)
        # light
        self._apply_lights_random()
        # articulated objects
        self._apply_articulated_objects_random(env_ids)

    def _apply_robots_random(self, env_ids: Sequence[int] | None = None):
        # 
        if self.cfg.random.robots_cfg is None: # type: ignore
            return
        #
        for robot_name, random_cfg in self.cfg.random.robots_cfg.items(): # type: ignore
            # ignore random config of rigid objects which are not in scene and no material random config
            if robot_name not in self.cfg.robots_cfg.keys(): 
                continue
            # transform
            self._apply_robots_transform_random(env_ids,robot_name,random_cfg.position,random_cfg.orientation)
            # visual material
            if random_cfg.visual_material:
                self._apply_visual_material_random(env_ids,self.robots[robot_name].cfg.prim_path,random_cfg.visual_material)
            # rigid physics material
            if random_cfg.physics_material:
                self._apply_rigid_physics_material_random(env_ids,self.robots[robot_name].cfg.prim_path,random_cfg.physics_material)
            # joint 
            if random_cfg.joint:
                self._apply_joint_random(env_ids,self.robots[robot_name],random_cfg.joint)
            # prim random
            if random_cfg.prim:
                self._apply_robots_prim_random(env_ids,robot_name,random_cfg.prim)

    def _apply_robots_transform_random(self, env_ids: Sequence[int] | None = None, name : str | None = None, pos_config: PositionRandomCfg | None = None, ori_config: OrientationRandomCfg | None = None):
        #
        if name is None or (pos_config is None and pos_config is None):
            return 
        # get default state
        state = self._robots[name].data.default_root_state.clone()
        #
        if pos_config:
            #
            enable = torch.tensor(pos_config.enable,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
            # compute position offset
            if pos_config.type == "list":
                # random index of list
                indexs = torch.randint(0,len(pos_config.offset_list),[self.num_envs],device = self.device) # type: ignore
                offset = torch.tensor(pos_config.offset_list,device = self.device)
                offset = torch.index_select(offset, 0, indexs)
                offset = torch.mul(offset, enable) 
            else:
                offset_range = torch.tensor(pos_config.offset_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
                offset = 2.0 * (torch.rand((self.num_envs,3),device=self.device) - 0.5)
                offset = offset.mul(offset_range)
                offset = torch.mul(offset, enable) 
            # 
            state[:,:3] = state[:,:3] + offset + self.env_origins
        #
        if ori_config:
            #
            eular_base_index = torch.randint(0,len(ori_config.eular_base),[self.num_envs],device = self.device) # type: ignore
            eular_base = torch.tensor(ori_config.eular_base,device = self.device)

            eular_base = torch.index_select(eular_base, 0, eular_base_index)

            enable = torch.tensor(ori_config.enable,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
            # compute orientation 
            if ori_config.type == "list":
                # random index of list
                eular_list = torch.tensor(ori_config.eular_list,device = self.device)
                eular_list = torch.index_select(eular_list, 0, eular_base_index)
                # 
                eular_num = len(ori_config.eular_list[0]) # type: ignore
                indexs = torch.randint(0,eular_num,[self.num_envs],device = self.device) # type: ignore
                eular = eular_list[:,0,:].squeeze(1)
                for i in range(self.num_envs):
                    eular[i,:] = torch.index_select(eular_list[i,:,:], 0, indexs[i])
                # eular = torch.index_select(eular_list, 1, indexs[])
                eular = torch.mul(eular, enable)
            else:
                eular_range = torch.tensor(ori_config.eular_range,device = self.device)
                eular_range = torch.index_select(eular_range, 0, eular_base_index)
                eular = 2.0 * (torch.rand((self.num_envs,3),device=self.device) - 0.5)
                eular = eular.mul(eular_range)
                eular = torch.mul(eular, enable) 
            #
            eular = eular_base + eular
            # print(eular_base[0,:])
            # eular to quat
            quat = quat_from_euler_xyz(eular[:,0],eular[:,1],eular[:,2])
            # eular to quat
            state[:,3:7] = quat
            # TODO: does robot need height offset?
            # pos_z_offset = torch.tensor(ori_config.pos_z_offset,device = self.device)
            # pos_z_offset = torch.index_select(pos_z_offset, 0, eular_base_index)
            #
        #
        #
        # state[:,2]+=pos_z_offset

        # state.
        self._robots[name].write_root_state_to_sim(state[env_ids,:],env_ids)

    def _apply_robots_prim_random(self, env_ids: Sequence[int] | None = None, name : str | None = None, prim_random_config:dict[str,PrimRandomCfg]  | None = None):

        #
        if name is None or (prim_random_config is None):
            return 
        
        #
        # Test change prim translate
        robot_prim_paths = sim_utils.find_matching_prim_paths(self.cfg.robots_cfg[name].prim_path) # type: ignore
        # iterate all robot
        for robot_prim_path in robot_prim_paths:
            # get robot prim 
            robot_prim = prim_utils.get_prim_at_path(robot_prim_path)
            temp = re.search(r"env_[0-9]+",robot_prim_path).span()
            env_index = int(robot_prim_path[temp[0]:temp[1]].split("_")[-1])
            # get robot translate first
            robot_pos = self.robots[name].data.root_link_pos_w[env_index,:].cpu().tolist()
            robot_quat = self.robots[name].data.root_link_quat_w[env_index,:].cpu().tolist()
            # iterate all prim of robot
            for prim_name,prim_cfg in prim_random_config.items():
                # get prim first
                prim = prim_utils.get_prim_at_path(robot_prim_path+"/" + prim_name)
                # position offset of config
                if True in prim_cfg.position.enable: # type: ignore
                    # 
                    if prim_cfg.position.type == "list": # type: ignore
                        # random index of list
                        index = random.randint(0,len(prim_cfg.position.offset_list)) # type: ignore
                        offset = prim_cfg.position.offset_list[index] # type: ignore
                    else:
                        offset_range = prim_cfg.position.offset_range # type: ignore
                        offset = [2.0 * (random.random() - 0.5) * offset_range[0], # type: ignore
                                2.0 * (random.random() - 0.5) * offset_range[1], # type: ignore
                                2.0 * (random.random() - 0.5) * offset_range[2]] # type: ignore
                    #
                    pos_offset = [(offset[i] if prim_cfg.position.enable[i] else 0) for i in range(3)] # type: ignore
                else:
                    pos_offset = [0,0,0]
                # orientation offset of config
                if True in prim_cfg.orientation.enable: # type: ignore
                    # 
                    if prim_cfg.orientation.type == "list": # type: ignore
                        # random index of list
                        index = random.randint(0,len(prim_cfg.orientation.eular_list[0])) # type: ignore
                        ori_offset = prim_cfg.orientation.eular_list[0][index] # type: ignore
                    else:
                        ori_offset_range = prim_cfg.orientation.eular_range[0] # type: ignore
                        ori_offset = [2.0 * (random.random() - 0.5) * ori_offset_range[0], # type: ignore
                                2.0 * (random.random() - 0.5) * ori_offset_range[1], # type: ignore
                                2.0 * (random.random() - 0.5) * ori_offset_range[2]] # type: ignore
                else:
                    ori_offset = [0,0,0]
                # compute pos offset of robot frame
                pos_offset = [prim_cfg.position_initial[i] + pos_offset[i] for i in range(3)]
                # compute pos offset of world frame            
                prim_pos = (quat_apply(
                    torch.tensor(robot_quat,device = self.device),
                    torch.tensor(pos_offset,device = self.device)) + \
                    torch.tensor(robot_pos,device = self.device) - \
                    self.env_origins[env_index,:]
                    ).cpu().tolist()  

                #
                translate_attr = prim.GetAttribute("xformOp:translate")
                translate_attr.Set(tuple(prim_pos)) # type: ignore
                #
                ori_offset_quat = quat_from_euler_xyz(
                    torch.tensor(ori_offset[0],device = self.device),
                    torch.tensor(ori_offset[1],device = self.device),
                    torch.tensor(ori_offset[2],device = self.device))
                # compute quat from robot frame to camera frame 
                quat_robot_to_camera = quat_mul(
                    torch.tensor(prim_cfg.orientation_initial,device = self.device),
                    torch.tensor(robot_quat,device = self.device))
                #
                ori_new = quat_mul(
                    ori_offset_quat,
                    quat_robot_to_camera).cpu().tolist()
                #
                orient_attr = prim.GetAttribute("xformOp:orient")
                orient_new = orient_attr.Get()
                orient_new.real  = ori_new[0]
                orient_new.imaginary = tuple(ori_new[1:])
                orient_attr.Set(orient_new)
                
    def _apply_lights_random(self, env_ids: Sequence[int] | None = None):
        
        # global light 
        if self.cfg.random and self.cfg.random.global_light_cfg:
            # 
            random_cfg = self.cfg.random.global_light_cfg
            # get prim first
            prim = prim_utils.get_prim_at_path(self.cfg.global_light_cfg.prim_path)
            # intensity
            if random_cfg.random_intensity:
                if self.cfg.random.global_light_cfg.random_type == "list":
                    index = random.randint(0,len(self.cfg.random.global_light_cfg.intensity_list)-1) # type: ignore
                    intensity = self.cfg.random.global_light_cfg.intensity_list[index]# type: ignore
                else:
                    intensity= random.randint(random_cfg.intensity_range[0],random_cfg.intensity_range[1]) # type: ignore
                # 
                # change property: intensity
                intensity_constant = prim.GetAttribute("inputs:intensity")
                if not intensity_constant.IsValid():
                    intensity_constant = prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float)
                intensity_constant.Set(intensity) # type: ignore
            # color
            if random_cfg.random_color:
                if self.cfg.random.global_light_cfg.random_type == "list":
                    index = random.randint(0,len(self.cfg.random.global_light_cfg.color_list)-1) # type: ignore
                    color = [self.cfg.random.global_light_cfg.color_list[index][i]  /255.0 for i in range(3)] # type: ignore
                else:
                    color_range = random_cfg.color_range # type: ignore
                    color = [
                        random.randint(color_range[0][0],color_range[1][0]) /255.0, # type: ignore
                        random.randint(color_range[0][1],color_range[1][1]) /255.0, # type: ignore
                        random.randint(color_range[0][2],color_range[1][2]) /255.0, # type: ignore
                    ]
                # 
                color_constant = prim.GetAttribute("inputs:color")
                if not color_constant.IsValid():
                    color_constant = prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Float3)
                color_constant.Set(tuple(color)) # type: ignore
        
        # local lights
        if self.cfg.random and self.cfg.random.local_lights_cfg:
            # traverse all random config
            for light_name, random_cfg in self.cfg.random.local_lights_cfg.items():
                # ignore random config of lights which are not in scene
                if light_name not in self.cfg.local_lights_cfg.keys():
                    continue
                # get prim first
                # resolve prim paths for spawning and cloning
                prim_paths = sim_utils.find_matching_prim_paths(self.cfg.local_lights_cfg[light_name].prim_path)
                for prim_path in prim_paths:
                    prim = prim_utils.get_prim_at_path(prim_path)
                    # intensity
                    if random_cfg.random_intensity:
                        if random_cfg.random_type == "list":
                            index = random.randint(0,len(random_cfg.intensity_list)-1) # type: ignore
                            intensity = random_cfg.intensity_list[index]# type: ignore
                        else:
                            intensity= random.randint(random_cfg.intensity_range[0],random_cfg.intensity_range[1]) # type: ignore
                        # 
                        # change property: intensity
                        intensity_constant = prim.GetAttribute("inputs:intensity")
                        if not intensity_constant.IsValid():
                            intensity_constant = prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float)
                        intensity_constant.Set(intensity) # type: ignore
                    # color
                    if random_cfg.random_color:
                        if random_cfg.random_type == "list":
                            index = random.randint(0,len(random_cfg.color_list)-1) # type: ignore
                            color = [random_cfg.color_list[index][i]  /255.0 for i in range(3)] # type: ignore
                        else:
                            color_range = random_cfg.color_range # type: ignore
                            color = [
                                random.randint(color_range[0][0],color_range[1][0]) /255.0, # type: ignore
                                random.randint(color_range[0][1],color_range[1][1]) /255.0, # type: ignore
                                random.randint(color_range[0][2],color_range[1][2]) /255.0, # type: ignore
                            ]
                        # 
                        color_constant = prim.GetAttribute("inputs:color")
                        if not color_constant.IsValid():
                            color_constant = prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Float3)
                        color_constant.Set(tuple(color)) # type: ignore
        
        pass

    def _apply_rigid_objects_random(self, env_ids: Sequence[int] | None = None):
        # 
        if self.cfg.random.rigid_objects_cfg is None: # type: ignore
            return
        # transform
        self._apply_rigid_objects_transform_random(env_ids)
        # material
        for rigid_name, rigid_random_cfg in self.cfg.random.rigid_objects_cfg.items(): # type: ignore
            # ignore random config of rigid objects which are not in scene and no material random config
            if rigid_name not in self.cfg.rigid_objects_cfg.keys(): 
                continue
            # visual material
            if rigid_random_cfg.visual_material:
                self._apply_visual_material_random(env_ids,self.rigid_objects[rigid_name].cfg.prim_path,rigid_random_cfg.visual_material)
            # rigid physics material
            if rigid_random_cfg.physics_material:
                self._apply_rigid_physics_material_random(env_ids,self.rigid_objects[rigid_name].cfg.prim_path,rigid_random_cfg.physics_material)
            # mass
            if rigid_random_cfg.mass:
                self._apply_mass_random(env_ids,self.rigid_objects[rigid_name].cfg.prim_path,rigid_random_cfg.mass)

    def _apply_rigid_objects_transform_random(self, env_ids: Sequence[int] | None = None):

        # traverse all rigid objects random config 
        for rigid_name, rigid_random_cfg in self.cfg.random.rigid_objects_cfg.items(): # type: ignore
            # ignore random config of rigid objects which are not in scene
            if rigid_name not in self.cfg.rigid_objects_cfg.keys():
                continue
            # get default state
            state = self.rigid_objects[rigid_name].data.default_root_state.clone()
            # position
            if rigid_random_cfg.position:
                #
                pos_random_cfg = rigid_random_cfg.position
                #
                enable = torch.tensor(pos_random_cfg.enable,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
                # compute position offset
                if pos_random_cfg.type == "list":
                    # random index of list
                    indexs = torch.arange(0,self.num_envs,1,device = self.device)
                    indexs = indexs % torch.tensor(len(pos_random_cfg.offset_list),device = self.device) # type: ignore
                    # indexs = torch.randint(0,len(pos_random_cfg.offset_list),[self.num_envs],device = self.device) # type: ignore
                    offset = torch.tensor(pos_random_cfg.offset_list,device = self.device)
                    offset = torch.index_select(offset, 0, indexs)
                    offset = torch.mul(offset, enable) 
                else:
                    offset_range = torch.tensor(pos_random_cfg.offset_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
                    offset = 2.0 * (torch.rand((self.num_envs,3),device=self.device) - 0.5)
                    offset = offset.mul(offset_range)
                    offset = torch.mul(offset, enable) 
                # 
                state[:,:3] = state[:,:3] + offset
           
            # orientation
            if rigid_random_cfg.orientation:
                #
                ori_random_cfg = rigid_random_cfg.orientation
                #
                eular_base_index = torch.randint(0,len(ori_random_cfg.eular_base),[self.num_envs],device = self.device) # type: ignore
                self.eular_base_index = eular_base_index
                eular_base = torch.tensor(ori_random_cfg.eular_base,device = self.device)

                eular_base = torch.index_select(eular_base, 0, eular_base_index)

                enable = torch.tensor(ori_random_cfg.enable,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
                # compute orientation 
                if ori_random_cfg.type == "list":
                    # random index of list
                    eular_list = torch.tensor(ori_random_cfg.eular_list,device = self.device)
                    eular_list = torch.index_select(eular_list, 0, eular_base_index)
                    # 
                    eular_num = len(ori_random_cfg.eular_list[0]) # type: ignore
                    indexs = torch.randint(0,eular_num,[self.num_envs],device = self.device) # type: ignore
                    eular = eular_list[:,0,:].squeeze(1)
                    for i in range(self.num_envs):
                        eular[i,:] = torch.index_select(eular_list[i,:,:], 0, indexs[i])
                    # eular = torch.index_select(eular_list, 1, indexs[])
                    eular = torch.mul(eular, enable)
                else:
                    eular_range = torch.tensor(ori_random_cfg.eular_range,device = self.device)
                    eular_range = torch.index_select(eular_range, 0, eular_base_index)
                    eular = 2.0 * (torch.rand((self.num_envs,3),device=self.device) - 0.5)
                    eular = eular.mul(eular_range)
                    eular = torch.mul(eular, enable) 
                #
                eular = eular_base + eular
                # print(eular_base[0,:])
                # eular to quat
                quat = quat_from_euler_xyz(eular[:,0],eular[:,1],eular[:,2])
                # eular to quat
                state[:,3:7] = quat
                # 
                if self.rigid_objects[rigid_name].cfg.enable_height_offset and ori_random_cfg.height_offset is not None:
                    # 将None替换为0
                    height_offset_temp = ori_random_cfg.height_offset.copy()
                    for i in range(len(height_offset_temp)):
                        if height_offset_temp[i] is None:
                            height_offset_temp[i] = 0
                    
                    pos_z_offset = torch.tensor(height_offset_temp,device = self.device)
                    pos_z_offset = torch.index_select(pos_z_offset, 0, eular_base_index)
                    #
                    state[:,2]+=pos_z_offset
            # add height offset if no orientation random config active
            else:
                #
                if self.rigid_objects[rigid_name].cfg.enable_height_offset and self.rigid_objects[rigid_name].cfg.height_offset:
                    state[:,2]+= self.rigid_objects[rigid_name].cfg.height_offset * torch.ones_like(state[:,2])
            # add env origins
            state[:,:3] += self.env_origins
            # state
            self.rigid_objects[rigid_name].write_root_state_to_sim(state[env_ids,:],env_ids)

    def _apply_articulated_objects_random(self, env_ids: Sequence[int] | None = None):
        # 
        if self.cfg.random.articulated_objects_cfg is None: # type: ignore
            return
        #
        for object_name, random_cfg in self.cfg.random.articulated_objects_cfg.items(): # type: ignore
            # ignore random config of rigid objects which are not in scene and no material random config
            if object_name not in self.cfg.articulated_objects_cfg.keys(): 
                continue
            # transform
            self._apply_articulated_objects_transform_random(env_ids,object_name,random_cfg.position,random_cfg.orientation)
            # visual material
            if random_cfg.visual_material:
                self._apply_visual_material_random(env_ids,self.articulated_objects[object_name].cfg.prim_path,random_cfg.visual_material)
            # rigid physics material
            if random_cfg.physics_material:
                self._apply_rigid_physics_material_random(env_ids,self.articulated_objects[object_name].cfg.prim_path,random_cfg.physics_material)
            # joint 
            if random_cfg.joint:
                self._apply_joint_random(env_ids,self.articulated_objects[object_name],random_cfg.joint)
            # mass
            if random_cfg.mass:
                self._apply_mass_random(env_ids,self.articulated_objects[object_name].cfg.prim_path,random_cfg.mass)
    
    def _apply_articulated_objects_transform_random(self, env_ids: Sequence[int] | None = None, name : str | None = None, pos_config: PositionRandomCfg | None = None, ori_config: OrientationRandomCfg | None = None):
        #
        if name is None or (pos_config is None and pos_config is None):
            return 
        # get default state
        state = self._articulated_objects[name].data.default_root_state.clone()
        #
        if pos_config:
            #
            enable = torch.tensor(pos_config.enable,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
            # compute position offset
            if pos_config.type == "list":
                # random index of list
                indexs = torch.randint(0,len(pos_config.offset_list),[self.num_envs],device = self.device) # type: ignore
                offset = torch.tensor(pos_config.offset_list,device = self.device)
                offset = torch.index_select(offset, 0, indexs)
                offset = torch.mul(offset, enable) 
            else:
                offset_range = torch.tensor(pos_config.offset_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
                offset = 2.0 * (torch.rand((self.num_envs,3),device=self.device) - 0.5)
                offset = offset.mul(offset_range)
                offset = torch.mul(offset, enable) 
            # 
            state[:,:3] = state[:,:3] + offset + self.env_origins
        #
        if ori_config:
            #
            eular_base_index = torch.randint(0,len(ori_config.eular_base),[self.num_envs],device = self.device) # type: ignore
            eular_base = torch.tensor(ori_config.eular_base,device = self.device)

            eular_base = torch.index_select(eular_base, 0, eular_base_index)

            enable = torch.tensor(ori_config.enable,device = self.device).unsqueeze(0).repeat(self.num_envs,1)
            # compute orientation 
            if ori_config.type == "list":
                # random index of list
                eular_list = torch.tensor(ori_config.eular_list,device = self.device)
                eular_list = torch.index_select(eular_list, 0, eular_base_index)
                # 
                eular_num = len(ori_config.eular_list[0]) # type: ignore
                indexs = torch.randint(0,eular_num,[self.num_envs],device = self.device) # type: ignore
                eular = eular_list[:,0,:].squeeze(1)
                for i in range(self.num_envs):
                    eular[i,:] = torch.index_select(eular_list[i,:,:], 0, indexs[i])
                # eular = torch.index_select(eular_list, 1, indexs[])
                eular = torch.mul(eular, enable)
            else:
                eular_range = torch.tensor(ori_config.eular_range,device = self.device)
                eular_range = torch.index_select(eular_range, 0, eular_base_index)
                eular = 2.0 * (torch.rand((self.num_envs,3),device=self.device) - 0.5)
                eular = eular.mul(eular_range)
                eular = torch.mul(eular, enable) 
            #
            eular = eular_base + eular
            # print(eular_base[0,:])
            # eular to quat
            quat = quat_from_euler_xyz(eular[:,0],eular[:,1],eular[:,2])
            # eular to quat
            state[:,3:7] = quat
            # TODO: does articulation object need height offset?
            # pos_z_offset = torch.tensor(ori_config.pos_z_offset,device = self.device)
            # pos_z_offset = torch.index_select(pos_z_offset, 0, eular_base_index)
            #
        #
        #
        # state[:,2]+=pos_z_offset

        # state.
        self._articulated_objects[name].write_root_state_to_sim(state[env_ids,:],env_ids)

    def _apply_mass_random(self, env_ids: Sequence[int] | None = None,prim_path: str| None = None, config: MassRandomCfg | None = None):
        #
        if not prim_path or not config or not config.enable:
            return 
        # 
        if config.prim_path:
            mass_prim_paths = [prim_path + relative_prim_path for relative_prim_path in config.prim_path]
        else:
            mass_prim_paths = [prim_path]
        #
        for mass_prim_path in mass_prim_paths: # type: ignore
            #
            asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", mass_prim_path) is None
            # resolve prim paths
            paths = sim_utils.find_matching_prim_paths(mass_prim_path)
            #
            for path in paths:
                #
                if asset_path_is_regex:
                    re_result = re.search("/env_[0-9]+/",path).span()
                    env_index = int(path[re_result[0]+1:re_result[1]-1].split("_")[-1])
                    if env_index not in env_ids: # type: ignore
                        continue
                # get prim first
                prim = prim_utils.get_prim_at_path(path+"/base_link")
                #
                mass = None
                density = None 
                #
                if config.type == "range":
                    # mass
                    if config.mass_range:
                        mass_range = config.mass_range
                        mass = mass_range[0]+ (mass_range[1]-mass_range[0]) * random.random() # type: ignore
                    # density
                    if config.density_range:
                        density_range = config.density_range
                        density = density_range[0]+ (density_range[1]-density_range[0]) * random.random() # type: ignore
                else:
                    # mass
                    if config.mass_list:
                        index = random.randint(0,len(config.mass_list)-1) # type: ignore
                        mass = config.mass_list[index]
                    # density
                    if config.density_list:
                        index = random.randint(0,len(config.density_list)-1) # type: ignore
                        density = config.density_list[index]

                # modify
                mass_property = MassPropertiesCfg(
                    mass=mass,
                    density=density,
                )
                modify_mass_properties(path,mass_property)

    def _apply_visual_material_random(self, env_ids: Sequence[int] | None = None, prim_path: str| None = None, config: VisualMaterialRandomCfg | None = None):
        # 
        if not prim_path or not config or not config.enable:
            return 
        #
        for relative_shader_path in config.shader_path:
            #
            shader_path = prim_path + relative_shader_path # type: ignore
            #
            asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", shader_path) is None
            # resolve prim paths
            paths = sim_utils.find_matching_prim_paths(shader_path)
            # 
            for path in paths:
                # 
                if asset_path_is_regex:
                    re_result = re.search("/env_[0-9]+/",path).span()
                    env_index = int(path[re_result[0]+1:re_result[1]-1].split("_")[-1])
                    if env_index not in env_ids: # type: ignore
                        continue
                # get prim first
                prim = prim_utils.get_prim_at_path(path)
                # change properties: color and texture
                if config.material_type == "color":
                    self._apply_visual_material_random_color(prim,config)
                    #
                    texture = prim.GetAttribute('inputs:diffuse_texture')
                    if texture:
                        texture.Set('')
                elif config.material_type == "texture":
                    self._apply_visual_material_random_texture(prim,config)
                else: 
                    self._apply_visual_material_random_color(prim,config)
                    #
                    self._apply_visual_material_random_texture(prim,config)
                
                # change properties: material param
                self._apply_visual_material_random_param(prim,config)
            
    def _apply_visual_material_random_color(self, prim: Usd.Prim, config: VisualMaterialRandomCfg | None = None):
        # color
        if config.random_type == "range": # type: ignore
            color_range = config.color_range # type: ignore
            color = [
                random.randint(color_range[0][0],color_range[1][0]) /255.0, # type: ignore
                random.randint(color_range[0][1],color_range[1][1]) /255.0, # type: ignore
                random.randint(color_range[0][2],color_range[1][2]) /255.0, # type: ignore
            ]
        else:
            index = random.randint(0,len(config.color_list)-1) # type: ignore
            color = config.color_list[index] # type: ignore
        # change property: diffuse_color_constant
        set_attribute_on_usd_prim(prim,"inputs:diffuse_color_constant",tuple(color),False) # type: ignore
        # change property: diffuse_tint
        set_attribute_on_usd_prim(prim,"inputs:diffuse_tint",tuple(color),False) # type: ignore

    def _apply_visual_material_random_texture(self, prim: Usd.Prim, config: VisualMaterialRandomCfg | None = None):
        #
        if len(config.texture_list)>0: # type: ignore
            # change property: texture
            index = random.randint(0,len(config.texture_list)-1) # type: ignore
            set_attribute_on_usd_prim(prim,"inputs:diffuse_texture",config.texture_list[index],False) # type: ignore

    def _apply_visual_material_random_param(self, prim: Usd.Prim, config: VisualMaterialRandomCfg | None = None):
        # change properties: reflection parameter
        if config.random_type == "range": # type: ignore
            # roughness
            roughness_range = config.roughness_range
            roughness = roughness_range[0]+ (roughness_range[1]-roughness_range[0]) * random.random()
            set_attribute_on_usd_prim(prim,"inputs:reflection_roughness_constant",roughness,False) # type: ignore
            # metalness
            metalness_range = config.metalness_range
            metalness = metalness_range[0]+ (metalness_range[1]-metalness_range[0]) * random.random()
            set_attribute_on_usd_prim(prim,"inputs:metallic_constant",metalness,False) # type: ignore
            # specular
            specular_range = config.specular_range
            specular = specular_range[0]+ (specular_range[1]-specular_range[0]) * random.random()
            set_attribute_on_usd_prim(prim,"inputs:specular_level",specular,False) # type: ignore

        else:
            # roughness
            index = random.randint(0,len(config.roughness_list)-1) # type: ignore
            set_attribute_on_usd_prim(prim,"inputs:specular_level",config.roughness_list[index],False) # type: ignore
            # metalness
            index = random.randint(0,len(config.metalness_list)-1) # type: ignore
            set_attribute_on_usd_prim(prim,"inputs:metallic_constant",config.metalness_list[index],False) # type: ignore
            # specular
            index = random.randint(0,len(config.specular_list)-1) # type: ignore
            set_attribute_on_usd_prim(prim,"inputs:specular_level",config.specular_list[index],False) # type: ignore

    def _apply_rigid_physics_material_random(self, env_ids: Sequence[int] | None = None, prim_path: str| None = None, config: RigidPhysicMaterialRandomCfg | None = None):
        # 
        if not prim_path or not config or not config.enable:
            return 
        #
        if config.material_path:
            physics_material_paths = config.material_path 
        else:
            physics_material_paths = [prim_path + "/physics_material"]
        #
        for physics_material_path in physics_material_paths:
            #
            asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", physics_material_path) is None
            # resolve prim paths for spawning and cloning
            prim_paths = sim_utils.find_matching_prim_paths(physics_material_path)
            # 
            for prim_path in prim_paths:
                # 
                if asset_path_is_regex:
                    re_result = re.search("/env_[0-9]+/",prim_path).span()
                    env_index = int(prim_path[re_result[0]+1:re_result[1]-1].split("_")[-1])
                    if env_index not in env_ids: # type: ignore
                        continue
                # get prim first
                prim = prim_utils.get_prim_at_path(prim_path)
                # range random
                if config.random_type == "range": # type: ignore
                    # static friction 
                    static_friction_range = config.static_friction_range
                    static_friction = static_friction_range[0]+ (static_friction_range[1]-static_friction_range[0]) * random.random()
                    # dynamic friction
                    dynamic_friction_range = config.dynamic_friction_range
                    dynamic_friction = dynamic_friction_range[0]+ (dynamic_friction_range[1]-dynamic_friction_range[0]) * random.random()
                    # restitution
                    restitution_range = config.restitution_range
                    restitution = restitution_range[0]+ (restitution_range[1]-restitution_range[0]) * random.random()
                    # modify
                    rigid_physics_material = RigidBodyMaterialCfg(
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=restitution
                    )
                    modify_physics_material_properties(prim_path,rigid_physics_material)
                elif config.random_type == "list": # type: ignore
                    # static friction 
                    index = random.randint(0,len(config.static_friction_list)-1) # type: ignore
                    static_friction = config.static_friction_list[index]
                    # dynamic friction
                    index = random.randint(0,len(config.dynamic_friction_list)-1) # type: ignore
                    dynamic_friction = config.dynamic_friction_list[index]
                    # restitution
                    index = random.randint(0,len(config.restitution_list)-1) # type: ignore
                    restitution = config.restitution_list[index]
                    # modify
                    rigid_physics_material = RigidBodyMaterialCfg(
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=restitution
                    )
                    modify_physics_material_properties(prim_path,rigid_physics_material)
    
    def _apply_joint_random(self, env_ids: Sequence[int] | None = None, target : RobotBase | ArticulatedObject | None= None, config: JointRandomCfg | None = None):
        # 
        if target is None or config is None or not config.enable:
            return 
        #
        joint_num = len(config.joint_names)
        joint_names = config.joint_names
        joint_indexs = target.find_joints(joint_names,preserve_order=True)[0]
        #
        if config.type == "range":
            # position
            if config.position_range:
                position_range = torch.tensor(config.position_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1,1)
                position_norm = torch.rand((self.num_envs,joint_num),device=self.device)
                position = position_norm * (position_range[:,:,1] - position_range[:,:,0]) + position_range[:,:,0]
                target.write_joint_position_to_sim(position[env_ids,:],joint_indexs,env_ids)
                # robot should set target 
                if isinstance(target,RobotBase):
                    target.data.joint_pos_target[env_ids,:][:,joint_indexs] = position[env_ids,:]
                    #
                    target.write_data_to_sim()
            # damping
            if config.damping_range:
                
                damping_range = torch.tensor(config.damping_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1,1)
                damping_norm = torch.rand((self.num_envs,joint_num),device=self.device)
                damping = damping_norm * (damping_range[:,:,1] - damping_range[:,:,0]) + damping_range[:,:,0]
                target.write_joint_damping_to_sim(damping[env_ids,:],joint_indexs,env_ids)
            # stiffness
            if config.stiffness_range:
                stiffness_range = torch.tensor(config.stiffness_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1,1)
                stiffness_norm = torch.rand((self.num_envs,joint_num),device=self.device)
                stiffness = stiffness_norm * (stiffness_range[:,:,1] - stiffness_range[:,:,0]) + stiffness_range[:,:,0]
                target.write_joint_stiffness_to_sim(stiffness[env_ids,:],joint_indexs,env_ids)
            # friction
            if config.friction_range:
                friction_range = torch.tensor(config.friction_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1,1)
                friction_norm = torch.rand((self.num_envs,joint_num),device=self.device)
                friction = friction_norm * (friction_range[:,:,1] - friction_range[:,:,0]) + friction_range[:,:,0]
                target.write_joint_friction_to_sim(friction[env_ids,:],joint_indexs,env_ids)
            # armature
            if config.armature_range:
                armature_range = torch.tensor(config.armature_range,device = self.device).unsqueeze(0).repeat(self.num_envs,1,1)
                armature_norm = torch.rand((self.num_envs,joint_num),device=self.device)
                armature = armature_norm * (armature_range[:,:,1] - armature_range[:,:,0]) + armature_range[:,:,0]
                target.write_joint_armature_to_sim(armature[env_ids,:],joint_indexs,env_ids)
        else:
            # position
            if config.position_list:
                #
                position_list = torch.tensor(config.position_list,device = self.device)
                #
                position = torch.zeros((self.num_envs,joint_num),device=self.device)
                for i in range(joint_num):
                    # random index of list
                    indexs = torch.randint(0,len(config.position_list[0]),[self.num_envs],device = self.device) # type: ignore
                    position[:,i] = torch.index_select(position_list[i], 0, indexs)
                target.write_joint_position_to_sim(position[env_ids,:],joint_indexs,env_ids)
            # damping
            if config.damping_list:
                #
                damping_list = torch.tensor(config.damping_list,device = self.device)
                #
                damping = torch.zeros((self.num_envs,joint_num),device=self.device)
                for i in range(joint_num):
                    # random index of list
                    indexs = torch.randint(0,len(config.damping_list[0]),[self.num_envs],device = self.device) # type: ignore
                    damping[:,i] = torch.index_select(damping_list[i], 0, indexs)
                target.write_joint_damping_to_sim(damping[env_ids,:],joint_indexs,env_ids)
            # stiffness
            if config.stiffness_list:
                #
                stiffness_list = torch.tensor(config.stiffness_list,device = self.device)
                #
                stiffness = torch.zeros((self.num_envs,joint_num),device=self.device)
                for i in range(joint_num):
                    # random index of list
                    indexs = torch.randint(0,len(config.stiffness_list[0]),[self.num_envs],device = self.device) # type: ignore
                    stiffness[:,i] = torch.index_select(stiffness_list[i], 0, indexs)
                target.write_joint_stiffness_to_sim(stiffness[env_ids,:],joint_indexs,env_ids)
            # friction
            if config.friction_list:
                #
                friction_list = torch.tensor(config.friction_list,device = self.device)
                #
                friction = torch.zeros((self.num_envs,joint_num),device=self.device)
                for i in range(joint_num):
                    # random index of list
                    indexs = torch.randint(0,len(config.friction_list[0]),[self.num_envs],device = self.device) # type: ignore
                    friction[:,i] = torch.index_select(friction_list[i], 0, indexs)
                target.write_joint_friction_to_sim(friction[env_ids,:],joint_indexs,env_ids)
            # armature
            if config.armature_list:
                #
                armature_list = torch.tensor(config.armature_list,device = self.device)
                #
                armature = torch.zeros((self.num_envs,joint_num),device=self.device)
                for i in range(joint_num):
                    # random index of list
                    indexs = torch.randint(0,len(config.armature_list[0]),[self.num_envs],device = self.device) # type: ignore
                    armature[:,i] = torch.index_select(armature_list[i], 0, indexs)
                target.write_joint_armature_to_sim(armature[env_ids,:],joint_indexs,env_ids)

    @property
    def robots(self) -> dict[str, RobotBase]:
        """A dictionary of robots in the scene."""
        return self._robots
    
    @property
    def cameras(self) -> dict[str, Camera]:
        """A dictionary of robots in the scene."""
        return self._cameras
    
    @property
    def tiled_cameras(self) -> dict[str, TiledCamera]:
        """A dictionary of robots in the scene."""
        return self._tiled_cameras
    
    @property
    def lights(self) -> dict[str, Light]:
        """A dictionary of light in the scene."""
        return self._lights
    
    @property
    def visualizer(self) -> VisualizationMarkers:
        """A dictionary of robots in the scene."""
        return self._visualizer
    
    @property
    def articulated_objects(self) -> dict[str, ArticulatedObject]:
        """A dictionary of robots in the scene."""
        return self._articulated_objects
    
    @property
    def particle_cloths(self)-> dict[str, ParticleCloth]:
        """A dictionary of particle cloth in the scene."""
        return self._particle_cloths
    