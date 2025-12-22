import numpy as np
import torch as th
import random
import os
import yaml
import omni.usd
import random 
# from omni.isaac.core.utils.prims import set_prim_attribute
from pxr import UsdLux, Usd ,UsdGeom ,Gf
import omni.kit.commands
import omni.timeline
import omni.kit.commands
import omni.usd
from omni.isaac.dynamic_control import _dynamic_control
# import omnigibson as og
# import omnigibson.lazy as lazy
# from omnigibson.macros import gm
# from omnigibson.objects import REGISTERED_OBJECTS
# from omnigibson.utils.python_utils import create_class_from_registry_and_config

SIM_FORCE_LIGHT_INTENSITY = 15000 


class EnvAugmentor:
    """环境增强器：用于随机改变环境状态"""
    
    def __init__(self, texture_dir, default_texture_path, obj_config_path, position_offset, default_pos, light_color, light_intensity, default_obj_config_path, unit_offset=0.05):
        global stage 
        self.texture_dir = texture_dir
        self.default_texture_path = default_texture_path
        self.obj_config_path = obj_config_path
        self.position_offset = position_offset
        self.default_pos = default_pos
        self.light_color = light_color
        self.light_intensity = light_intensity
        self.default_obj_config_path = default_obj_config_path
        self.unit_offset = unit_offset
        self.log = dict()
        stage = omni.usd.get_context().get_stage()
        
    def change_obj_pose(self, env):
        is_exists = self.check_prim_exists("/World/B31V2")
        if not is_exists :
            target = env.scene.rigid_objects["B31V2"] # isaac lab中 定义的 目标抓取物 
            if target is None:
                raise ValueError("Could not find target in environment")
            
            target_init_pos = target.cfg.init_state.pos
            target_init_quat = th.tensor(target.cfg.init_state.rot,device=env.sim.device)
            target_init_vel = th.zeros(6,device=env.sim.device)
            
            index = random.randint(1, len(self.position_offset)-1)
            new_pos=[self.position_offset[index][0]+target_init_pos[0], self.position_offset[index][1]+target_init_pos[1], target_init_pos[2]]
            new_target_pose = th.tensor(new_pos ,device=env.sim.device)
            target.write_root_state_to_sim(th.cat((new_target_pose,target_init_quat,target_init_vel),0).unsqueeze(0))  
            # ball = env.scene.object_registry("name", "ball0")
            # ball.set_position(new_pos)
            self.log['change_obj_pose'] = new_pos
    def adjust_light_intensity(self, env):
        """随机调整光照强度"""
        scale = random.choice(self.light_intensity)
        light_prim = stage.GetPrimAtPath("/World/Light")
        self._recursive_light_update(light_prim, 'intensity', scale)
        self.log['adjust_light_intensity'] = scale
    
    def adjust_light_color(self, env):
        """随机改变光照颜色"""
        color = random.choice(self.light_color)
        light_prim = stage.GetPrimAtPath("/World/Room/DomeLight")
        self._recursive_light_update(light_prim ,"color",color)
        self.log['adjust_light_color'] = color
        
    def _recursive_light_update(self, light_prim, attr_type, color):
        if light_prim:
            light = UsdLux.DomeLight(light_prim)
            if attr_type == 'intensity':
                light.GetIntensityAttr().Set(SIM_FORCE_LIGHT_INTENSITY * color)
                # light.get
            elif attr_type == 'color':
                # prim.GetAttribute("inputs:color").Set(lazy.pxr.Gf.Vec3f(color))
                light.GetColorAttr().Set(Gf.Vec3f(color))  # 设置为绿色
    def replace_object(self ,env):
        """随机替换物体"""
        # timeline = omni.timeline.get_timeline_interface()
        # timeline.pause()
        target_prim_path = "/World/B31V2"
        omni.kit.commands.execute("DeletePrims", paths=[target_prim_path])
        with open(self.obj_config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        obj_config = random.choice(cfg["objects"])
        asset_prim_path = obj_config['asset_path']
        prim_path = obj_config['prim_path']
        is_exists = self.check_prim_exists(prim_path)
        if not is_exists :
            omni.kit.commands.execute(
                "CreateReference",
                path_to=prim_path,  # 放置资产的路径
                asset_path=asset_prim_path,   # USD 资产路径
                usd_context=omni.usd.get_context()
            )       
            # new_position = [-0.0,-0.0,0.9700463624969125]
            # new_orientation = Gf.Quatf(1.0, 0.0, 0.0, 0.0)  
            new_position = obj_config['pos']
            new_orientation  =obj_config['rot']
            ta_quatf = Gf.Quatd()
            ta_quatf.SetReal(new_orientation[0])
            ta_quatf.SetImaginary(tuple(new_orientation[1:]))
            scale  =obj_config['scale']                
            prim = stage.GetPrimAtPath(prim_path)
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(new_position))
            prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(scale))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(ta_quatf))
            
            self.log["replace_object"] =  f'''{obj_config['name']}_{obj_config['asset_path']}'''
        
        # timeline.play()

    def add_plate(self, env):
         with open(self.default_obj_config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            obj_config = cfg["objects"][1]
        
        # 创建并添加新物体
            obj = create_class_from_registry_and_config(
                cls_name=obj_config["type"],
                cls_registry=REGISTERED_OBJECTS,
                cfg=obj_config,
                cls_type_descriptor="object",
            )
            env.scene.add_object(obj)
            position = obj_config.pop("position", None)
            orientation = obj_config.pop("orientation", None)
            obj.set_position_orientation(position=position, orientation=orientation, frame="scene")
            self.log['add_plate'] = f'''{obj_config['category']}_{obj_config['model']}'''
    
    def change_texture(self, env):
        """随机改变纹理"""
        texture_path = os.path.join(
            self.texture_dir,
            random.choice(os.listdir(self.texture_dir))
        )    
        table_prim = stage.GetPrimAtPath("/World/table")  # 后面可以更换自己的资产  
        self._get_texture_prim(table_prim).GetAttribute("inputs:file").Set(texture_path)      
        self.log['change_texture'] = texture_path
    #xyh
    def _get_texture_prim(self, table_prim):
        """获取纹理采样器"""
        return (table_prim
                .GetChild("Looks")
                .GetChild("boardMat")
                .GetChild("diffuseTexture"))
        
    def check_prim_exists(self ,prim_path: str) -> bool:
        """检查指定路径的 Prim 是否存在"""
        if not stage:
            print("Error: Unable to get USD stage.")
            return False

        prim = stage.GetPrimAtPath(prim_path)
        return prim.IsValid()

    
    # Albert Mao 
    def apply_random_aug(self ,env):
        """随机应用一种增强"""
        aug_funcs = [
            self.replace_object,
            # self.apply_object_pos,
            # self.change_obj_pose,
            # self.adjust_light_intensity,
            # self.adjust_light_color,
            # self.change_texture,
        ]
        random.choice(aug_funcs)(env)
    def apply_object_pos(self, env):
        """随机应用两种增强"""
        aug_funcs = [
            self.replace_object,
            self.change_obj_pose,
        ]
        selected_funcs = random.sample(list(enumerate(aug_funcs)), 2)
        selected_funcs.sort(key=lambda x: x[0])
        for _, func in selected_funcs:
            func(env)
    def apply_two_aug(self, env):
        """随机应用两种增强"""
        aug_funcs = [
            # self.replace_object,
            self.change_obj_pose,
            self.adjust_light_intensity,
            self.adjust_light_color,
            self.change_texture
        ]
        selected_funcs = random.sample(list(enumerate(aug_funcs)), 2)
        selected_funcs.sort(key=lambda x: x[0])
        for _, func in selected_funcs:
            func(env)
    def apply_three_aug(self, env):
        """随机应用三种增强"""
        aug_funcs = [
            # self.replace_object,
            self.change_obj_pose,
            self.adjust_light_intensity,
            self.adjust_light_color,
            self.change_texture
        ]
        selected_funcs = random.sample(list(enumerate(aug_funcs)), 3)
        selected_funcs.sort(key=lambda x: x[0])
        for _, func in selected_funcs:
            func(env)
    def apply_random_n_aug(self, env, min_augs=1, max_augs=3):
        """随机应用1到3种增强
        
        Args:
            env: 要应用增强的环境
            min_augs: 最少应用的增强数量，默认为1
            max_augs: 最多应用的增强数量，默认为3
        """
        aug_funcs = [
            self.replace_object,
            self.change_obj_pose,
            self.adjust_light_intensity,
            self.adjust_light_color,
            self.change_texture
        ]
        
        # 随机决定应用几种增强
        num_augs = random.randint(min_augs, max_augs)
        
        # 随机选择指定数量的增强函数
        selected_funcs = random.sample(list(enumerate(aug_funcs)), num_augs)
        
        # 按原始顺序排序并应用
        selected_funcs.sort(key=lambda x: x[0])
        for _, func in selected_funcs:
            func(env)
    def restore2defaultstate(self, env):
        # 更换桌子的纹理
        table_prim = stage.GetPrimAtPath("/World/table")  # 后面可以更换自己的材质   
        self._get_texture_prim(table_prim).GetAttribute("inputs:file").Set(self.default_texture_path)
        
        # 恢复灯光 颜色
        light_prim = stage.GetPrimAtPath("/World/Room/DomeLight")
        self._recursive_light_update(light_prim, 'intensity', 1.0)
        self._recursive_light_update(light_prim, 'color', [1.0, 1.0, 1.0])
        # 恢复target的初始位置 
        is_exists = self.check_prim_exists("/World/B31V2")
        if not is_exists :
            target = env.scene.rigid_objects["B31V2"] # isaac lab中 定义的 目标抓取物 
            target_init_pos = th.tensor(target.cfg.init_state.pos,device=env.sim.device)
            target_init_quat = th.tensor(target.cfg.init_state.rot,device=env.sim.device)
            target_init_vel = th.zeros(6,device=env.sim.device)
            target.write_root_state_to_sim(th.cat((target_init_pos,target_init_quat,target_init_vel),0).unsqueeze(0))
        self.log = dict()
