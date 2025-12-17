# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import importlib

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.controllers import DifferentialIKControllerCfg,OperationalSpaceControllerCfg
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg

from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
    ArticulationRootPropertiesCfg,
    FixedTendonPropertiesCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
    DeformableBodyPropertiesCfg
)

from isaaclab.sim.spawners import (
    SpawnerCfg,
    DeformableObjectSpawnerCfg, 
    RigidObjectSpawnerCfg, 
    FileCfg,
    UsdFileCfg,
    UrdfFileCfg,
    GroundPlaneCfg,
    LightCfg,
    DiskLightCfg,
    DistantLightCfg,
    DomeLightCfg,
    CylinderLightCfg,
    SphereLightCfg,
    VisualMaterialCfg, 
    PreviewSurfaceCfg, 
    MdlFileCfg,
    GlassMdlCfg,
    PhysicsMaterialCfg, 
    RigidBodyMaterialCfg, 
    DeformableBodyMaterialCfg,
    PinholeCameraCfg, 
    FisheyeCameraCfg,
    MultiAssetSpawnerCfg, 
    MultiUsdFileCfg, 
)

from isaaclab.assets import (
    AssetBaseCfg,
    ArticulationCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
    DeformableObjectCfg
)

from isaaclab.sensors import (
    SensorBaseCfg,
    CameraCfg,
    TiledCameraCfg,
    ContactSensorCfg

)

from isaaclab.actuators.actuator_cfg import (
    ActuatorBaseCfg,
    ImplicitActuatorCfg,
    IdealPDActuatorCfg,
    DCMotorCfg,
    ActuatorNetLSTMCfg,
    ActuatorNetMLPCfg,
    DelayedPDActuatorCfg,
    RemotizedPDActuatorCfg
)

""" Psilab Modules  """ 
from psilab.scene.sence_cfg import SceneCfg
from psilab.random.random_cfg import RandomCfg
from psilab.assets.robot.robot_base_cfg import RobotBaseCfg
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg
from psilab.random.rigid_random_cfg import RigidRandomCfg
from psilab.random.light_random_cfg import LightRandomCfg


# ************* psi lab config ***************

def scene_cfg(data:dict)->SceneCfg:

    cfg = SceneCfg()

    # attributes from SpawnerCfg

    for attr,value in vars(interactive_scene_cfg(data)).items():
        setattr(cfg,attr,value)
    #
    # attributes for LightCfg
    for key, value in data.items():
        #
        if value is None:
            setattr(cfg,key,None)
            continue
        elif value is {}:
            setattr(cfg,key,{})
            continue
        #
        if key == "global_light_cfg":
            setattr(cfg,key,asset_base_cfg(value))
        elif key == "local_lights_cfg":
            local_lights_cfg = {}
            for local_light_name,local_light_cfg in value.items():
                local_lights_cfg[local_light_name] = asset_base_cfg(local_light_cfg)
            setattr(cfg,key,local_lights_cfg)
        elif key == "robots_cfg":
            pass
            robots_cfg = {}
            for robot_name,robot_cfg in value.items():
                robots_cfg[robot_name] = robot_base_cfg(robot_cfg)
            setattr(cfg,key,robots_cfg)
        elif key == "static_objects_cfg":
            static_objects_cfg = {}
            for static_name,static_cfg in value.items():
                static_objects_cfg[static_name] = asset_base_cfg(static_cfg)
            setattr(cfg,key,static_objects_cfg)
        elif key == "rigid_objects_cfg":
            rigid_objects_cfg = {}
            for rigid_name,rigid_cfg in value.items():
                rigid_objects_cfg[rigid_name] = rigid_object_cfg(rigid_cfg)
            setattr(cfg,key,rigid_objects_cfg)
        elif key == "deformable_objects_cfg":
            deformable_objects_cfg = {}
            for deformable_name,deformable_cfg in value.items():
                deformable_objects_cfg[deformable_name] = deformable_object_cfg(deformable_cfg)
            setattr(cfg,key,deformable_objects_cfg)
        elif key == "cameras_cfg":
            cameras_cfg = {}
            for camera_name,camera in value.items():
                cameras_cfg[camera_name] = camera_cfg(camera)
            setattr(cfg,key,cameras_cfg)
        elif key == "tiled_cameras_cfg":
            tiled_cameras_cfg = {}
            for camera_name,camera in value.items():
                tiled_cameras_cfg[camera_name] = tiled_camera_cfg(camera)
            setattr(cfg,key,tiled_cameras_cfg)
        elif key == "contact_sensors_cfg":
            contact_sensors_cfg = {}
            for sensor_name,sensor_cfg in value.items():
                contact_sensors_cfg[sensor_name] = contact_sensor_cfg(sensor_cfg)
            setattr(cfg,key,contact_sensors_cfg)            
        elif key == "marker_cfg":
            setattr(cfg,key,visualization_markers_cfg(value))  
        elif key == "random":
            setattr(cfg,key,random_cfg(value))  

    return cfg

def robot_base_cfg(data:dict)->RobotBaseCfg:
    cfg = RobotBaseCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(articulation_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "diff_ik_controllers":
            ik_controllers = {}
            for ik_name,ik_cfg in value.items():
                ik_controllers[ik_name] = diff_ik_controller_cfg(ik_cfg)
            setattr(cfg,key,ik_controllers)
        elif key == "eef_links":
            eef_links = {}
            for eef_link_name,eef_link in value.items():
                eef_links[eef_link_name] = eef_link
            setattr(cfg,key,eef_links)
        elif key == "cameras":
            cameras = {}
            for name,camera in value.items():
                cameras[name] = camera_cfg(camera)
                # class_type = camera["class_type"].split(":")[1]
                # if class_type=="TiledCamera":
                #     cameras[name] = tiled_camera_cfg(camera)
                # else:
                    # cameras[name] = camera_cfg(camera)

            setattr(cfg,key,cameras)
        elif key == "tiled_cameras":
            tiled_cameras = {}
            for name,camera in value.items():
                tiled_cameras[name] = tiled_camera_cfg(camera)
                # class_type = camera["class_type"].split(":")[1]
                # if class_type=="TiledCamera":
                    # tiled_cameras[name] = tiled_camera_cfg(camera)
                # else:
                #     tiled_cameras[name] = camera_cfg(camera)

            setattr(cfg,key,tiled_cameras)
    
    return cfg

def diff_ik_controller_cfg(data:dict)->DiffIKControllerCfg:

    cfg = DiffIKControllerCfg(
        command_type = "position",
        ik_method = "dls",
    )

    # attributes from SpawnerCfg
    for attr,value in vars(differential_ik_controller_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key in ["joint_name","eef_link_name"]:
            setattr(cfg,key,value)
    
    return cfg

def light_random_cfg(data:dict)->LightRandomCfg:
        
    cfg = LightRandomCfg()

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key in [
            "fake_random",
            "random_intensity",
            "random_color",
            "random_material",
            "intensity_range",
            "color_range",
            "intensity_list",
            "color_list",
        ]:
            setattr(cfg,key,visual_material_cfg(value))  

    
    return cfg

def rigid_random_cfg(data:dict)->RigidRandomCfg:
    cfg = RigidRandomCfg()

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key in [
            "random_type",
            "random_position",
            "random_orientation",
            "random_material",
            "position_range",
            "position_list",
            "orientation_list",
            # "materials",
        ]:
            setattr(cfg,key,value)  

    
    return cfg

#TODO: Add material random cfg

def random_cfg(data:dict)->RandomCfg:
    
    cfg = RandomCfg()
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key == "global_light_cfg":
            setattr(cfg,key,light_random_cfg(value))
        elif key == "local_lights_cfg":
            local_lights_cfg = {}
            for local_light_name,local_light_cfg in value.items():
                local_lights_cfg[local_light_name] = light_random_cfg(local_light_cfg)
            setattr(cfg,key,local_lights_cfg)
        elif key == "rigid_objects_cfg":
            rigid_objects_cfg = {}
            for rigid_object_name,rigid_object_cfg in value.items():
                rigid_objects_cfg[rigid_object_name] = rigid_random_cfg(rigid_object_cfg)
            setattr(cfg,key,rigid_objects_cfg)
        

    return cfg

# ************* isaac lab assets config ***************

"""
isaac lab actutors
"""

def actuator_base_cfg(data:dict)->ActuatorBaseCfg:

    cfg = ActuatorBaseCfg()
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "joint_names_expr":
            setattr(cfg,key,value)
        elif key in [
            "effort_limit",
            "velocity_limit",
            "effort_limit_sim",
            "velocity_limit_sim",
            "stiffness",
            "damping",
            "armature",
            "friction"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def implicit_actuator_cfg(data:dict)->ImplicitActuatorCfg:

    cfg = ImplicitActuatorCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(actuator_base_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        
    return cfg

def ideal_pda_actuator_cfg(data:dict)->IdealPDActuatorCfg:

    cfg = IdealPDActuatorCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(actuator_base_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        
    return cfg

def dc_motor_cfg(data:dict)->DCMotorCfg:

    cfg = DCMotorCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(ideal_pda_actuator_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "saturation_effort"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def actuator_net_lstm_cfg(data:dict)->ActuatorNetLSTMCfg:

    cfg = ActuatorNetLSTMCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(dc_motor_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "network_file"
            ]:
            setattr(cfg,key,value)
    #
    setattr(cfg,"stiffness",None)
    setattr(cfg,"damping",None)

    return cfg

def actuator_net_mlp_cfg(data:dict)->ActuatorNetMLPCfg:

    cfg = ActuatorNetMLPCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(dc_motor_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "network_file",
            "pos_scale",
            "vel_scale",
            "torque_scale",
            "input_order",
            "input_idx"
            ]:
            setattr(cfg,key,value)
    #
    setattr(cfg,"stiffness",None)
    setattr(cfg,"damping",None)   
    return cfg

def delayed_pd_actuator_cfg(data:dict)->DelayedPDActuatorCfg:

    cfg = DelayedPDActuatorCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(ideal_pda_actuator_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "min_delay",
            "max_delay"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def remotized_pd_actuator_cfg(data:dict)->RemotizedPDActuatorCfg:

    cfg = RemotizedPDActuatorCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(delayed_pd_actuator_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "joint_parameter_lookup":
            setattr(cfg,key,value)
        
    return cfg


"""
isaac lab assets
"""

def asset_base_cfg(data:dict)->AssetBaseCfg:

    cfg = AssetBaseCfg()

    def initial_state(data:dict)->AssetBaseCfg.InitialStateCfg:
        cfg = AssetBaseCfg.InitialStateCfg()
        for key, value in data.items():
            if value is None:
                continue
            if key in ["pos","rot"]:
                setattr(cfg,key,tuple(value))
        return cfg
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "spawn":
            # get spawner cfg accoding to spawner func
            spawn_func = value["func"].split(":")[-1]
            if spawn_func in spawner_func_cfg_map.keys():
                if spawner_func_cfg_map[spawn_func] is not None:
                    setattr(cfg,key,spawner_func_cfg_map[spawn_func](value))
                else:
                    # get spawner cfg accoding to prim type
                    if "prim_type" in value.keys() and value["prim_type"] in prim_type_spawner_cfg_map.keys():
                        setattr(cfg,key,prim_type_spawner_cfg_map[value["prim_type"]](value))
                    else:
                        setattr(cfg,key,spawner_cfg(value))
            else:
                setattr(cfg,key,spawner_cfg(value))
                
        elif key == "init_state":
            setattr(cfg,key,initial_state(value))
        elif key in [
            "prim_path",
            "collision_group",
            "debug_vis"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def articulation_cfg(data:dict)->ArticulationCfg:
    
    cfg = ArticulationCfg()

    def initial_state(data:dict)->ArticulationCfg.InitialStateCfg:

        cfg = ArticulationCfg.InitialStateCfg()
        # attribute from AssetBaseCfg.InitialStateCfg
        # for attr,value in vars(asset_base_cfg.initial_state(data)).items():
        #     setattr(cfg,attr,value)
        # #
        for key, value in data.items():
            if value is None:
                continue
            if key in ["pos","rot","lin_vel","ang_vel"]:
                setattr(cfg,key,tuple(value))
            elif key in ["joint_pos","joint_vel"]:
                setattr(cfg,key,value)
        return cfg
    
    # attributes from SpawnerCfg
    for attr,value in vars(asset_base_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "init_state":
            setattr(cfg,key,initial_state(value))
        elif key == "actuators":
            actuators = {}
            for actuator_name,actuator_cfg in value.items():
                actuator_class_type = actuator_cfg["class_type"].split(":")[-1]
                if actuator_class_type in actuator_class_type_map.keys():
                    actuators[actuator_name] = actuator_class_type_map[actuator_class_type](actuator_cfg)
                else:
                    actuators[actuator_name] = actuator_base_cfg(actuator_cfg)
            setattr(cfg,key,actuators)
        elif key in ["soft_joint_pos_limit_factor"]:
            setattr(cfg,key,value)
        
    return cfg

def rigid_object_cfg(data:dict)->RigidObjectCfg:
    
    cfg = RigidObjectCfg()

    def initial_state(data:dict, )->RigidObjectCfg.InitialStateCfg:

        cfg = RigidObjectCfg.InitialStateCfg()
        # attribute from AssetBaseCfg.InitialStateCfg
        # for attr,value in vars(asset_base_cfg.initial_state(data)).items():
        #     setattr(cfg,attr,value)
        #
        for key, value in data.items():
            if value is None:
                continue
            if key in ["pos","rot","lin_vel","ang_vel"]:
                setattr(cfg,key,tuple(value))
        return cfg
    
    # attributes from SpawnerCfg
    for attr,value in vars(asset_base_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "spawn":
            # get spawner cfg accoding to prim type
            setattr(cfg,key,usd_file_cfg(value))
        elif key == "init_state":
            setattr(cfg,key,initial_state(value))
        
    return cfg

def rigid_object_collection_cfg(data:dict)->RigidObjectCollectionCfg:
    
    cfg = RigidObjectCollectionCfg()

 
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "rigid_objects":
            rigid_objects = {}
            for rigid_object_name,rigid_object_cfg in value.items():
                rigid_objects[rigid_object_name] = rigid_object_cfg(rigid_object_cfg)
            setattr(cfg,key,rigid_objects)
        
    return cfg

def deformable_object_cfg(data:dict)->DeformableObjectCfg:
    
    cfg = DeformableObjectCfg()
    
    # attributes from SpawnerCfg
    for attr,value in vars(asset_base_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "visualizer_cfg":
            setattr(cfg,key,visualization_markers_cfg(value))
        
    return cfg

"""
isaac lab controllers
"""

def differential_ik_controller_cfg(data:dict)->DifferentialIKControllerCfg:

    cfg = DifferentialIKControllerCfg(
        command_type = "position",
        ik_method = "dls",
    )

    def initial_state(data:dict)->AssetBaseCfg.InitialStateCfg:
        cfg = AssetBaseCfg.InitialStateCfg()
        for key, value in data.items():
            if value is None:
                continue
            if key in ["pos","rot"]:
                setattr(cfg,key,tuple(value))
        return cfg
    
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in ["ik_params"]:
            setattr(cfg,key,value)
        elif key in [   
            "command_type",
            "use_relative_mode",
            "ik_method"
            ]:
            setattr(cfg,key,value)
        
    return cfg

"""
isaac lab scene
"""
def interactive_scene_cfg(data:dict)->InteractiveSceneCfg:

    cfg = InteractiveSceneCfg()

   
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key in [
                "num_envs",
                "env_spacing",
                "lazy_sensor_update",
                "replicate_physics",
                "filter_collisions"
                ]:
                setattr(cfg,key,value)
        
    return cfg


"""
isaac lab visualization markers
"""

def visualization_markers_cfg(data:dict)->VisualizationMarkersCfg:
    
    cfg = VisualizationMarkersCfg()

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "markers":
            markers = {}
            for marker_name,marker_cfg in value.items():
                markers[marker_name] = usd_file_cfg(marker_cfg)
            setattr(cfg,key,markers)
        if key in ["prim_path"]:
            setattr(cfg,key,value)

    return cfg


"""
isaac lab sensors
"""
def sensor_base_cfg(data:dict)->SensorBaseCfg:

    cfg = SensorBaseCfg()

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "prim_path",
            "update_period",
            "history_length",
            "debug_vis"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def camera_cfg(data:dict)->CameraCfg:

    cfg = CameraCfg()

    def offset_cfg(data:dict)->CameraCfg.OffsetCfg:
        cfg = CameraCfg.OffsetCfg()
        for key, value in data.items():
            if value is None:
                continue
            if key in ["convention"]:
                setattr(cfg,key,value)
            elif key in ["pos","rot"]:
                setattr(cfg,key,tuple(value))
        return cfg
    
    # attributes from SensorBaseCfg
    for attr,value in vars(sensor_base_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name)) 
        elif key == "offset": 
            setattr(cfg,key,offset_cfg(value))
        elif key == "spawn":
            if value["projection_type"] == "pinhole":
                setattr(cfg,key,pinhole_camera_cfg(value))
            else:
                setattr(cfg,key,fisheye_camera_cfg(value))
        # elif key == "data_types":
        elif key in [
            "depth_clipping_behavior",
            "width",
            "height",
            "colorize_semantic_segmentation",
            "colorize_instance_id_segmentation",
            "colorize_instance_segmentation",
            "data_types",
            "semantic_filter"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def tiled_camera_cfg(data:dict)->TiledCameraCfg:

    cfg = TiledCameraCfg()

    # attributes from SensorBaseCfg
    for attr,value in vars(camera_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            setattr(cfg,key,None)
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name)) 
        elif key in [
            "return_latest_camera_pose"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def contact_sensor_cfg(data:dict)->ContactSensorCfg:

    cfg = ContactSensorCfg()

    # attributes from SensorBaseCfg
    for attr,value in vars(sensor_base_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "class_type":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name)) 
        elif key == "visualizer_cfg":
            pass
        # elif key == "data_types":
        elif key in [
            "track_pose",
            "track_air_time",
            "force_threshold",
            "filter_prim_paths_expr",
            ]:
            setattr(cfg,key,value)
        
    return cfg

# ************* isaac lab sim schemas ***************

def articulation_root_properties_cfg(data:dict) ->ArticulationRootPropertiesCfg:
    cfg = ArticulationRootPropertiesCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key in [
            "articulation_enabled",
            "enabled_self_collisions","solver_position_iteration_count",
            "solver_velocity_iteration_count",
            "sleep_threshold"
            "stabilization_threshold"
            "fix_root_link"
            ]:
            setattr(cfg,key,value)  
    return cfg

def rigidbody_properties_cfg(data:dict)->RigidBodyPropertiesCfg:
    cfg = RigidBodyPropertiesCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key in [
            "rigid_body_enabled",
            "kinematic_enabled",
            "disable_gravity",
            "linear_damping",
            "angular_damping"
            "max_linear_velocity"
            "max_angular_velocity",
            "max_depenetration_velocity"
            "max_contact_impulse"
            "enable_gyroscopic_forces"
            "retain_accelerations"
            "solver_position_iteration_count"
            "solver_velocity_iteration_count"
            "sleep_threshold"
            "stabilization_threshold"
            ]:
            setattr(cfg,key,value)  
    return cfg

def collision_properties_cfg(data:dict)->CollisionPropertiesCfg:
    cfg = CollisionPropertiesCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key in [
            "collision_enabled",
            "contact_offset",
            "rest_offset",
            "torsional_patch_radius",
            "min_torsional_patch_radius"
            ]:
            setattr(cfg,key,value)  
    return cfg

def mass_properties_cfg(data:dict)->MassPropertiesCfg:
    cfg = MassPropertiesCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key in [
            "mass",
            "density"
            ]:
            setattr(cfg,key,value)  
    return cfg

def joint_drive_properties_cfg(data:dict)->JointDrivePropertiesCfg:
    cfg = JointDrivePropertiesCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key in [
            "drive_type",
            ]:
            setattr(cfg,key,value)  
    return cfg

def fixed_tendon_properties_cfg(data:dict)->FixedTendonPropertiesCfg:
    cfg = FixedTendonPropertiesCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key in [
            "tendon_enabled",
            "stiffness",
            "damping",
            "limit_stiffness",
            "offset",
            "rest_length"
            ]:
            setattr(cfg,key,value)  
    return cfg

def deformablebody_properties_cfg(data:dict)->DeformableBodyPropertiesCfg:
    cfg = DeformableBodyPropertiesCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key in [
            "deformable_enabled",
            "kinematic_enabled",
            "self_collision",
            "self_collision_filter_distance",
            "settling_threshold",
            "sleep_damping",
            "sleep_threshold",
            "solver_position_iteration_count",
            "vertex_velocity_damping",
            "simulation_hexahedral_resolution",
            "collision_simplification",
            "collision_simplification_remeshing",
            "collision_simplification_remeshing_resolution",
            "collision_simplification_target_triangle_count",
            "collision_simplification_force_conforming",
            "contact_offset",
            "rest_offset",
            "max_depenetration_velocity"
            ]:
            setattr(cfg,key,value)  
    return cfg


# ************* isaac lab spawner config ***************

def spawner_cfg(data:dict)->SpawnerCfg:
    cfg = SpawnerCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            pass
            # setattr(cfg,key,value)  
        elif key == "semantic_tags":
            for type, id in value.items():
                cfg.semantic_tags.append((type,id))  # type: ignore
        elif key in ["visible","copy_from_source"]:
            setattr(cfg,key,value)

    return cfg

def rigid_object_spawner_cfg(data:dict)->RigidObjectSpawnerCfg:
    
    cfg = RigidObjectSpawnerCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "mass_props":
            setattr(cfg,key,mass_properties_cfg(value))  
        elif key == "rigid_props":
            setattr(cfg,key,rigidbody_properties_cfg(value)) 
        elif key == "collision_props":
            setattr(cfg,key,collision_properties_cfg(value))  
        elif key in ["activate_contact_sensors"]:
            setattr(cfg,key,value)
    return cfg

def deformable_object_spawner_cfg(data:dict)->DeformableObjectSpawnerCfg:
        
    cfg = DeformableObjectSpawnerCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "mass_props":
            setattr(cfg,key,mass_properties_cfg(value))  
        elif key == "deformable_props":
            setattr(cfg,key,deformablebody_properties_cfg(value)) 

    return cfg

"""
isaac lab from files cfg
"""
def file_cfg(data:dict)->FileCfg:
    cfg = FileCfg()

    # attributes from RigidObjectSpawnerCfg
    for attr,value in vars(rigid_object_spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes from DeformableObjectSpawnerCfg
    for attr,value in vars(deformable_object_spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes for FileCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "scale":
            setattr(cfg,key,tuple(value))  
        if key == "articulation_props":
            setattr(cfg,key,articulation_root_properties_cfg(value))  
        elif key == "fixed_tendons_props":
            setattr(cfg,key,rigidbody_properties_cfg(value)) 
        elif key == "joint_drive_props":
            setattr(cfg,key,collision_properties_cfg(value)) 
        elif key == "visual_material":
            # get different material according to func
            material_type = value["func"].split(":")[-1]
            if material_type == "spawn_preview_surface":
                setattr(cfg,key,preview_surface_cfg(value)) 
            elif material_type == "spawn_from_mdl_file":
                setattr(cfg,key,mdl_file_cfg(value)) 
            else:
                setattr(cfg,key,visual_material_cfg(value)) 

        elif key in ["visual_material_path"]:
            setattr(cfg,key,value)
    return cfg

def usd_file_cfg(data:dict)->UsdFileCfg:
    cfg = UsdFileCfg()

    # attributes from FileCfg
    for attr,value in vars(file_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for FileCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name)) 
        elif key == "variants":
            pass 
        elif key in ["usd_path"]:
            setattr(cfg,key,value)
    return cfg

def urdf_file_cfg(data:dict)->UrdfFileCfg:

    cfg = UrdfFileCfg()

    # attributes from FileCfg
    for attr,value in vars(file_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for FileCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name)) 
    return cfg

"""
isaac lab multi asset spawner config
"""

def multi_asset_spawner_cfg(data:dict)->MultiAssetSpawnerCfg:

    cfg = MultiAssetSpawnerCfg()

    # attributes from RigidObjectSpawnerCfg
    for attr,value in vars(rigid_object_spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes from DeformableObjectSpawnerCfg
    for attr,value in vars(deformable_object_spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes for FileCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name)) 
        elif key == "assets_cfg":
            assets_cfg =[]
            for asset_cfg in value:
                assets_cfg.append(spawner_cfg(asset_cfg))
            setattr(cfg,key,assets_cfg)  
        elif key in ["random_choice"]:
            setattr(cfg,key,value)
    return cfg

def multi_usd_file_cfg(data:dict)->MultiUsdFileCfg:

    cfg = MultiUsdFileCfg()

    # attributes from RigidObjectSpawnerCfg
    for attr,value in vars(usd_file_cfg(data)).items():
        setattr(cfg,attr,value)

    # attributes for FileCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name)) 
        elif key == "usd_path":
            setattr(cfg,key,value)
            # if isinstance(value,str):
            #     setattr(cfg,key,value)
            # else:
            #     assets_cfg =[]
            #     for asset_cfg in value:
            #         assets_cfg.append(spawner_cfg(asset_cfg))
            #     setattr(cfg,key,assets_cfg)  
        elif key in ["random_choice"]:
            setattr(cfg,key,value)
    return cfg

"""
isaac lab ground plane config
"""
def ground_plane_cfg(data:dict)->GroundPlaneCfg:

    cfg = GroundPlaneCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in ["color","size"]:
            setattr(cfg,key,tuple(value))
        elif key == "physics_material":
            setattr(cfg,key,rigidbody_material_cfg(value))
        elif key in ["usd_path"]:
            setattr(cfg,key,value)
        
    return cfg

"""
isaac lab sensor config
"""
def pinhole_camera_cfg(data:dict)->PinholeCameraCfg:

    cfg = PinholeCameraCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "clipping_range":
            setattr(cfg,key,tuple(value))
        elif key in [
            "projection_type",
            "focal_length",
            "focus_distance",
            "f_stop",
            "horizontal_aperture",
            "vertical_aperture",
            "horizontal_aperture_offset",
            "vertical_aperture_offset",
            "lock_camera"
            ]:
            setattr(cfg,key,value)
        
    return cfg

def fisheye_camera_cfg(data:dict)->FisheyeCameraCfg:

    cfg = FisheyeCameraCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(pinhole_camera_cfg(data)).items():
        setattr(cfg,attr,value)
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "projection_type",
            "fisheye_nominal_width",
            "fisheye_nominal_height",
            "fisheye_optical_centre_x",
            "fisheye_optical_centre_y",
            "fisheye_max_fov",
            "fisheye_polynomial_a",
            "fisheye_polynomial_b",
            "fisheye_polynomial_c",
            "fisheye_polynomial_d",
            "fisheye_polynomial_e",
            "fisheye_polynomial_f"
            ]:
            setattr(cfg,key,value)
        
    return cfg

"""
isaac lab material config
"""
def visual_material_cfg(data:dict)->VisualMaterialCfg:
    cfg = VisualMaterialCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            pass
            # setattr(cfg,key,value)  
    return cfg

def preview_surface_cfg(data:dict)->PreviewSurfaceCfg:

    cfg = PreviewSurfaceCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(visual_material_cfg(data)).items():
        setattr(cfg,attr,value)
        pass

    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in ["diffuse_color", "emissive_color"]:
            setattr(cfg,key,tuple(value))
        elif key in ["roughness","metallic","opacity"]:
            setattr(cfg,key,value)
        
    return cfg

def mdl_file_cfg(data:dict)->MdlFileCfg:

    cfg = MdlFileCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(visual_material_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "texture_scale":
            setattr(cfg,key,tuple(value))
        elif key in ["mdl_path","project_uvw","albedo_brightness"]:
            setattr(cfg,key,value)
        
        
            

    return cfg

def glass_mdl_cfg(data:dict)->GlassMdlCfg:

    cfg = GlassMdlCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(visual_material_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "glass_color":
            setattr(cfg,key,tuple(value))
        elif key in ["mdl_path","frosting_roughness","thin_walled","glass_ior"]:
            setattr(cfg,key,value)
        
        
            

    return cfg

"""
isaac lab physics material config
"""
def physics_material_cfg(data:dict)->PhysicsMaterialCfg:
    cfg = PhysicsMaterialCfg()
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            pass
    return cfg

def rigidbody_material_cfg(data:dict)->RigidBodyMaterialCfg:

    cfg = RigidBodyMaterialCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(physics_material_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "static_friction",
            "dynamic_friction",
            "restitution",
            "improve_patch_friction",
            "friction_combine_mode",
            "restitution_combine_mode",
            "compliant_contact_stiffness",
            "compliant_contact_damping",
            ]:
            setattr(cfg,key,value)
        
    return cfg

def deformablebody_material_cfg(data:dict)->DeformableBodyMaterialCfg:

    cfg = DeformableBodyMaterialCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(physics_material_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key in [
            "density",
            "dynamic_friction",
            "youngs_modulus",
            "poissons_ratio",
            "elasticity_damping",
            "damping_scale"
            ]:
            setattr(cfg,key,value)
        
    return cfg

"""
isaac lab light config
"""
def light_cfg(data:dict)->LightCfg:

    cfg = LightCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(spawner_cfg(data)).items():
        setattr(cfg,attr,value)
        pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key == "func":
            module_name = value.split(":")[0]
            func_name = value.split(":")[1]
            setattr(cfg,key,getattr(importlib.import_module(module_name), func_name))  
        elif key == "color":
            setattr(cfg,key,tuple(value))
        elif key in ["prim_type","enable_color_temperature","color_temperature","normalize","exposure","intensity"]:
            setattr(cfg,key,value)
        
    return cfg

def disk_light_cfg(data:dict)->DiskLightCfg:
    cfg = DiskLightCfg()

    # attributes from SpawnerCfg
    for attr,value in vars(light_cfg(data)).items():
        setattr(cfg,attr,value)
        # pass
    # attributes for LightCfg
    for key, value in data.items():
        if key in ["radius","prim_type"] and value is not None:
            setattr(cfg,key,value)
    return cfg

def distant_light_cfg(data:dict)->DistantLightCfg:

    cfg = DistantLightCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(light_cfg(data)).items():
        setattr(cfg,attr,value)
        # pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key in ["angle","prim_type"]:
            setattr(cfg,key,value)
    return cfg

def dome_light_cfg(data:dict)->DomeLightCfg:
    cfg = DomeLightCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(light_cfg(data)).items():
        setattr(cfg,attr,value)
        # pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key in ["texture_file","prim_type","visible_in_primary_ray","texture_format"]:
            setattr(cfg,key,value)
    return cfg

def cylinder_light_cfg(data:dict)->CylinderLightCfg:
    cfg = CylinderLightCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(light_cfg(data)).items():
        setattr(cfg,attr,value)
        # pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key in ["prim_type","length","radius","treat_as_line"]:
            setattr(cfg,key,value)
    return cfg

def sphere_cfg(data:dict)->SphereLightCfg:
    cfg = SphereLightCfg()
    # attributes from SpawnerCfg
    for attr,value in vars(light_cfg(data)).items():
        setattr(cfg,attr,value)
        # pass
    # attributes for LightCfg
    for key, value in data.items():
        if value is None:
            continue
        if key in ["prim_type","radius","treat_as_point"]:
            setattr(cfg,key,value)
    return cfg

prim_type_spawner_cfg_map = {
    "DiskLight": disk_light_cfg, # type: ignore
    "DistantLight": distant_light_cfg, # type: ignore
    "DomeLight": dome_light_cfg, # type: ignore
    "CylinderLight": cylinder_light_cfg, # type: ignore
    "SphereLight": sphere_cfg, # type: ignore
}

spawner_func_cfg_map = {
    "spawn_light": None,
    "spawn_ground_plane": ground_plane_cfg,
    "spawn_from_urdf": urdf_file_cfg,
    "spawn_from_usd": usd_file_cfg,
    "spawn_camera": None
}

actuator_class_type_map = {
    "ActuatorBase": actuator_base_cfg,
    "ImplicitActuator": implicit_actuator_cfg,
    "IdealPDActuatorCfg": ideal_pda_actuator_cfg,
    "DCMotorCfg": dc_motor_cfg,
    "ActuatorNetLSTMCfg": actuator_net_lstm_cfg,
    "ActuatorNetMLPCfg": actuator_net_mlp_cfg,
    "DelayedPDActuatorCfg": delayed_pd_actuator_cfg,
    "RemotizedPDActuatorCfg": remotized_pd_actuator_cfg,

}
