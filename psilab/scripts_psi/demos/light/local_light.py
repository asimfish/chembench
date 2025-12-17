# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-07
# Vesion: 1.0


"""
===============================================================================
Description:
    This script is used to demonstrate how to load and randomize local light
    in PsiLab.
Usage:
    1. conda activate <your_environment_name> (e.g. psilab)
    2. python global_light.py

===============================================================================
"""

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates random demo")

""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)


""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext,SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.envs.common import ViewerCfg

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene.sence import Scene,SceneCfg
from psilab.assets.light.light_cfg import DomeLightCfg,RectLightCfg
from psilab.random.random_cfg import RandomCfg,LightRandomCfg


# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(2.2,0.0,1.2),
    lookat=(-15.0,0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 240, 
    render_interval=1,
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 32,
        max_velocity_iteration_count = 4,
        bounce_threshold_velocity = 0.002,
        enable_ccd=True,
        gpu_found_lost_pairs_capacity = 137401003
    ),
    render=RenderCfg(),
)

# scene config
SCENE_CFG = SceneCfg(
        
        num_envs = 1, 
        env_spacing=10.0, 
        replicate_physics=True,

        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=0.0, 
                color=(0.75, 0.75, 0.75)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0,0,2)
            )
        ),
        # local light
        local_lights_cfg={
            "rect_light_01":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_01",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_02":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_02",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_03":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_03",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_04":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_04",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_05":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_05",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_06":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_06",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_07":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_07",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_08":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_08",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_09":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_09",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_10":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_10",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_11":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_11",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_12":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_12",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
            "rect_light_13":RectLightCfg(
                prim_path="/World/envs/env_[0-9]+/Room/Lights/RectLight_13",
                light_type="RectLight",
                spawn=None,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-2,1,2)
                )
            ),
        },

        # static object
        static_objects_cfg = {
            "room" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Room", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_garage/GarageScene.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (0.707, 0.707, 0.0, 0.0)
                )
            ),
        },
        
        # random config
        random = RandomCfg(
            local_lights_cfg = {
                "rect_light_01": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_02": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_03": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_04": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_05": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_06": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_07": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_08": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_09": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_10": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_11": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_12": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                ),
                "rect_light_13": LightRandomCfg(
                    random_type="range",
                    random_intensity= True,
                    random_color=True,
                    intensity_range=[20000,60000],
                    color_range = [
                        [0,0,0],
                        [255,255,255]
                    ], # type: ignore
                    intensity_list=[
                        0,
                        1000,
                        8000
                    ],
                    color_list=[
                        [0,0,0],
                        [210,124,14],
                        [0,30,20],
                        [90,0,100],
                    ] # type: ignore
                )
            }
        ),
    )

# create a simulation context to control the simulator
if SimulationContext.instance() is None:
    sim: SimulationContext = SimulationContext(SIM_CFG)
else:
    raise RuntimeError("Simulation context already exists. Cannot create a new one.")

# create scene
scene = Scene(SCENE_CFG)
# 
sim.reset()
sim.set_camera_view(eye=VIEWER_CFG.eye, target=VIEWER_CFG.lookat)
#
step = 0
step_max = 100
# main loop
while(True):
    # sim reset
    if step % step_max == 0:
        scene.reset()
    #
    sim.step()
    scene.update(scene.physics_dt)
    step+=1


