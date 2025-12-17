# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
import numpy

""" OpenTelevision """
from dex_retargeting.retargeting_config import RetargetingConfig

""" PsiLab Modules  """ 
from psilab import PSILAB_URDF_ASSET_DIR
from psilab.devices.open_television.vuer_tp_cfg import VuerTpCfg


VUER_PSI_DC_01_CFG = VuerTpCfg(
    device = "cuda:0",
    resolution=(720,1280),
    head_pos = (0.2,0.0,1.6),
    eye_left_offset = (0.0,0.033,0.0),
    eye_right_offset = (0.0,-0.033,0.0),
    hand_left_offset = (0.0,0.0,1.3),
    hand_right_offset = (0.0,0.0,1.3),
    # hand_vr2usd = numpy.array([[0, 0, 1, 0],
    #                         [0.9396926, -0.3420202,  0.0000000, 0],
    #                         [0.3420202,  0.9396926, -0.0000000, 0],
    #                         [0, 0, 0, 1]]),
    hand_vr2usd = numpy.array([[0, 0, 1, 0],
                                [-0.8660254,  0.5000000,  0.0000000, 0],
                                [-0.5000000, -0.8660254,  0.0000000, 0],
                                [0, 0, 0, 1]]),
    finger_vr2usd = numpy.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 0, 1]]),
    head_optimize = numpy.array([[0.7071068,  0.0000000,  0.7071068, 0],
                                [0.0000000,  1.0000000,  0.0000000, 0],
                                [-0.7071068,  0.0000000,  0.7071068, 0],
                                [0, 0, 0, 1]]),
    cert_file= "source/psilab/psilab/devices/open_television/cert.pem",
    key_file = "source/psilab/psilab/devices/open_television/key.pem",
    left_hand_retarget_cfg = RetargetingConfig(
            type="vector",
            urdf_path=PSILAB_URDF_ASSET_DIR + "/robots/InspiredHand_OY/InspireHand_OY_Left/InspireHand_OY_Left.urdf",    #urdf相对路径
            wrist_link_name="hand1_link_base",  #腕部link name
            # 需要retarget的joint name
            target_joint_names=[
                "hand1_joint_link_1_1",
                "hand1_joint_link_1_2",
                "hand1_joint_link_1_3",
                "hand1_joint_link_2_1",
                "hand1_joint_link_2_2",
                "hand1_joint_link_3_1",
                "hand1_joint_link_3_2",
                "hand1_joint_link_4_1",
                "hand1_joint_link_4_2",
                "hand1_joint_link_5_1",
                "hand1_joint_link_5_2"
            ],  
            # 每根手指的根部link name
            target_origin_link_names=[
                "hand1_link_base",
                "hand1_link_base",
                "hand1_link_base",
                "hand1_link_base",
                "hand1_link_base",
            ],
            # 每根手指的远端link name
            target_task_link_names = [
                "hand1_link_1_4",
                "hand1_link_2_3",
                "hand1_link_3_3",
                "hand1_link_4_3",
                "hand1_link_5_3",
            ],
            scaling_factor=1.1,
            target_link_human_indices=numpy.array([ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ]),
            low_pass_alpha = 0.5
        ),
    right_hand_retarget_cfg = RetargetingConfig(
            type="vector",
            urdf_path=PSILAB_URDF_ASSET_DIR + "/robots/InspiredHand_OY/InspireHand_OY_Right/InspireHand_OY_Right.urdf",  #urdf相对路径
            wrist_link_name="hand2_link_base",  #腕部link name
            # 需要retarget的joint name
            target_joint_names=[
                "hand2_joint_link_1_1",
                "hand2_joint_link_2_1",
                "hand2_joint_link_3_1",
                "hand2_joint_link_4_1",
                "hand2_joint_link_5_1",
                "hand2_joint_link_1_2",
                "hand2_joint_link_2_2",
                "hand2_joint_link_3_2",
                "hand2_joint_link_4_2",
                "hand2_joint_link_5_2",
                "hand2_joint_link_1_3"
            ],
            # 每根手指的根部link name
            target_origin_link_names=[
                "hand2_link_base",
                "hand2_link_base",
                "hand2_link_base",
                "hand2_link_base",
                "hand2_link_base",
            ],
            # 每根手指的远端link name
            target_task_link_names = [
                "hand2_link_1_4",
                "hand2_link_2_3",
                "hand2_link_3_3",
                "hand2_link_4_3",
                "hand2_link_5_3",
            ],
            scaling_factor=1.1,
            target_link_human_indices=numpy.array([ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ]),
            low_pass_alpha = 0.5
        ),

    hand_retarget_indexs = [0,3,5,7,9,1],
    hand_scale=[1.5,1.0,1.0],

    )

VUER_PSI_DC_02_CFG = VuerTpCfg(
    device = "cuda:0",
    resolution=(720,1280),
    head_pos = (0.2,0.0,1.6),
    eye_left_offset = (0.0,0.033,0.0),
    eye_right_offset = (0.0,-0.033,0.0),
    hand_left_offset = (0.0,0.0,1.3),
    hand_right_offset = (0.0,0.0,1.3),
    # hand_vr2usd = numpy.array([[0, 0, 1, 0],
    #                         [0.9396926, -0.3420202,  0.0000000, 0],
    #                         [0.3420202,  0.9396926, -0.0000000, 0],
    #                         [0, 0, 0, 1]]),
    hand_vr2usd = numpy.array([[-0.0000000, -0.0000000,  1.0000000, 0],
                                [0.0000000, -1.0000000,  0.0000000, 0],
                                [1.0000000,  0.0000000,  0.0000000, 0],
                                [0, 0, 0, 1]]),
    finger_vr2usd = numpy.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 0, 1]]),
    head_optimize = numpy.array([[0.7071068,  0.0000000,  0.7071068, 0],
                                [0.0000000,  1.0000000,  0.0000000, 0],
                                [-0.7071068,  0.0000000,  0.7071068, 0],
                                [0, 0, 0, 1]]),
    cert_file= "source/psilab/psilab/devices/open_television/cert.pem",
    key_file = "source/psilab/psilab/devices/open_television/key.pem",
    left_hand_retarget_cfg = RetargetingConfig(
            type="vector",
            urdf_path=PSILAB_URDF_ASSET_DIR + "/robots/InspiredHand_OY/InspireHand_OY_Left/InspireHand_OY_Left.urdf",    #urdf相对路径
            wrist_link_name="hand1_link_base",  #腕部link name
            # 需要retarget的joint name
            target_joint_names=[
                "hand1_joint_link_1_1",
                "hand1_joint_link_1_2",
                "hand1_joint_link_1_3",
                "hand1_joint_link_2_1",
                "hand1_joint_link_2_2",
                "hand1_joint_link_3_1",
                "hand1_joint_link_3_2",
                "hand1_joint_link_4_1",
                "hand1_joint_link_4_2",
                "hand1_joint_link_5_1",
                "hand1_joint_link_5_2"
            ],  
            # 每根手指的根部link name
            target_origin_link_names=[
                "hand1_link_base",
                "hand1_link_base",
                "hand1_link_base",
                "hand1_link_base",
                "hand1_link_base",
            ],
            # 每根手指的远端link name
            target_task_link_names = [
                "hand1_link_1_4",
                "hand1_link_2_3",
                "hand1_link_3_3",
                "hand1_link_4_3",
                "hand1_link_5_3",
            ],
            scaling_factor=1.1,
            target_link_human_indices=numpy.array([ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ]),
            low_pass_alpha = 0.5
        ),
    right_hand_retarget_cfg = RetargetingConfig(
            type="vector",
            urdf_path=PSILAB_URDF_ASSET_DIR + "/robots/InspiredHand_OY/InspireHand_OY_Right/InspireHand_OY_Right.urdf",  #urdf相对路径
            wrist_link_name="hand2_link_base",  #腕部link name
            # 需要retarget的joint name
            target_joint_names=[
                "hand2_joint_link_1_1",
                "hand2_joint_link_2_1",
                "hand2_joint_link_3_1",
                "hand2_joint_link_4_1",
                "hand2_joint_link_5_1",
                "hand2_joint_link_1_2",
                "hand2_joint_link_2_2",
                "hand2_joint_link_3_2",
                "hand2_joint_link_4_2",
                "hand2_joint_link_5_2",
                "hand2_joint_link_1_3"
            ],
            # 每根手指的根部link name
            target_origin_link_names=[
                "hand2_link_base",
                "hand2_link_base",
                "hand2_link_base",
                "hand2_link_base",
                "hand2_link_base",
            ],
            # 每根手指的远端link name
            target_task_link_names = [
                "hand2_link_1_4",
                "hand2_link_2_3",
                "hand2_link_3_3",
                "hand2_link_4_3",
                "hand2_link_5_3",
            ],
            scaling_factor=1.1,
            target_link_human_indices=numpy.array([ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ]),
            low_pass_alpha = 0.5
        ),

    hand_retarget_indexs = [0,3,5,7,9,1],
    hand_scale=[1.5,1.0,1.0],

    )
