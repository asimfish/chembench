# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-17
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" IsaacLab Modules  """ 
from isaaclab.sim import SimulationCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.utils import configclass

""" PsiLab Modules  """ 
from psilab.scene import SceneCfg

@configclass
class CTEnvCfg:
    """Configuration for an reinforcement learning environment.
    """

    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""

    # ui settings
    ui_window_class_type: type | None = BaseEnvWindow
    """The class type of the UI window. Default is None.

    If None, then no UI window is created.

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """

    # general settings
    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed is not set.

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """

    

    # environment settings
    # Change Scene Config From InteractiveSceneCfg To Psi SceneCfg
    # Author: Feng Yunduo
    # Time: 2025-04-07    
    # Code Bak: scene: InteractiveSceneCfg = MISSING
    scene: SceneCfg = MISSING   # type: ignore
    """Scene settings.

    Please refer to the :class:`psilab.scene.SceneCfg` class for more details.
    """

    

    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    """

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True."""


    # # Add other flags, Author: Feng Yunduo, Date:2025-04-17, Start
    # enable_wandb: bool = False # type: ignore
    # """Whether log data to wandb"""

    # enable_eval: bool = False # type: ignore
    # """Whether print eval result"""

    # enable_output : bool = False  # type: ignore
    # """ Whether Save Data or not. """

    # output_folder: str | None = MISSING  # type: ignore
    # """ Data Ouptut Folder. """

    # sample_step : int = 1  # type: ignore
    # """ Simulator step numbers per Sample step. """
    
    # async_reset : bool = False
    # """ Environments asynchronous reset, now is only for recording data while playing. """

    # enable_random : bool = False  # type: ignore
    # """ Whether enable random or not. """
    
    # enable_marker : bool = False  # type: ignore
    # """ Whether show marker or not. """
    # # Add other flags, Author: Feng Yunduo, Date:2025-04-17, End
