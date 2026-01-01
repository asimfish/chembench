# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-07-08
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import TYPE_CHECKING, Any

from collections.abc import Sequence
import weakref
""" Common Modules  """ 
import re

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.assets.asset_base import AssetBase
import isaacsim.core.utils.prims as prim_utils
import omni.kit.app
import omni.timeline

""" PsiLab Modules  """ 

if TYPE_CHECKING:
    from .light_cfg import LightBaseCfg

class Light(AssetBase):

    cfg: LightBaseCfg

    def __init__(self, cfg: LightBaseCfg):

        # super().__init__(cfg)

        # check that the config is valid
        cfg.validate() # type: ignore
        # store inputs
        self.cfg = cfg.copy() # type: ignore
        # flag for whether the asset is initialized
        self._is_initialized = False

        # check if base asset path is valid
        # note: currently the spawner does not work if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Robot_[1,2]" since the spawner will not
        #   know which prim to spawn. This is a limitation of the spawner and not the asset.
        asset_path = self.cfg.prim_path.split("/")[-1]
        asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None
        # spawn the asset
        if self.cfg.spawn is not None and not asset_path_is_regex:
            self.cfg.spawn.func(
                self.cfg.prim_path,
                self.cfg.spawn,
                translation=self.cfg.init_state.pos,
                orientation=self.cfg.init_state.rot,
            )
        # BUG: asset spawned as no order, but light without spawn is dependented on env usd which has light.
        # That means light is probably not in stage while light init, so  check prim exist will get error
        # check that spawn was successful
        # matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        # if len(matching_prims) == 0:
        #     raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")

        # # note: Use weakref on all callbacks to ensure that this object can be deleted when its destructor is called.
        # # add callbacks for stage play/stop
        # # The order is set to 10 which is arbitrary but should be lower priority than the default order of 0
        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj=weakref.proxy(self): obj._initialize_callback(event),
            order=10,
        )
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
    
    @property
    def num_instances(self) -> int:
        return 0

    @property
    def data(self) -> Any:
        return 0
    

    @property
    def color(self, env_ids :int = 0) -> tuple[float,float,float]:

        asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", self.cfg.prim_path) is None
        #
        if asset_path_is_regex:
            prim_path = self.cfg.prim_path
            re.sub("/env_[0-9]+/","/env_"+ str(env_ids),prim_path,1)
        #
        prim = prim_utils.get_prim_at_path(self.cfg.prim_path)
        color = prim.GetAttribute("inputs:color")
        
        return color.Get() if color.IsValid() else (0.0,0.0,0.0)

    @property
    def intensity(self, env_ids :int = 0) -> float:

        asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", self.cfg.prim_path) is None
        #
        if asset_path_is_regex:
            prim_path = self.cfg.prim_path
            re.sub("/env_[0-9]+/","/env_"+ str(env_ids),prim_path,1)
        #
        prim = prim_utils.get_prim_at_path(self.cfg.prim_path)
        intensity = prim.GetAttribute("inputs:intensity")
        
        return intensity.Get() if intensity.IsValid() else 0.0

    def reset(self, env_ids: Sequence[int] | None = None):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        pass

    """
    Implementation specific.
    """

    def _initialize_impl(self):
        pass
