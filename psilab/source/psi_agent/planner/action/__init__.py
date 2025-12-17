from .grasp import GraspStage, PickStage, HookStage
from .place import PlaceStage
from .insert import InsertStage, HitStage
from .slide import SlideStage
from .pour import PourStage
from .pull import PullStage
from .push import PushStage, ClickStage

ACTION_STAGE = {
    "grasp": GraspStage,
    "pick": PickStage,
    "hook": HookStage,
    "place": PlaceStage,
    "insert": InsertStage,
    "slide": SlideStage,
    "shave": NotImplemented,
    "brush": NotImplemented,
    "wipe": NotImplemented,
    "hit": NotImplemented,
    "pour": PourStage,
    "push": PushStage,
    'click': ClickStage,
    'touch': ClickStage,
    "pull": PullStage
}

def build_stage(action):
    if action not in ACTION_STAGE:
        raise NotImplementedError
    return ACTION_STAGE[action]
