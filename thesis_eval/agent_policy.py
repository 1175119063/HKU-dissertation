"""
Agent Policy — Wraps the LLM-driven agent for the eval harness.
"""

import os as _os, sys as _sys
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)

import json, numpy as np
from agent.agent_loop import AgentLoop


TASK_DESCRIPTIONS = {
    "PickPlaceCounterToCabinet": "从台面上抓起目标物体，放到柜子里",
    "PickPlaceCabinetToCounter": "从柜子里取出目标物体，放到台面上",
    "PickPlaceCounterToStove": "从台面上抓起目标物体，放到灶台上",
    "PickPlaceSinkToCounter": "从水槽里抓起目标物体，放到台面上",
    "OpenCabinet": "打开柜门",
    "CloseCabinet": "关上柜门",
    "TurnOnSinkFaucet": "打开水龙头",
    "NavigateKitchen": "导航到厨房指定位置",
    "PrepareCoffee": "准备咖啡",
    "LoadDishwasher": "将餐具装入洗碗机",
    "PreSoakPan": "将锅放到水槽里浸泡",
    "MoveToCounter": "将物体移到台面上",
    "ClearSink": "清理水槽区域",
    "DeliverStraw": "将吸管送到指定位置",
    "ArrangeBreadBasket": "整理面包篮",
}


class AgentPolicy:

    def __init__(self, task_name: str):
        self.task_name = task_name

    def predict(self, obs, info):
        return {
            "action.end_effector_position": np.zeros(3, dtype=np.float32),
            "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
            "action.gripper_close": np.array([0.0], dtype=np.float32),
            "action.base_motion": np.zeros(4, dtype=np.float32),
            "action.control_mode": np.array([0.0], dtype=np.float32),
        }

    def reset(self):
        pass

    def run_episode(self, gym_env, obs, info, max_steps) -> tuple:
        try:
            from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
            e = gym_env
            for _ in range(6):
                if isinstance(e, RoboCasaGymEnv):
                    break
                e = getattr(e, 'env', None) or getattr(e, '_env', None)
            inner_env = e.env

            # Get task metadata
            task_desc = TASK_DESCRIPTIONS.get(self.task_name, self.task_name)
            fixture_refs = {}
            if hasattr(inner_env, 'get_ep_meta'):
                meta = inner_env.get_ep_meta()
                if meta.get('lang'):
                    task_desc = meta['lang']
                if meta.get('fixture_refs'):
                    fixture_refs = meta['fixture_refs']

            agent = AgentLoop(inner_env, task_desc, fixture_refs)
            success, steps = agent.run_episode(obs, info, max_steps)
            return success, steps

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, 0
