"""
DP Policy — wraps trained Diffusion Policy model for eval harness.
"""

import os as _os, sys as _sys
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)

import numpy as np
import torch
from PIL import Image

from bc.dp_model import DiffusionPolicy


class DPPolicyWrapper:

    def __init__(self, task_name: str, checkpoint_path: str = None, image_size: int = 128):
        self.task_name = task_name
        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiffusionPolicy(image_size=image_size, diffusion_steps=50).to(self.device)

        if checkpoint_path and _os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def predict(self, obs, info):
        img_key = None
        for k in obs:
            if "agentview" in k and "video" in k:
                img_key = k
                break
        if img_key is None:
            return self._zero_action()

        img = Image.fromarray(obs[img_key])
        w, h = img.size
        sz = min(w, h)
        img = img.crop(((w - sz) // 2, (h - sz) // 2, (w + sz) // 2, (h + sz) // 2))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        state_keys = [
            "state.base_position", "state.base_rotation",
            "state.end_effector_position_relative", "state.end_effector_rotation_relative",
            "state.gripper_qpos",
        ]
        state_parts = []
        for k in state_keys:
            if k in obs:
                v = obs[k]
                state_parts.append(v.flatten() if isinstance(v, np.ndarray) else [v])
        if not state_parts:
            return self._zero_action()
        state = np.concatenate(state_parts).astype(np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_raw = self.model.predict_action(img_tensor, state_tensor).cpu().numpy()

        return {
            "action.end_effector_position": action_raw[5:8].astype(np.float32),
            "action.end_effector_rotation": action_raw[8:11].astype(np.float32),
            "action.gripper_close": action_raw[11:12].astype(np.float32),
            "action.base_motion": action_raw[0:4].astype(np.float32),
            "action.control_mode": action_raw[4:5].astype(np.float32),
        }

    def _zero_action(self):
        return {
            "action.end_effector_position": np.zeros(3, dtype=np.float32),
            "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
            "action.gripper_close": np.array([0.0], dtype=np.float32),
            "action.base_motion": np.zeros(4, dtype=np.float32),
            "action.control_mode": np.array([0.0], dtype=np.float32),
        }

    def reset(self):
        pass

    def run_episode(self, *args, **kwargs):
        return False, 0
