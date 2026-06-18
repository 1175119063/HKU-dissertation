"""
Skill Library for RoboCasa Mobile Manipulation.

Correct action layout (12-dim for HybridMobileBase controller):
  [0:3]   = arm delta position (OSC_POSE, max ±5cm/step)
  [3:6]   = arm delta rotation
  [6]     = gripper (-1=close, 1=open)
  [7:10]  = base velocity [vx, vy, omega] (JOINT_VELOCITY)
  [10]    = torso position
  [11]    = mode (>0=base mode, <0=arm mode)
"""

import numpy as np


class SkillLibrary:

    def __init__(self, env):
        self._env = env  # raw robocasa inner env

    # ── Helpers ──────────────────────────────────────────────

    def _idle(self):
        return np.zeros(12, dtype=np.float32)

    def _step(self, action):
        return self._env.step(action)

    def _get_raw_obs(self):
        return self._env._get_observations()

    def _get_eef_xyz(self):
        return self._get_raw_obs().get("robot0_eef_pos", np.zeros(3))

    def _get_gripper_open(self):
        q = self._get_raw_obs().get("robot0_gripper_qpos", np.zeros(2))
        return q[0] > 0.015

    def _get_robot_xy(self):
        """Base XY from MuJoCo body."""
        try:
            bid = self._env.sim.model.body_name2id("mobilebase0_wheeled_base")
            return self._env.sim.data.body_xpos[bid][:2].copy()
        except Exception:
            return np.zeros(2)

    def _get_body_pos(self, name):
        try:
            bid = self._env.sim.model.body_name2id(name)
            return self._env.sim.data.body_xpos[bid].copy()
        except Exception:
            return None

    # ── Arm Movement ─────────────────────────────────────────

    def move_arm_to(self, target_xyz, steps=800):
        """Move EEF to world-frame XYZ. NOTE: x,y inverted (base faces -x)."""
        for _ in range(steps):
            eef = self._get_eef_xyz()
            error = target_xyz - eef
            if np.linalg.norm(error) < 0.04:
                return True
            a = self._idle()
            a[0] = np.clip(-error[0], -0.05, 0.05)  # x inverted
            a[1] = np.clip(-error[1], -0.05, 0.05)  # y inverted
            a[2] = np.clip(error[2], -0.05, 0.05)   # z correct
            a[11] = -1.0
            self._step(a)
        return np.linalg.norm(target_xyz - self._get_eef_xyz()) < 0.08

    def move_arm_up(self, steps=150):
        for _ in range(steps):
            a = self._idle()
            a[2] = 0.05; a[11] = -1.0
            self._step(a)

    def move_arm_down(self, target_z, steps=800):
        for _ in range(steps):
            cur_z = self._get_eef_xyz()[2]
            if cur_z <= target_z + 0.02:
                return True
            a = self._idle()
            # Use max speed when far, slow when close
            error = target_z - cur_z
            speed = -0.05 if error < -0.05 else max(-1.0, error * 3)
            a[2] = max(-1.0, speed)  # max downward = -1.0
            a[11] = -1.0
            self._step(a)
        return True

    # ── Gripper ───────────────────────────────────────────────

    def grasp(self, steps=60):
        for _ in range(steps):
            a = self._idle()
            a[6] = 1.0; a[11] = -1.0   # +1 = close
            self._step(a)
        return not self._get_gripper_open()

    def release(self, steps=30):
        for _ in range(steps):
            a = self._idle()
            a[6] = -1.0; a[11] = -1.0  # -1 = open
            self._step(a)

    # ── Navigation ────────────────────────────────────────────

    def navigate_to(self, target_xy, steps=800):
        """Move base toward target.
        NOTE: base velocity axes are INVERTED relative to world frame.
        a[7]=+1 → base moves -x, a[8]=+1 → base moves -y.
        """
        best_dist = float('inf')
        for _ in range(steps):
            cur = self._get_robot_xy()
            error = target_xy - cur
            dist = np.linalg.norm(error)
            best_dist = min(best_dist, dist)
            if dist < 0.5:
                return True
            # Negate: base frame is inverted relative to world
            a = self._idle()
            a[7] = -error[0]  # sign flipped
            a[8] = -error[1]  # sign flipped
            a[11] = 1.0
            self._step(a)
        return best_dist < 0.8

    # ── Pick & Place ──────────────────────────────────────────

    _NAME_MAP = {"目标物体": "obj_main"}

    def _resolve_body(self, name):
        if name in self._NAME_MAP:
            return self._NAME_MAP[name]
        try:
            self._env.sim.model.body_name2id(name)
            return name
        except Exception:
            pass
        for bn in self._env.sim.model.body_names:
            if name.lower() in bn.lower():
                return bn
        return None

    def pick(self, object_name):
        body = self._resolve_body(object_name)
        if not body:
            return False
        pos = self._get_body_pos(body)
        if pos is None:
            return False
        # Step 1: align xy above object (2000 steps for precision)
        above = np.array([pos[0], pos[1], pos[2] + 0.08])
        self.move_arm_to(above, steps=2000)
        # Step 2: descend EEF so fingers reach object mid-height (finger ~4cm above EEF)
        self.move_arm_down(pos[2] - 0.05, steps=1500)
        # Step 3: close gripper
        self.grasp()
        # Step 4: lift
        self.move_arm_up()
        return True

    def move_arm_up(self, steps=300):
        for _ in range(steps):
            a = self._idle()
            a[2] = 0.05; a[11] = -1.0
            self._step(a)

    def place_at(self, target_xyz):
        approach = target_xyz + np.array([0, 0, 0.12])
        self.move_arm_to(approach)
        self.move_arm_down(target_xyz[2] + 0.02)
        self.release()
        self.move_arm_up()
        return True

    # ── Fixture ───────────────────────────────────────────────

    _FIXTURE_PATTERNS = {
        "柜子": ["hingecabinet", "cabinet", "cab_"],
        "台面": ["counter_"], "水槽": ["sink_"],
        "灶台": ["stovetop"], "冰箱": ["fridge"],
        "洗碗机": ["dishwasher"], "烤箱": ["oven_"],
        "微波炉": ["microwave_"], "抽屉": ["drawer"],
    }

    def _resolve_fixture(self, name):
        body = self._resolve_body(name)
        if body:
            return body
        patterns = self._FIXTURE_PATTERNS.get(name, [name])
        robot_xy = self._get_robot_xy()
        best, best_dist = None, float('inf')
        for bn in self._env.sim.model.body_names:
            if not bn.endswith('_main'):
                continue
            # Skip handles, doors, knobs, shelves, distractors
            if any(x in bn for x in ('handle', 'door', 'knob', 'hinge', 'rack', 'level', 'inner', 'Clear', 'Shelf', 'distr_')):
                continue
            if any(p in bn for p in patterns):
                pos = self._get_body_pos(bn)
                if pos is not None:
                    d = float(np.linalg.norm(pos[:2] - robot_xy))
                    if d < best_dist:
                        best_dist, best = d, bn
        return best

    def open_fixture(self, fixture_name):
        pos = self._get_body_pos(fixture_name)
        if pos is None:
            return False
        self.navigate_to(pos[:2] - np.array([0.8, 0]))
        self.move_arm_to(pos + np.array([0, 0, 0.05]))
        self.grasp()
        for _ in range(100):
            a = self._idle()
            a[1] = -0.04  # pull toward robot
            a[11] = -1.0
            self._step(a)
        self.release()
        return True

    def close_fixture(self, fixture_name):
        pos = self._get_body_pos(fixture_name)
        if pos is None:
            return False
        self.move_arm_to(pos + np.array([0, 0, 0.05]))
        for _ in range(100):
            a = self._idle()
            a[1] = 0.04   # push away
            a[11] = -1.0
            self._step(a)
        return True

    # ── Scene Description ─────────────────────────────────────

    def get_scene_text(self, task_desc: str, fixture_refs: dict = None) -> str:
        fixture_refs = fixture_refs or {}
        robot_xy = self._get_robot_xy()
        eef = self._get_eef_xyz()
        grp = "closed" if not self._get_gripper_open() else "open"

        parts = [
            f"任务: {task_desc}",
            f"机器人: 底盘({robot_xy[0]:.2f},{robot_xy[1]:.2f}) 手爪({eef[0]:.2f},{eef[1]:.2f},{eef[2]:.2f}) 状态({grp})",
        ]

        # Show fixture refs (target fixtures from task)
        if fixture_refs:
            parts.append("\n🎯 任务目标设备:")
            for ref_name, body_name in fixture_refs.items():
                pos = self._get_body_pos(body_name)
                if pos is not None:
                    d = np.linalg.norm(pos[:2] - robot_xy)
                    parts.append(f"  {ref_name} = {body_name} @ ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) 距离={d:.1f}m")

        parts.append("\n关键物体 (用这些名字调用 pick):")
        for body, label in [("obj_main", "目标物体"), ("distr_counter_main", "台面杂物"), ("distr_cab_main", "柜内杂物")]:
            pos = self._get_body_pos(body)
            if pos is not None:
                d = np.linalg.norm(pos[:2] - robot_xy)
                eef_dist = np.linalg.norm(self._get_eef_xyz() - pos)
                reachable = "✅可达" if eef_dist < 0.8 else f"❌太远({eef_dist:.1f}m>0.8m臂展)"
                parts.append(f"  \"{label}\" ({body}): ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) 距离={d:.1f}m {reachable}")

        parts.append("\n设备:")
        seen = set()
        for bn in self._env.sim.model.body_names:
            if not bn.endswith('_main'):
                continue
            if any(x in bn for x in ('door', 'handle', 'knob', 'hinge', 'rack', 'Clear', 'Button', 'Lid', 'Shelf', 'level', 'inner', 'distr_')):
                continue
            for kw, label in [("counter_", "台面"), ("hingecabinet", "柜子"), ("sink_", "水槽"), ("stovetop", "灶台"), ("fridge", "冰箱"), ("dishwasher", "洗碗机"), ("oven_", "烤箱"), ("microwave_", "微波炉"), ("island", "岛台")]:
                if kw in bn:
                    if label not in seen:
                        seen.add(label)
                        pos = self._get_body_pos(bn)
                        if pos is not None:
                            d = np.linalg.norm(pos[:2] - robot_xy)
                            if d < 3.0:
                                parts.append(f"  [{label}] ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) 距离={d:.1f}m")
                    break

        parts.append(f"\n技能: navigate_to(x,y) | pick(\"物体名\") | place_at(x,y,z) | open_fixture(\"设备名\") | close_fixture(\"设备名\") | done()")
        parts.append("JSON: {\"skill\": \"...\", \"args\": [...], \"thought\": \"...\"}")
        return "\n".join(parts)
