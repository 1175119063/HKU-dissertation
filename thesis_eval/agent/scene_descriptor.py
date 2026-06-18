"""
Scene Descriptor — Converts RoboCasa simulator state to LLM-readable text.

Uses simulator privileged info to generate an accurate textual description
of the kitchen scene, including object positions, fixture states, and robot state.
This avoids the need for a vision model (DeepSeek is text-only).
"""

import numpy as np


class SceneDescriptor:
    """Generate rich text descriptions of RoboCasa scenes."""

    def __init__(self, env):
        self._env = env
        self._object_registry = self._build_object_registry()

    def _build_object_registry(self):
        """Build a list of manipulable objects and fixtures from the MuJoCo model."""
        model = self._env._env.sim.model
        objects = []
        fixtures = []

        for i, name in enumerate(model.body_names):
            # Skip internal/world bodies
            if name in ("world", "table", "floor", "robot0"):
                continue
            if any(skip in name for skip in ("_link", "_sensor", "_marker", "visual", "collision", "deco_")):
                continue
            if name.startswith("robot0"):
                continue

            pos = model.body_pos[i].copy()
            # Classify
            if any(kw in name for kw in ("cabinet", "drawer", "door", "sink", "stove", "fridge", "microwave", "oven", "dishwasher", "toaster", "blender", "kettle", "faucet")):
                # Check if it has a hinge joint (articulated fixture)
                has_hinge = any(f"{name}_hinge" in j for j in model.joint_names)
                fixtures.append({"name": name, "pos": pos, "articulated": has_hinge})
            else:
                objects.append({"name": name, "pos": pos})

        return {"objects": objects, "fixtures": fixtures}

    def describe(self, task_description=""):
        """Generate a full scene description for the LLM."""
        obs = self._env._env._get_observations()
        if obs is None:
            return "Error: cannot get observation."

        robot_state = self._get_robot_state(obs)
        objects_text = self._get_objects_nearby()
        fixtures_text = self._get_fixture_states()
        gripper_obj = self._get_gripper_object()

        parts = []
        parts.append(f"## 任务\n{task_description}")
        parts.append(f"\n## 机器人状态\n{robot_state}")
        if gripper_obj:
            parts.append(f"手中持有: {gripper_obj}")
        parts.append(f"\n## 附近物体\n{objects_text}")
        parts.append(f"\n## 设备状态\n{fixtures_text}")
        parts.append(f"\n## 可用技能\n"
                      f"1. navigate_to(x, y) - 移动底盘到目标坐标\n"
                      f"2. move_arm_to_object(物体名) - 将机械臂移到物体上方\n"
                      f"3. grasp() - 闭合手爪抓取\n"
                      f"4. release() - 松开手爪\n"
                      f"5. pick(物体名) - 完整抓取流程\n"
                      f"6. place(位置描述) - 放置手中物体\n"
                      f"7. open_fixture(设备名) - 打开柜门/抽屉\n"
                      f"8. close_fixture(设备名) - 关闭柜门/抽屉\n"
                      f"\n请输出下一步要执行的技能和参数，格式：\n"
                      f"{{'skill': '技能名', 'args': [...], 'reason': '原因'}}")

        return "\n".join(parts)

    def _get_robot_state(self, obs):
        """Extract robot state from observation."""
        try:
            jp = obs.get("robot0_joint_pos", np.zeros(10))
            base_xy = jp[0:2]
            base_yaw = jp[2]
            eef_xyz = jp[3:6]
            gripper = "open" if jp[8] > 0.02 else "closed"
            return (f"- 底盘位置: ({base_xy[0]:.2f}, {base_xy[1]:.2f}), 朝向: {base_yaw:.2f} rad\n"
                    f"- 机械臂末端: ({eef_xyz[0]:.2f}, {eef_xyz[1]:.2f}, {eef_xyz[2]:.2f})\n"
                    f"- 手爪状态: {gripper}")
        except Exception:
            return "- 无法获取机器人状态"

    def _get_objects_nearby(self, radius=2.0):
        """List movable objects near the robot."""
        try:
            robot_xy = self._env._env.sim.data.body_xpos[
                self._env._env.sim.model.body_name2id("robot0")
            ][0:2]
        except Exception:
            robot_xy = np.array([0, 0])

        nearby = []
        for obj in self._object_registry["objects"]:
            dist = np.linalg.norm(obj["pos"][0:2] - robot_xy)
            if dist < radius:
                nearby.append(f"  {obj['name']} at ({obj['pos'][0]:.2f}, {obj['pos'][1]:.2f}, {obj['pos'][2]:.2f}), dist={dist:.2f}m")

        if not nearby:
            return "  附近没有物体"
        return "\n".join(nearby[:10])  # limit to 10

    def _get_fixture_states(self):
        """Report state of all fixtures (open/closed)."""
        lines = []
        for fix in self._object_registry["fixtures"]:
            is_open = self._is_open(fix["name"])
            state = "开着" if is_open else "关着"
            lines.append(f"  {fix['name']}: {state}")
        return "\n".join(lines) if lines else "  无设备信息"

    def _is_open(self, fixture_name):
        """Check if a fixture is open."""
        try:
            joint_names = [
                f"{fixture_name}_hinge",
                f"{fixture_name}_joint",
                f"{fixture_name}_door_joint",
            ]
            for jn in joint_names:
                try:
                    joint_id = self._env._env.sim.model.joint_name2id(jn)
                    angle = self._env._env.sim.data.qpos[joint_id]
                    if abs(angle) > 0.1:
                        return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    def _get_gripper_object(self):
        """Check if robot is holding something."""
        try:
            # Check contact between gripper and objects
            gripper_qpos = self._env._env.sim.data.qpos[
                self._env._env.sim.model.joint_name2id("gripper0_finger_joint1")
            ]
            if gripper_qpos < 0.015:  # mostly closed
                for contact in self._env._env.sim.data.contact:
                    # Simple heuristic: check if gripper link is in contact
                    name1 = self._env._env.sim.model.geom_id2name(contact.geom1)
                    name2 = self._env._env.sim.model.geom_id2name(contact.geom2)
                    if name1 and name2 and "gripper" in str(name1).lower() and "gripper" not in str(name2).lower():
                        return str(name2).split("_")[0]
            return None
        except Exception:
            return None
