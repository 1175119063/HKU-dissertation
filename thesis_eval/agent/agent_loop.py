"""
Agent Loop — LLM-driven mobile manipulation in RoboCasa.

Loop: describe scene → ask DeepSeek → execute skill → repeat.
"""

import json, re, os
import numpy as np
from openai import OpenAI

import os as _os, sys as _sys
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
if _this_dir not in _sys.path:
    _sys.path.insert(0, _this_dir)

from skills import SkillLibrary

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

SYSTEM_PROMPT = """You control a kitchen mobile robot. Complete the task step by step.

CRITICAL:
1. Arm reach = 0.8m. Check EEF distance in scene before calling pick().
2. pick() argument MUST be EXACTLY one of: "目标物体", "台面杂物", "柜内杂物" — never invent names.
3. open_fixture/close_fixture use names from 🎯 任务目标设备 section (e.g. "cab", "counter").
4. If pick() fails 3 times: first navigate closer, then retry.
5. If same skill fails 4+ times: call done("stuck").
6. Complete task = navigate→pick→navigate to target→open fixture→place→close fixture→done()

JSON only: {"skill": "name", "args": [...], "thought": "brief reason"}

Skills: navigate_to(x,y) | pick("name") | place_at(x,y,z) | open_fixture("name") | close_fixture("name") | done()"""


class AgentLoop:

    MAX_SKILLS = 25
    MAX_IDLE = 4

    def __init__(self, env, task_description: str, fixture_refs: dict = None, verbose: bool = True):
        self._env = env
        self._task = task_description
        self._fixture_refs = fixture_refs or {}
        self._skills = SkillLibrary(env)
        self._client = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL)
        self._history = []
        self._verbose = verbose

    def _log(self, msg):
        if self._verbose:
            print(f"  [Agent] {msg}")

    def reset(self):
        self._history = []

    def run_episode(self, obs, info, max_env_steps=20000):
        self.reset()
        total_steps = 0
        idle_count = 0

        for skill_step in range(self.MAX_SKILLS):
            # 1. Describe scene (with fixture refs)
            scene = self._skills.get_scene_text(self._task, self._fixture_refs)

            # Debug: print scene for first 3 steps
            if False and self._verbose:
                print(f"  [Scene] {'-'*50}")
                for line in scene.split('\n')[:25]:
                    print(f"  [Scene] {line}")
                print(f"  [Scene] {'-'*50}")

            # 2. Ask LLM
            plan = self._ask_llm(scene)
            name = plan.get("skill", "done")
            args = plan.get("args", [])
            thought = plan.get("thought", "")

            self._log(f"第{skill_step+1}步: {name}({args}) — {thought}")
            if name == "done":
                obs, r, done, info = self._env.step(self._skills._idle())
                total_steps += 1
                success = self._env._check_success() if hasattr(self._env, '_check_success') else False
                return success, total_steps

            if not name or name == "unknown":
                idle_count += 1
                if idle_count >= self.MAX_IDLE:
                    return False, total_steps
                continue
            idle_count = 0

            # 4. Execute skill
            steps_used, feedback = self._exec(name, args)
            total_steps += steps_used
            if feedback:
                self._history.append({"role": "user", "content": f"[执行结果] {feedback}"})

            # 5. Check success after each skill
            obs, r, done, info = self._env.step(self._skills._idle())
            total_steps += 1
            success = self._env._check_success() if hasattr(self._env, '_check_success') else False
            if success:
                return True, total_steps
            if total_steps >= max_env_steps:
                return False, total_steps

        return False, total_steps

    def _ask_llm(self, scene_text: str) -> dict:
        """Send scene to DeepSeek, get back a skill plan."""
        self._history.append({"role": "user", "content": scene_text})
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history[-6:]

        try:
            resp = self._client.chat.completions.create(
                model="deepseek-chat",
                messages=msgs,
                temperature=0.1,
                max_tokens=300,
            )
            reply = resp.choices[0].message.content.strip()
            parsed = self._parse(reply)
            self._history.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            return parsed
        except Exception as e:
            return {"skill": "done", "args": [], "thought": f"API error: {e}"}

    @staticmethod
    def _parse(text: str) -> dict:
        text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*\}', text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {"skill": "done", "args": [], "thought": "parse failed"}

    def _exec(self, name: str, args: list) -> tuple[int, str]:
        """Execute a skill, return (env_steps_used, feedback_message)."""
        try:
            if isinstance(args, str):
                args = [args]

            if name == "navigate_to":
                x, y = float(args[0]), float(args[1])
                ok = self._skills.navigate_to((x, y))
                cur = self._skills._get_robot_xy()
                dist = np.linalg.norm(np.array([x, y]) - cur)
                fb = f"navigate_to({x:.2f},{y:.2f}) {'到达' if ok else '接近'}目标，现在位置({cur[0]:.2f},{cur[1]:.2f})，距目标{dist:.2f}m"
                return 50, fb  # minimal budget hit

            elif name == "pick":
                arg = args
                while isinstance(arg, (list, tuple)):
                    arg = arg[0] if len(arg) > 0 else "目标物体"
                ok = self._skills.pick(str(arg))
                grp = "closed" if not self._skills._get_gripper_open() else "open"
                fb = f"pick({arg}) {'成功' if ok else '失败'}，手爪{grp}"
                return 50, fb

            elif name == "place_at":
                x, y, z = float(args[0]), float(args[1]), float(args[2])
                ok = self._skills.place_at(np.array([x, y, z]))
                fb = f"place_at({x:.2f},{y:.2f},{z:.2f}) {'完成' if ok else '失败'}"
                return 50, fb

            elif name == "open_fixture":
                arg = args[0] if isinstance(args, (list, tuple)) and len(args) > 0 else str(args)
                # First check fixture_refs for exact body name
                fixture_body = self._fixture_refs.get(str(arg))
                if not fixture_body:
                    fixture_body = self._skills._resolve_fixture(str(arg))
                if fixture_body:
                    ok = self._skills.open_fixture(fixture_body)
                    fb = f"open_fixture({arg}→{fixture_body}) {'成功' if ok else '失败'}"
                else:
                    ok, fb = False, f"open_fixture({arg}) 失败: 找不到设备"
                return 50, fb

            elif name == "close_fixture":
                arg = args[0] if isinstance(args, (list, tuple)) and len(args) > 0 else str(args)
                fixture_body = self._fixture_refs.get(str(arg))
                if not fixture_body:
                    fixture_body = self._skills._resolve_fixture(str(arg))
                if fixture_body:
                    ok = self._skills.close_fixture(fixture_body)
                    fb = f"close_fixture({arg}→{fixture_body}) {'成功' if ok else '失败'}"
                else:
                    ok, fb = False, f"close_fixture({arg}) 失败: 找不到设备"
                return 50, fb

            return 50, f"未知技能: {name}"
        except Exception as e:
            return 50, f"错误: {str(e)[:80]}"
