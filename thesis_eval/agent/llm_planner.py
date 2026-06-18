"""
LLM Planner — Sends scene descriptions to DeepSeek API, gets skill plans back.

Uses OpenAI-compatible API format. DeepSeek base URL: https://api.deepseek.com
"""

import json
import re
from openai import OpenAI


class LLMPlanner:
    """Calls DeepSeek API to plan the next skill for a given scene."""

    SYSTEM_PROMPT = """你是一个厨房移动操作机器人（PandaOmron：Franka 机械臂 + 全向轮式底盘）。

你可以调用以下技能：
- pick(物体名): 完整抓取流程（移动机械臂到物体上方 → 抓起 → 抬起）
- place(位置描述): 放置手中物体到指定位置（如 counter, sink, stove, cabinet 等）
- navigate_to(x, y): 移动底盘到指定坐标
- open_fixture(设备名): 打开柜门/抽屉
- close_fixture(设备名): 关闭柜门/抽屉

规则：
1. 每次只输出一个动作，不要规划整个序列
2. 只会用上述技能，不要发明新技能
3. 如果任务已完成，输出 {"skill": "done", "args": [], "reason": "任务完成"}
4. 如果卡住了无法继续，输出 {"skill": "done", "args": [], "reason": "无法继续：..."}

输出必须是严格 JSON 格式，不要有任何其它文字：
{"skill": "技能名", "args": ["参数1", "参数2", ...], "reason": "简短原因"}"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._conversation = []

    def reset(self):
        """Clear conversation history for a new episode."""
        self._conversation = []

    def next_skill(self, scene_description: str) -> dict:
        """Ask LLM what to do next given the current scene.

        Returns: {"skill": str, "args": list, "reason": str}
        """
        self._conversation.append({
            "role": "user",
            "content": scene_description
        })

        # Keep context window manageable (last 6 messages)
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages += self._conversation[-6:]

        try:
            response = self._client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.1,
                max_tokens=200,
            )
            reply = response.choices[0].message.content.strip()

            # Parse JSON from reply
            parsed = self._parse_json(reply)
            if parsed is None:
                # Retry with stricter prompt
                self._conversation.append({
                    "role": "user",
                    "content": "你的上一条回复格式不对。请只输出 JSON：{\"skill\": \"...\", \"args\": [...], \"reason\": \"...\"}"
                })
                response = self._client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages + self._conversation[-1:],
                    temperature=0.0,
                    max_tokens=200,
                )
                reply = response.choices[0].message.content.strip()
                parsed = self._parse_json(reply)

            if parsed is None:
                parsed = {"skill": "done", "args": [], "reason": "JSON解析失败"}

            self._conversation.append({
                "role": "assistant",
                "content": json.dumps(parsed, ensure_ascii=False)
            })

            return parsed

        except Exception as e:
            return {"skill": "done", "args": [], "reason": f"API错误: {str(e)[:80]}"}

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from LLM output, handling markdown code fences etc."""
        # Remove markdown code fences
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None
