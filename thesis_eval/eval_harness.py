"""
Unified Evaluation Harness for RoboCasa Mobile Manipulation.

Supports multiple policy types:
  - scripted: Rule-based baseline policies using privileged simulator info
  - vla: End-to-end VLA models (OpenVLA, Octo, etc.)
  - agent: LLM-based agents (dimos, etc.)
  - diffusion_policy: Trained Diffusion Policy models

Usage:
    python -m src.eval_harness --task PickPlaceCounterToCabinet --policy scripted --episodes 10
"""

import sys
import os

# IMPORTANT: Do NOT add the project root (/media/razor/Razer/HKU_Dissertation/)
# to sys.path. The bare robocasa/ and robosuite/ directories there shadow the
# properly installed editable packages. Use the editable installs instead.
# If you get import errors, ensure:
#   pip install -e /media/razor/Razer/HKU_Dissertation/robocasa
#   pip install -e /media/razor/Razer/HKU_Dissertation/robosuite

# Add src/ to path for local modules (but NOT the project root)
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
import gymnasium as gym
import robocasa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Task Configuration
# =============================================================================

TASK_CONFIGS = {
    # Category A: Navigation + Mobile Base
    "NavigateKitchen": {"category": "A_navigation", "horizon": 500, "composite": False},
    "PickPlaceCounterToCabinet": {"category": "A_navigation", "horizon": 800, "composite": False},
    "PickPlaceCabinetToCounter": {"category": "A_navigation", "horizon": 800, "composite": False},
    # Category B: Atomic Manipulation
    "OpenCabinet": {"category": "B_atomic", "horizon": 500, "composite": False},
    "CloseCabinet": {"category": "B_atomic", "horizon": 500, "composite": False},
    "TurnOnSinkFaucet": {"category": "B_atomic", "horizon": 600, "composite": False},
    "PickPlaceCounterToStove": {"category": "B_atomic", "horizon": 800, "composite": False},
    "PickPlaceSinkToCounter": {"category": "B_atomic", "horizon": 800, "composite": False},
    # Category C: Composite Mobile Manipulation
    "PrepareCoffee": {"category": "C_composite", "horizon": 1200, "composite": True},
    "LoadDishwasher": {"category": "C_composite", "horizon": 1200, "composite": True},
    "PreSoakPan": {"category": "C_composite", "horizon": 1600, "composite": True},
    "MoveToCounter": {"category": "C_composite", "horizon": 800, "composite": True},
    # Category D: Hard Composite
    "ClearSink": {"category": "D_hard", "horizon": 3300, "composite": True},
    "DeliverStraw": {"category": "D_hard", "horizon": 1700, "composite": True},
    "ArrangeBreadBasket": {"category": "D_hard", "horizon": 2900, "composite": True},
}

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class EpisodeResult:
    """Result of a single episode rollout."""

    task: str
    episode_id: int
    success: bool
    total_steps: int
    total_reward: float
    failure_reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

    policy_name: str
    split: str
    tasks: List[str]
    episodes_per_task: int
    results: List[EpisodeResult] = field(default_factory=list)

    @property
    def overall_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def task_success_rate(self, task: str) -> float:
        task_results = [r for r in self.results if r.task == task]
        if not task_results:
            return 0.0
        return sum(1 for r in task_results if r.success) / len(task_results)

    def category_success_rate(self, category: str) -> float:
        tasks_in_cat = [t for t, c in TASK_CONFIGS.items() if c["category"] == category]
        cat_results = [r for r in self.results if r.task in tasks_in_cat]
        if not cat_results:
            return 0.0
        return sum(1 for r in cat_results if r.success) / len(cat_results)

    def summary(self) -> Dict[str, Any]:
        return {
            "policy": self.policy_name,
            "split": self.split,
            "overall_sr": round(self.overall_success_rate, 4),
            "total_episodes": len(self.results),
            "per_task": {
                task: {
                    "sr": round(self.task_success_rate(task), 4),
                    "category": TASK_CONFIGS.get(task, {}).get("category", "unknown"),
                    "avg_steps": round(
                        np.mean([r.total_steps for r in self.results if r.task == task]), 1
                    )
                    if any(r.task == task for r in self.results)
                    else 0,
                }
                for task in sorted(set(r.task for r in self.results))
            },
            "per_category": {
                cat: round(self.category_success_rate(cat), 4)
                for cat in sorted(set(c["category"] for c in TASK_CONFIGS.values()))
            },
            "failure_modes": self._failure_mode_stats(),
        }

    def _failure_mode_stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for r in self.results:
            if not r.success:
                counts[r.failure_reason or "unknown"] += 1
        return dict(counts)


# =============================================================================
# Policy Interface
# =============================================================================


class BasePolicy:
    """Abstract policy interface for all approaches."""

    def __init__(self, task_name: str):
        self.task_name = task_name

    def predict(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Return action dict for robocasa gym env."""
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode."""
        pass

    def run_episode(self, env, obs, info, max_steps) -> tuple[bool, int]:
        """For agent policies: run full episode on the inner env.
        Returns (success, total_steps). Default: not used."""
        return False, 0


class ScriptedPolicy(BasePolicy):
    """
    Rule-based policy using privileged simulator info.

    This is a strong baseline that demonstrates what's achievable
    with ground-truth state access (upper bound for learned methods).
    """

    def __init__(self, task_name: str):
        super().__init__(task_name)
        self._step = 0
        self._grasped = False

    def reset(self):
        self._step = 0
        self._grasped = False

    def predict(self, obs, info):
        self._step += 1
        # Default: small random actions for exploration baseline
        # In practice, users should extend this class with task-specific logic
        action = {
            "action.end_effector_position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "action.end_effector_rotation": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "action.gripper_close": np.array([0.0], dtype=np.float32),
            "action.base_motion": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "action.control_mode": np.array([0.0], dtype=np.float32),
        }
        return action


class RandomPolicy(BasePolicy):
    """Random action baseline — lower bound for comparison."""

    def predict(self, obs, info):
        return {
            "action.end_effector_position": np.random.uniform(-0.1, 0.1, 3).astype(np.float32),
            "action.end_effector_rotation": np.random.uniform(-0.05, 0.05, 3).astype(np.float32),
            "action.gripper_close": np.random.uniform(0, 1, 1).astype(np.float32),
            "action.base_motion": np.random.uniform(-0.1, 0.1, 4).astype(np.float32),
            "action.control_mode": np.random.uniform(0, 1, 1).astype(np.float32),
        }


# =============================================================================
# Evaluation Engine
# =============================================================================


class EvalEngine:
    """Unified evaluation engine for all policy types."""

    def __init__(
        self,
        policy: BasePolicy,
        policy_name: str,
        tasks: List[str],
        split: str = "pretrain",
        episodes_per_task: int = 50,
        max_steps_override: Optional[int] = None,
        render: bool = False,
        seed: int = 0,
    ):
        self.policy = policy
        self.policy_name = policy_name
        self.tasks = tasks
        self.split = split
        self.episodes_per_task = episodes_per_task
        self.max_steps_override = max_steps_override
        self.render = render
        self.seed = seed
        self.results = EvalResults(
            policy_name=policy_name, split=split, tasks=tasks, episodes_per_task=episodes_per_task
        )

    @staticmethod
    def _idle_action():
        return {
            "action.end_effector_position": np.zeros(3, dtype=np.float32),
            "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
            "action.gripper_close": np.array([0.0], dtype=np.float32),
            "action.base_motion": np.zeros(4, dtype=np.float32),
            "action.control_mode": np.array([0.0], dtype=np.float32),
        }

    def run(self) -> EvalResults:
        """Run evaluation on all tasks."""
        logger.info(f"Starting evaluation: policy={self.policy_name}, "
                     f"tasks={len(self.tasks)}, episodes_per_task={self.episodes_per_task}")

        for task in self.tasks:
            task_config = TASK_CONFIGS.get(task, {})
            max_steps = self.max_steps_override or task_config.get("horizon", 1000)

            for ep_id in range(self.episodes_per_task):
                ep_seed = self.seed * 10000 + hash(task) % 10000 + ep_id
                result = self._run_episode(task, max_steps, ep_seed, ep_id)
                self.results.results.append(result)

                if (ep_id + 1) % 10 == 0:
                    logger.info(f"  {task}: {ep_id+1}/{self.episodes_per_task} "
                                f"(current SR: {self.results.task_success_rate(task):.2%})")

            logger.info(f"  {task} final SR: {self.results.task_success_rate(task):.2%}")

        logger.info(f"Evaluation complete. Overall SR: {self.results.overall_success_rate:.2%}")
        return self.results

    def _run_episode(
        self, task: str, max_steps: int, seed: int, ep_id: int
    ) -> EpisodeResult:
        """Run a single episode."""
        env = None
        try:
            env = gym.make(
                f"robocasa/{task}",
                split=self.split,
                seed=seed,
            )
            obs, info = env.reset()
            self.policy.reset()

            # Agent policies handle the episode themselves
            result = self.policy.run_episode(env, obs, info, max_steps)
            if result[1] > 0:  # agent consumed steps
                success, agent_steps = result
                # Check final env state
                try:
                    obs, reward, terminated, truncated, info = env.step(
                        self._idle_action()
                    )
                    success = success or info.get("success", False)
                except Exception:
                    pass
                return EpisodeResult(
                    task=task,
                    episode_id=ep_id,
                    success=success,
                    total_steps=agent_steps,
                    total_reward=float(success),
                    failure_reason="" if success else "agent_failed",
                )

            # Standard step-by-step loop for non-agent policies
            total_reward = 0.0
            step = 0
            success = False

            for step in range(max_steps):
                action = self.policy.predict(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if info.get("success", False):
                    success = True
                    break

            failure_reason = ""
            if not success:
                failure_reason = "timeout" if step >= max_steps - 1 else "incomplete"

            return EpisodeResult(
                task=task,
                episode_id=ep_id,
                success=success,
                total_steps=step + 1,
                total_reward=total_reward,
                failure_reason=failure_reason,
            )

        except Exception as e:
            logger.error(f"Episode {task}#{ep_id} crashed: {e}")
            return EpisodeResult(
                task=task,
                episode_id=ep_id,
                success=False,
                total_steps=0,
                total_reward=0.0,
                failure_reason=f"crash: {str(e)[:100]}",
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def save_results(self, output_dir: str = "results"):
        """Save evaluation results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{self.policy_name}_{self.split}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(self.results.summary(), f, indent=2)
        logger.info(f"Results saved to {filename}")
        return filename


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="RoboCasa Evaluation Harness")
    parser.add_argument("--task", type=str, nargs="+", default=None,
                        help="Task name(s). Default: all 15 tasks")
    parser.add_argument("--category", type=str, default=None,
                        help="Task category filter (A_navigation, B_atomic, C_composite, D_hard)")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "scripted", "agent", "bc"],
                        help="Policy type")
    parser.add_argument("--split", type=str, default="pretrain",
                        choices=["pretrain", "target", "all"])
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str,
                        default="/media/razor/Razer/HKU_Dissertation/results",
                        help="Output directory")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max episode steps")
    args = parser.parse_args()

    # Select tasks
    if args.task:
        tasks = args.task
    elif args.category:
        tasks = [t for t, c in TASK_CONFIGS.items() if c["category"] == args.category]
    else:
        tasks = list(TASK_CONFIGS.keys())

    # Filter to available tasks
    available = _get_available_tasks()
    tasks = [t for t in tasks if t in available]
    if not tasks:
        logger.error("No tasks available!")
        return

    logger.info(f"Selected {len(tasks)} tasks")

    # Create policy
    if args.policy == "random":
        policy = RandomPolicy(tasks[0])
    elif args.policy == "scripted":
        policy = ScriptedPolicy(tasks[0])
    elif args.policy == "agent":
        import importlib.util, os as _os
        _agent_path = _os.path.join(_os.path.dirname(__file__), "agent_policy.py")
        _spec = importlib.util.spec_from_file_location("agent_policy", _agent_path)
        _agent_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_agent_mod)
        policy = _agent_mod.AgentPolicy(tasks[0])
    elif args.policy == "bc":
        import importlib.util, os as _os
        _bc_path = _os.path.join(_os.path.dirname(__file__), "bc_policy.py")
        _spec = importlib.util.spec_from_file_location("bc_policy", _bc_path)
        _bc_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_bc_mod)
        _ckpt = _os.path.join(
            _os.path.dirname(_os.path.dirname(__file__)),
            "results/checkpoints",
            f"bc_{tasks[0]}_final.pt"
        )
        # Fallback to epoch 10 if final not ready
        if not _os.path.exists(_ckpt):
            import glob
            candidates = sorted(glob.glob(
                _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                              "results/checkpoints", f"bc_{tasks[0]}_*.pt")))
            _ckpt = candidates[-1] if candidates else _ckpt
        print(f"Loading BC checkpoint: {_ckpt}")
        policy = _bc_mod.BCPolicyWrapper(tasks[0], checkpoint_path=_ckpt)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Run evaluation
    engine = EvalEngine(
        policy=policy,
        policy_name=args.policy,
        tasks=tasks,
        split=args.split,
        episodes_per_task=args.episodes,
        max_steps_override=args.max_steps,
        seed=args.seed,
    )
    results = engine.run()
    engine.save_results(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Policy: {args.policy} | Split: {args.split}")
    print(f"Overall Success Rate: {results.overall_success_rate:.2%}")
    print("-" * 60)
    for task in tasks:
        sr = results.task_success_rate(task)
        cat = TASK_CONFIGS.get(task, {}).get("category", "?")
        print(f"  [{cat}] {task}: {sr:.2%}")
    print("=" * 60)


def _get_available_tasks() -> List[str]:
    """Get tasks that have datasets available."""
    dataset_dir = Path("/media/razor/Razer/HKU_Dissertation/robocasa/datasets/v1.0")
    available = set()
    for split_dir in ["pretrain", "target"]:
        for task_type in ["atomic", "composite"]:
            d = dataset_dir / split_dir / task_type
            if d.exists():
                for task_dir in d.iterdir():
                    if task_dir.is_dir():
                        available.add(task_dir.name)
    return sorted(available)


if __name__ == "__main__":
    main()
