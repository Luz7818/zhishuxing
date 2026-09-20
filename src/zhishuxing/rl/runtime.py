"""MADDPG 运行时：扫描训练产物、加载 actor 权重、策略推理、奖励数据与引导仿真桥接。

供 Web API / CLI / PWA 调用；无权重或无 torch 时自动回退启发式并如实标注 policy_source。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import config as cfg
from ..analysis.plotting import ensure_parent
from ..core.navigation import NavigationAdapter
from ..core.scenarios import PassengerGroup
from ..core.simulation import run_guided_simulation as _run_guided_simulation

ENV_NAME = "integrated_hub_transfer"
CHECKPOINT_PATTERN = r"^(?P<algorithm>\w+)_actor_number_(?P<number>\d+)_step_(?P<step>\d+)k_agent_(?P<agent_id>\d+)\.pth$"


@dataclass
class CheckpointInfo:
    path: str
    algorithm: str
    number: int
    step_k: int
    agent_id: int


class MADDPGRuntime:
    def __init__(self, model_dir: Optional[Path] = None, data_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else cfg.paths.model_dir
        self.data_dir = Path(data_dir) if data_dir else cfg.paths.outputs
        self.checkpoints: List[CheckpointInfo] = []
        self.policy_loaded = False
        self.policy_source = "未加载"
        self.load_error = ""
        self.torch_version = self._probe_torch()

        self._actors: List[Any] = []
        self._obs_dims: List[int] = []
        self._action_dim = 2
        self._max_action = 1.0

        self.refresh()

    # ------------------------------------------------------------------ 状态

    @staticmethod
    def _probe_torch() -> str:
        try:
            import torch

            return torch.__version__
        except Exception:
            return ""

    def _scan_checkpoints(self, model_dir: Path) -> List[CheckpointInfo]:
        import re

        pattern = re.compile(CHECKPOINT_PATTERN)
        found: List[CheckpointInfo] = []
        if not model_dir.exists():
            return found
        for path in sorted(model_dir.rglob("*.pth")):
            match = pattern.match(path.name)
            if match:
                found.append(
                    CheckpointInfo(
                        path=str(path),
                        algorithm=match.group("algorithm"),
                        number=int(match.group("number")),
                        step_k=int(match.group("step")),
                        agent_id=int(match.group("agent_id")),
                    )
                )
        return found

    def refresh(self) -> Dict:
        self.checkpoints = self._scan_checkpoints(self.model_dir)
        return self.get_status()

    def _reward_files(self) -> List[Path]:
        """运行时输出优先，参考样本目录回退（按文件名去重）。"""
        seen = set()
        files: List[Path] = []
        for directory in (self.data_dir, cfg.paths.samples):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*_env_*.npy")):
                if path.name not in seen:
                    seen.add(path.name)
                    files.append(path)
        return files

    def get_status(self) -> Dict:
        agent_ids = sorted({item.agent_id for item in self.checkpoints})
        latest_step = max((item.step_k for item in self.checkpoints), default=0)
        return {
            "torch_available": bool(self.torch_version),
            "torch_version": self.torch_version,
            "model_dir": str(self.model_dir),
            "checkpoint_count": len(self.checkpoints),
            "checkpoints": [asdict(item) for item in self.checkpoints],
            "agent_ids": agent_ids,
            "latest_step_k": latest_step,
            "policy_loaded": self.policy_loaded,
            "policy_source": self.policy_source,
            "load_error": self.load_error,
            "reward_files": [path.name for path in self._reward_files()],
        }

    # ------------------------------------------------------------------ 策略加载与推理

    def load_policy(self, checkpoint_dir: Optional[str] = None) -> Dict:
        self.load_error = ""
        model_dir = Path(checkpoint_dir) if checkpoint_dir else self.model_dir
        checkpoints = self._scan_checkpoints(model_dir)
        # 让 get_status 反映本次实际加载来源
        self.checkpoints = checkpoints
        self.model_dir = model_dir

        if not checkpoints:
            self._actors = []
            self._obs_dims = []
            self.policy_loaded = False
            self.policy_source = "启发式回退"
            self.load_error = f"未找到 MADDPG actor 权重（{model_dir}/**/{ENV_NAME} 命名格式 .pth）"
            return self.get_status()

        try:
            import torch

            from .networks import Actor
        except Exception as exc:
            self._actors = []
            self._obs_dims = []
            self.policy_loaded = False
            self.policy_source = "启发式回退"
            self.load_error = f"加载 torch/Actor 失败: {exc}"
            return self.get_status()

        best: Dict[int, CheckpointInfo] = {}
        for item in checkpoints:
            if item.agent_id not in best or item.step_k > best[item.agent_id].step_k:
                best[item.agent_id] = item

        actors: List[Any] = []
        obs_dims: List[int] = []
        action_dim = 2
        errors: List[str] = []
        for agent_id in sorted(best):
            try:
                state = torch.load(best[agent_id].path, map_location="cpu")
                obs_dim = int(state["fc1.weight"].shape[1])
                hidden_dim = int(state["fc1.weight"].shape[0])
                action_dim = int(state["fc3.weight"].shape[0])
                args = SimpleNamespace(
                    obs_dim_n=[obs_dim],
                    action_dim_n=[action_dim],
                    hidden_dim=hidden_dim,
                    max_action=self._max_action,
                    use_orthogonal_init=False,
                )
                actor = Actor(args, 0)
                actor.load_state_dict(state)
                actor.eval()
                actors.append(actor)
                obs_dims.append(obs_dim)
            except Exception as exc:
                errors.append(f"agent_{agent_id}: {exc}")

        self._actors = actors
        self._obs_dims = obs_dims
        self._action_dim = action_dim
        self.policy_loaded = bool(actors)
        if not actors:
            self.policy_source = "启发式回退"
        elif len(actors) == len(best):
            self.policy_source = "MADDPG"
        else:
            self.policy_source = "MADDPG(部分)+启发式"
        self.load_error = "; ".join(errors)
        return self.get_status()

    def act(self, observations: Sequence[Sequence[float]]) -> Dict:
        obs_n = [[float(value) for value in obs] for obs in observations]
        actions: List[List[float]] = []
        policy_used: List[str] = []
        for index, obs in enumerate(obs_n):
            actor = self._actors[index] if index < len(self._actors) else None
            if actor is None:
                actions.append(self._heuristic_action(obs))
                policy_used.append("启发式")
            else:
                actions.append(self._net_action(actor, self._obs_dims[index], obs))
                policy_used.append("MADDPG")
        return {
            "actions": actions,
            "policy_used": policy_used,
            "policy_source": self.policy_source,
        }

    def _net_action(self, actor: Any, obs_dim: int, obs: List[float]) -> List[float]:
        import torch

        padded = obs[:obs_dim] + [0.0] * max(0, obs_dim - len(obs))
        tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(tensor).squeeze(0).cpu().numpy()
        return [float(value) for value in np.clip(action, -self._max_action, self._max_action)]

    @staticmethod
    def _heuristic_action(obs: List[float]) -> List[float]:
        import math

        goal_dx = obs[0] if len(obs) > 0 else 0.0
        goal_dy = obs[1] if len(obs) > 1 else 0.0
        vel_x = obs[2] if len(obs) > 2 else 0.0
        vel_y = obs[3] if len(obs) > 3 else 0.0
        if abs(goal_dx) < 1e-6 and abs(goal_dy) < 1e-6:
            return [0.0, 0.0]
        desired = math.atan2(goal_dy, goal_dx)
        current = math.atan2(vel_y, vel_x) if (abs(vel_x) + abs(vel_y)) > 1e-6 else desired
        diff = (desired - current + math.pi) % (2 * math.pi) - math.pi
        turn = float(np.clip(2.0 * diff / math.pi, -1.0, 1.0))
        return [0.9, turn]

    # ------------------------------------------------------------------ 奖励数据

    def reward_series(self, max_points: int = 200) -> Dict:
        series: List[Dict] = []
        for path in self._reward_files():
            try:
                values = np.load(path, allow_pickle=False).astype(float).ravel()
            except Exception:
                continue
            if values.size == 0:
                continue
            count = values.size
            indices = np.linspace(0, count - 1, min(count, max_points)).round().astype(int)
            series.append(
                {
                    "file": path.name,
                    "points": int(count),
                    "first": round(float(values[0]), 3),
                    "last": round(float(values[-1]), 3),
                    "max": round(float(values.max()), 3),
                    "mean": round(float(values.mean()), 3),
                    "x": indices.tolist(),
                    "y": [round(float(values[i]), 3) for i in indices],
                }
            )
        return {"data_dir": str(self.data_dir), "series": series}

    def render_reward_curve(self, output_png: Optional[str] = None) -> Dict:
        import matplotlib.pyplot as plt

        output_path = ensure_parent(output_png or (self.data_dir / "rl_reward_curve_web.png"))
        payload = self.reward_series()
        series = payload["series"]
        if not series:
            return {
                "image": "",
                "series_count": 0,
                "error": f"{self.data_dir} 中未找到 *_env_*.npy 奖励数据文件",
            }

        fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=120)
        for index, item in enumerate(series):
            ax.plot(
                item["x"],
                item["y"],
                linewidth=1.8,
                label=f"{item['file']}（末值 {item['last']}）",
            )
        ax.set_title("MADDPG 评估奖励曲线（data）")
        ax.set_xlabel("评估序号")
        ax.set_ylabel("平均评估奖励")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

        return {
            "image": str(output_path),
            "series_count": len(series),
            "series": [{k: v for k, v in item.items() if k not in ("x", "y")} for item in series],
        }

    # ------------------------------------------------------------------ 引导仿真（委托 core.simulation）

    def run_guided_simulation(
        self,
        navigation: NavigationAdapter,
        groups: Sequence[PassengerGroup],
        output_png: str,
        title: str = "智枢星 MADDPG 引导仿真",
        seed: int = 42,
        max_steps: int = 240,
        agents_per_group: int = 6,
    ) -> Dict:
        return _run_guided_simulation(
            navigation=navigation,
            groups=groups,
            output_png=output_png,
            title=title,
            policy=self,
            seed=seed,
            max_steps=max_steps,
            agents_per_group=agents_per_group,
        )
