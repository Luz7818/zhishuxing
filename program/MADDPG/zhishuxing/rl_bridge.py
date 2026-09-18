from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from .navigation import NavigationAdapter, NavigationMap, Point
from .system import PassengerGroup
from .visualization import HubVisualizer

MADDPG_DIR = Path(__file__).resolve().parent.parent
if str(MADDPG_DIR) not in sys.path:
    sys.path.insert(0, str(MADDPG_DIR))

ENV_NAME = "integrated_hub_transfer"
CHECKPOINT_PATTERN = re.compile(
    r"^(?P<algorithm>\w+)_actor_number_(?P<number>\d+)_step_(?P<step>\d+)k_agent_(?P<agent_id>\d+)\.pth$"
)
# 与 UnityTemplate/HubTransferAgent.cs 的观测语义对齐：目标相对位置2维 + 自身速度2维 + K个邻近行人相对位置2*K维
NEARBY_K = 3
OBS_DIM = 4 + 2 * NEARBY_K
HEADINGS: Tuple[Point, Point, Point, Point] = ((1, 0), (0, 1), (-1, 0), (0, -1))
OBS_SCALE = 8.0


@dataclass
class CheckpointInfo:
    path: str
    algorithm: str
    number: int
    step_k: int
    agent_id: int


@dataclass
class _SimAgent:
    group: str
    index: int
    pos: Point
    heading: int
    route: List[Point]
    waypoint: int
    release_time: int
    moved: bool = False
    done_step: Optional[int] = None
    path: List[Point] = field(default_factory=list)


class MADDPGRuntime:
    """MADDPG 运行时桥接：扫描训练产物、加载 actor 权重、提供策略推理与内置多智能体引导仿真。"""

    def __init__(self, maddpg_dir: Path | str, data_dir: Path | str) -> None:
        self.maddpg_dir = Path(maddpg_dir)
        self.data_dir = Path(data_dir)
        self.model_dir = self.maddpg_dir / "model"
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
        found: List[CheckpointInfo] = []
        if not model_dir.exists():
            return found
        for path in sorted(model_dir.rglob("*.pth")):
            match = CHECKPOINT_PATTERN.match(path.name)
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
            "reward_files": [path.name for path in sorted(self.data_dir.glob("*_env_*.npy"))],
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

            from networks import Actor
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
        for path in sorted(self.data_dir.glob("*_env_*.npy")):
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
        output_path = Path(output_png) if output_png else self.data_dir / "rl_reward_curve_web.png"
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
        ax.set_title("MADDPG 评估奖励曲线（data_train）")
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

    # ------------------------------------------------------------------ 内置多智能体引导仿真

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
        nav_map: Optional[NavigationMap] = navigation.map
        if nav_map is None:
            raise RuntimeError("请先加载导航图。")
        if not groups:
            raise ValueError("缺少乘客群组。")

        max_steps = int(min(max(int(max_steps), 10), 1000))
        agents_per_group = int(min(max(int(agents_per_group), 1), 20))
        rng = np.random.default_rng(seed)

        agents: List[_SimAgent] = []
        routes: Dict[str, List[Point]] = {}
        for group in groups:
            route = navigation.plan_landmark_path(
                start=group.start, via=group.via_landmarks, goal=group.goal
            )
            routes[group.name] = route
            count = int(min(max(group.passengers, 1), agents_per_group))
            for index in range(count):
                spawn = self._spawn_position(navigation, group.start, rng)
                agents.append(
                    _SimAgent(
                        group=group.name,
                        index=index,
                        pos=spawn,
                        heading=int(rng.integers(0, 4)),
                        route=route,
                        waypoint=min(1, len(route) - 1),
                        release_time=group.release_time,
                        path=[spawn],
                    )
                )

        heat = np.zeros((nav_map.height, nav_map.width), dtype=float)
        collisions = 0
        step = 0
        for step in range(max_steps):
            active = [agent for agent in agents if agent.release_time <= step and agent.done_step is None]
            self._advance_waypoints(active, step)
            still_active = [agent for agent in active if agent.done_step is None]

            for agent in still_active:
                heat[agent.pos[1], agent.pos[0]] += 1

            if still_active:
                obs_batch = [self._build_obs(agent, still_active) for agent in still_active]
                actions = self.act(obs_batch)["actions"]
                for agent, action in zip(still_active, actions):
                    forward, turn = action[0], action[1]
                    moved = self._try_move(agent, forward, turn, navigation)
                    if not moved and forward >= 0.1:
                        collisions += 1
                    agent.moved = moved

            if all(agent.done_step is not None for agent in agents):
                break

        arrived = [agent for agent in agents if agent.done_step is not None]
        arrival_steps = [agent.done_step - agent.release_time for agent in arrived]
        baselines = [max(len(agent.route) - 1, 1) for agent in agents]

        group_summaries: List[Dict] = []
        for group in groups:
            members = [agent for agent in agents if agent.group == group.name]
            member_arrived = [agent for agent in members if agent.done_step is not None]
            member_steps = [agent.done_step - agent.release_time for agent in member_arrived]
            baseline = max(len(members[0].route) - 1, 1) if members else 0
            group_summaries.append(
                {
                    "group": group.name,
                    "agents": len(members),
                    "arrived": len(member_arrived),
                    "avg_transfer_steps": round(float(np.mean(member_steps)), 2) if member_steps else None,
                    "free_flow_baseline_steps": baseline,
                }
            )

        result = {
            "policy_source": self.policy_source,
            "policy_loaded": self.policy_loaded,
            "steps_executed": step + 1,
            "agents_total": len(agents),
            "agents_arrived": len(arrived),
            "avg_transfer_steps": round(float(np.mean(arrival_steps)), 2) if arrival_steps else None,
            "free_flow_baseline_steps": round(float(np.mean(baselines)), 2),
            "collision_events": collisions,
            "congestion_peak": int(heat.max()),
            "congestion_mean": round(float(heat.mean()), 4),
            "groups": group_summaries,
            "agents": [
                {
                    "group": agent.group,
                    "index": agent.index,
                    "release_time": agent.release_time,
                    "arrival_step": agent.done_step,
                    "path_length": len(agent.path),
                }
                for agent in agents
            ],
        }

        visualizer = HubVisualizer()
        trajectory_routes = {f"{agent.group}#{agent.index}": agent.path for agent in agents if len(agent.path) > 1}
        visualizer.render_snapshot(
            flow_grid=heat,
            blocked=nav_map.blocked,
            routes=trajectory_routes,
            output_file=output_png,
            title=f"{title}（策略: {self.policy_source}）",
        )
        result["image"] = str(output_png)
        return result

    def _spawn_position(self, navigation: NavigationAdapter, start: Point, rng: np.random.Generator) -> Point:
        nav_map = navigation.map
        assert nav_map is not None
        if nav_map.is_valid(start):
            candidates = [start]
            x, y = start
            for dx, dy in HEADINGS:
                candidate = (x + dx, y + dy)
                if nav_map.is_valid(candidate):
                    candidates.append(candidate)
            return candidates[int(rng.integers(0, len(candidates)))]
        return start

    def _advance_waypoints(self, active: List[_SimAgent], step: int) -> None:
        for agent in active:
            target = agent.route[min(agent.waypoint, len(agent.route) - 1)]
            if agent.pos == target:
                if agent.waypoint >= len(agent.route) - 1:
                    agent.done_step = step
                else:
                    agent.waypoint += 1

    def _build_obs(self, agent: _SimAgent, active: List[_SimAgent]) -> List[float]:
        px, py = agent.pos
        target = agent.route[min(agent.waypoint, len(agent.route) - 1)]
        dx = (target[0] - px) / OBS_SCALE
        dy = (target[1] - py) / OBS_SCALE
        heading_x, heading_y = HEADINGS[agent.heading]
        vx, vy = (float(heading_x), float(heading_y)) if agent.moved else (0.0, 0.0)

        others: List[Tuple[float, float]] = []
        for other in active:
            if other is agent:
                continue
            others.append(((other.pos[0] - px) / OBS_SCALE, (other.pos[1] - py) / OBS_SCALE))
        others.sort(key=lambda item: item[0] ** 2 + item[1] ** 2)

        obs = [dx, dy, vx, vy]
        for other_x, other_y in others[:NEARBY_K]:
            obs.extend([other_x, other_y])
        while len(obs) < OBS_DIM:
            obs.append(0.0)
        return obs[:OBS_DIM]

    def _try_move(self, agent: _SimAgent, forward: float, turn: float, navigation: NavigationAdapter) -> bool:
        nav_map = navigation.map
        assert nav_map is not None
        turn_steps = int(np.clip(round(float(turn)), -1, 1))
        agent.heading = (agent.heading + turn_steps) % 4
        if forward < 0.1:
            return False

        for heading in (agent.heading, (agent.heading + 1) % 4, (agent.heading + 3) % 4):
            delta_x, delta_y = HEADINGS[heading]
            candidate = (agent.pos[0] + delta_x, agent.pos[1] + delta_y)
            if nav_map.is_valid(candidate):
                agent.heading = heading
                agent.pos = candidate
                agent.path.append(candidate)
                return True
        return False
