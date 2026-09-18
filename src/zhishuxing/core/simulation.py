"""内置网格枢纽的多智能体引导仿真。

观测语义与 unity/HubTransferAgent.cs 对齐：
    [目标相对位置 dx,dy, 自身速度 vx,vy, K 个邻近行人相对位置 dx,dy]
动作语义（连续 2 维）：
    [forward: 前进/后退, turn: 转向]  —— turn 按整档 90° 量化。

策略通过 `PolicyPolicy` 协议注入（如 rl.runtime.MADDPGRuntime），无策略时使用
朝目标的启发式回退；仿真结果始终标注实际策略来源，不夸大模型效果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .navigation import NavigationAdapter, NavigationMap, Point
from .scenarios import PassengerGroup
from ..analysis.plotting import HubVisualizer

NEARBY_K = 3
OBS_DIM = 4 + 2 * NEARBY_K
OBS_SCALE = 8.0
HEADINGS: Tuple[Point, Point, Point, Point] = ((1, 0), (0, 1), (-1, 0), (0, -1))


class PolicyProtocol(Protocol):
    """仿真可调用的策略接口（MADDPGRuntime 或任何实现 act() 的对象）。"""

    def act(self, observations: Sequence[Sequence[float]]) -> Dict[str, Any]:
        ...


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


@dataclass
class HeuristicPolicy:
    """朝目标方向的启发式回退策略（输出动作语义与 MADDPG actor 一致）。"""

    name: str = "启发式"

    def act(self, observations: Sequence[Sequence[float]]) -> Dict[str, Any]:
        actions = [self._one(list(obs)) for obs in observations]
        return {"actions": actions, "policy_used": [self.name] * len(actions), "policy_source": self.name}

    @staticmethod
    def _one(obs: List[float]) -> List[float]:
        goal_dx = obs[0] if len(obs) > 0 else 0.0
        goal_dy = obs[1] if len(obs) > 1 else 0.0
        vel_x = obs[2] if len(obs) > 2 else 0.0
        vel_y = obs[3] if len(obs) > 3 else 0.0
        if abs(goal_dx) < 1e-6 and abs(goal_dy) < 1e-6:
            return [0.0, 0.0]
        import math

        desired = math.atan2(goal_dy, goal_dx)
        current = math.atan2(vel_y, vel_x) if (abs(vel_x) + abs(vel_y)) > 1e-6 else desired
        diff = (desired - current + math.pi) % (2 * math.pi) - math.pi
        turn = float(np.clip(2.0 * diff / math.pi, -1.0, 1.0))
        return [0.9, turn]


def policy_source_label(policy: Any) -> str:
    for attr in ("policy_source", "name"):
        label = getattr(policy, attr, None)
        if isinstance(label, str) and label:
            return label
    return "未标注"


def run_guided_simulation(
    navigation: NavigationAdapter,
    groups: Sequence[PassengerGroup],
    output_png: str,
    title: str = "智枢星 MADDPG 引导仿真",
    policy: Optional[PolicyProtocol] = None,
    seed: int = 42,
    max_steps: int = 240,
    agents_per_group: int = 6,
) -> Dict:
    """在导航图上运行多智能体引导仿真并渲染轨迹/拥堵快照。

    返回字段：policy_source、agents_total/arrived、avg_transfer_steps、
    free_flow_baseline_steps、collision_events、congestion_peak/mean、
    groups（分组统计）、agents（逐 agent 明细）、image。
    """
    nav_map: Optional[NavigationMap] = navigation.map
    if nav_map is None:
        raise RuntimeError("请先加载导航图。")
    if not groups:
        raise ValueError("缺少乘客群组。")

    policy = policy or HeuristicPolicy()
    max_steps = int(min(max(int(max_steps), 10), 1000))
    agents_per_group = int(min(max(int(agents_per_group), 1), 20))
    rng = np.random.default_rng(seed)

    agents: List[_SimAgent] = []
    for group in groups:
        route = navigation.plan_landmark_path(
            start=group.start, via=group.via_landmarks, goal=group.goal
        )
        count = int(min(max(group.passengers, 1), agents_per_group))
        for index in range(count):
            spawn = _spawn_position(nav_map, group.start, rng)
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
        _advance_waypoints(active, step)
        still_active = [agent for agent in active if agent.done_step is None]

        for agent in still_active:
            heat[agent.pos[1], agent.pos[0]] += 1

        if still_active:
            obs_batch = [_build_obs(agent, still_active) for agent in still_active]
            actions = policy.act(obs_batch)["actions"]
            for agent, action in zip(still_active, actions):
                forward, turn = action[0], action[1]
                moved = _try_move(agent, forward, turn, nav_map)
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
        "policy_source": policy_source_label(policy),
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
        title=f"{title}（策略: {result['policy_source']}）",
    )
    result["image"] = str(output_png)
    return result


def _spawn_position(nav_map: NavigationMap, start: Point, rng: np.random.Generator) -> Point:
    if nav_map.is_valid(start):
        candidates = [start]
        x, y = start
        for dx, dy in HEADINGS:
            candidate = (x + dx, y + dy)
            if nav_map.is_valid(candidate):
                candidates.append(candidate)
        return candidates[int(rng.integers(0, len(candidates)))]
    return start


def _advance_waypoints(active: List[_SimAgent], step: int) -> None:
    for agent in active:
        target = agent.route[min(agent.waypoint, len(agent.route) - 1)]
        if agent.pos == target:
            if agent.waypoint >= len(agent.route) - 1:
                agent.done_step = step
            else:
                agent.waypoint += 1


def _build_obs(agent: _SimAgent, active: List[_SimAgent]) -> List[float]:
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


def _try_move(agent: _SimAgent, forward: float, turn: float, nav_map: NavigationMap) -> bool:
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
