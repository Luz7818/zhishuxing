from __future__ import annotations

from typing import Dict, List

import numpy as np

from .navigation import NavigationAdapter, NavigationMap, Point
from .scenarios import PassengerGroup


def plan_group_routes(navigation: NavigationAdapter, groups: List[PassengerGroup]) -> Dict[str, List[Point]]:
    """按组规划引导路径（含必经地标）。"""
    routes: Dict[str, List[Point]] = {}
    for group in groups:
        routes[group.name] = navigation.plan_landmark_path(
            start=group.start,
            via=group.via_landmarks,
            goal=group.goal,
        )
    return routes


def generate_dynamic_flow(
    nav_map: NavigationMap,
    groups: List[PassengerGroup],
    routes: Dict[str, List[Point]],
    steps: int = 60,
    seed: int = 42,
) -> np.ndarray:
    """按释放时间沿路径撒乘客并随机扩散，生成归一化动态客流网格。"""
    rng = np.random.default_rng(seed)
    flow = np.zeros((nav_map.height, nav_map.width), dtype=float)

    for step in range(steps):
        for group in groups:
            if step < group.release_time:
                continue
            route = routes.get(group.name) or []
            if not route:
                continue
            idx = min(step - group.release_time, len(route) - 1)
            x, y = route[idx]
            flow[y, x] += group.passengers * (0.9 + 0.25 * rng.random())

            for _ in range(2):
                nx = np.clip(x + int(rng.integers(-1, 2)), 0, nav_map.width - 1)
                ny = np.clip(y + int(rng.integers(-1, 2)), 0, nav_map.height - 1)
                flow[ny, nx] += group.passengers * 0.08 * rng.random()

    flow = flow / max(flow.max(), 1.0)
    return flow
