from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import heapq
import json


Point = Tuple[int, int]


@dataclass
class NavigationMap:
    width: int
    height: int
    blocked: List[Point] = field(default_factory=list)
    landmarks: Dict[str, Point] = field(default_factory=dict)

    def is_valid(self, node: Point) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height and node not in self._blocked_set

    @property
    def _blocked_set(self) -> set:
        return set(tuple(item) for item in self.blocked)


class NavigationAdapter:
    def __init__(self) -> None:
        self.map: Optional[NavigationMap] = None

    def load_navigation(self, path: str) -> NavigationMap:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        nav_map = NavigationMap(
            width=int(payload["width"]),
            height=int(payload["height"]),
            blocked=[tuple(item) for item in payload.get("blocked", [])],
            landmarks={k: tuple(v) for k, v in payload.get("landmarks", {}).items()},
        )
        self.map = nav_map
        return nav_map

    def plan_path(self, start: Point, goal: Point) -> List[Point]:
        if self.map is None:
            raise RuntimeError("导航图未加载，请先调用 load_navigation。")
        return self._astar(start, goal)

    def plan_landmark_path(self, start: Point, via: List[str], goal: Point) -> List[Point]:
        if self.map is None:
            raise RuntimeError("导航图未加载，请先调用 load_navigation。")

        route: List[Point] = []
        current = start
        for name in via:
            waypoint = self.map.landmarks.get(name)
            if waypoint is None:
                raise KeyError(f"未找到地标: {name}")
            segment = self._astar(current, waypoint)
            if route:
                route.extend(segment[1:])
            else:
                route.extend(segment)
            current = waypoint

        tail = self._astar(current, goal)
        if route:
            route.extend(tail[1:])
        else:
            route.extend(tail)
        return route

    def _astar(self, start: Point, goal: Point) -> List[Point]:
        if self.map is None:
            raise RuntimeError("导航图未加载。")
        if not self.map.is_valid(start):
            raise ValueError(f"非法起点: {start}")
        if not self.map.is_valid(goal):
            raise ValueError(f"非法终点: {goal}")

        open_heap: List[Tuple[float, Point]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Point, Optional[Point]] = {start: None}
        g_score: Dict[Point, float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbor in self._neighbors(current):
                tentative = g_score[current] + 1.0
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score = tentative + self._manhattan(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        raise RuntimeError(f"无法规划路径: {start} -> {goal}")

    def _neighbors(self, node: Point) -> List[Point]:
        if self.map is None:
            return []
        x, y = node
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [p for p in candidates if self.map.is_valid(p)]

    @staticmethod
    def _manhattan(a: Point, b: Point) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct(came_from: Dict[Point, Optional[Point]], current: Point) -> List[Point]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
