from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import heapq
import json


Point = Tuple[int, int]

# 楼层/设施语义标签(cell_tags 的合法取值由数据决定,这里是展示用中文名)
TAG_LABELS: Dict[str, str] = {
    "stairs": "楼梯",
    "escalator": "扶梯",
    "elevator": "直梯",
    "crowd": "拥挤区",
}

LANDMARK_LABELS: Dict[str, str] = {
    "entry_a": "A入口",
    "entry_b": "B入口",
    "security": "主安检",
    "security_backup": "备用安检",
    "metro_gate": "地铁闸机",
    "rail_gate": "高铁闸机",
    "bus_gate": "公交站",
    "restroom_a": "卫生间A",
    "restroom_b": "卫生间B",
    "elevator_a": "无障碍直梯",
    "stairs_passage": "楼梯通道",
    "escalator_passage": "扶梯通道",
    "nursing_room": "母婴室",
}

# 中文说法 → 地标名(顺序敏感:具体词在前,避免"A口"被"安检"之类宽泛词抢先)
HUB_LANDMARK_ALIASES: List[Tuple[str, str]] = [
    ("A出口", "entry_a"),
    ("A口", "entry_a"),
    ("B出口", "entry_b"),
    ("B口", "entry_b"),
    ("备用安检", "security_backup"),
    ("主安检", "security"),
    ("安检", "security"),
    ("地铁", "metro_gate"),
    ("高铁", "rail_gate"),
    ("火车", "rail_gate"),
    ("公交", "bus_gate"),
    ("卫生间", "restroom_a"),
    ("洗手间", "restroom_a"),
    ("直梯", "elevator_a"),
]


@dataclass
class RouteCostSpec:
    """偏好 → 代价参数(由 llm.profile.PassengerProfile.to_cost_spec 生成)。

    - tag_penalties: 进入带对应 cell_tag 的格子时的附加代价;
    - forbidden_tags: 带这些标签的格子视为禁行(硬约束);
    - soft_via_landmarks: 软必经地标名(前缀匹配,如 "restroom" 匹配 restroom_a/b),
      顺路(代价增量不超过直达 × soft_via_tolerance)才纳入;
    - 空参数 = 与默认每格 1.0 的行为完全一致。
    """

    tag_penalties: Dict[str, float] = field(default_factory=dict)
    forbidden_tags: List[str] = field(default_factory=list)
    soft_via_landmarks: List[str] = field(default_factory=list)
    soft_via_tolerance: float = 1.3


@dataclass
class NavigationMap:
    width: int
    height: int
    blocked: List[Point] = field(default_factory=list)
    landmarks: Dict[str, Point] = field(default_factory=dict)
    # 设施语义层:tag -> 格子列表(如 stairs/escalator/elevator/crowd),blocked 之外的第二类"可走但有代价"语义
    cell_tags: Dict[str, List[Point]] = field(default_factory=dict)
    cell_size_m: float = 30.0

    def is_valid(self, node: Point) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height and node not in self._blocked_set

    @property
    def _blocked_set(self) -> set:
        return set(tuple(item) for item in self.blocked)

    def tag_cells(self, tag: str) -> set:
        return {tuple(item) for item in self.cell_tags.get(tag, [])}

    def landmarks_by_prefix(self, prefix: str) -> List[Tuple[str, Point]]:
        """按名字或前缀取地标(软必经点用:restroom 匹配 restroom_a/restroom_b)。"""
        return [
            (name, point)
            for name, point in self.landmarks.items()
            if name == prefix or name.startswith(f"{prefix}_") or name.startswith(prefix)
        ]


def landmark_label(name: str) -> str:
    return LANDMARK_LABELS.get(name, name)


def resolve_hub_landmark(text: str, nav_map: NavigationMap) -> Optional[Point]:
    """中文地标/设施说法 → 网格坐标;先精确匹配地标名,再按别名表。"""
    stripped = (text or "").strip()
    if not stripped or nav_map is None:
        return None
    for name, point in nav_map.landmarks.items():
        if name and name in stripped:
            return point
    for keyword, landmark in HUB_LANDMARK_ALIASES:
        if keyword in stripped and landmark in nav_map.landmarks:
            return nav_map.landmarks[landmark]
    return None


class NavigationAdapter:
    """JSON 路网加载 + A* 路径规划(4 邻接、曼哈顿启发式,支持偏好加权)。"""

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
            cell_tags={k: [tuple(p) for p in v] for k, v in payload.get("cell_tags", {}).items()},
            cell_size_m=float(payload.get("cell_size_m", 30.0)),
        )
        self.map = nav_map
        return nav_map

    # ------------------------------------------------------------ 对外规划接口

    def plan_path(self, start: Point, goal: Point, cost_spec: Optional[RouteCostSpec] = None) -> List[Point]:
        if self.map is None:
            raise RuntimeError("导航图未加载，请先调用 load_navigation。")
        return self._astar(start, goal, cost_spec)

    def plan_landmark_path(
        self,
        start: Point,
        via: List[str],
        goal: Point,
        cost_spec: Optional[RouteCostSpec] = None,
    ) -> List[Point]:
        """支持"必经安检"等业务路径：按途经地标分段规划后拼接。"""
        if self.map is None:
            raise RuntimeError("导航图未加载，请先调用 load_navigation。")

        route: List[Point] = []
        current = start
        for name in via:
            waypoint = self.map.landmarks.get(name)
            if waypoint is None:
                raise KeyError(f"未找到地标: {name}")
            segment = self._astar(current, waypoint, cost_spec)
            if route:
                route.extend(segment[1:])
            else:
                route.extend(segment)
            current = waypoint

        tail = self._astar(current, goal, cost_spec)
        if route:
            route.extend(tail[1:])
        else:
            route.extend(tail)
        return route

    def plan_with_preferences(
        self,
        start: Point,
        goal: Point,
        cost_spec: Optional[RouteCostSpec] = None,
    ) -> Dict[str, Any]:
        """偏好感知规划:加权 A* + 软必经点取舍 + 硬约束不可行时自动降级。

        返回 {route, length, cost, meters, degraded, tags_on_path, soft_via, notes}。
        """
        if self.map is None:
            raise RuntimeError("导航图未加载，请先调用 load_navigation。")
        spec = cost_spec or RouteCostSpec()
        notes: List[str] = []
        degraded = False

        try:
            route, cost = self._astar_with_cost(start, goal, spec)
        except RuntimeError:
            if not spec.forbidden_tags:
                raise
            degraded = True
            notes.append("硬约束下无可行路径,已自动放宽为重罚模式继续规划,请注意路线可能仍含受限设施")
            relaxed = RouteCostSpec(
                tag_penalties={**{tag: 8.0 for tag in spec.forbidden_tags}, **spec.tag_penalties},
                forbidden_tags=[],
                soft_via_landmarks=spec.soft_via_landmarks,
                soft_via_tolerance=spec.soft_via_tolerance,
            )
            route, cost = self._astar_with_cost(start, goal, relaxed)

        direct_cost = cost
        direct_length = len(route)
        soft_via_info: Dict[str, Any] = {}
        if spec.soft_via_landmarks:
            via_route, via_cost, via_point, via_name = self._best_soft_via_route(start, goal, spec)
            if via_route and via_cost <= direct_cost * spec.soft_via_tolerance:
                route, cost = via_route, via_cost
                extra_m = int(round((via_cost - direct_cost) * self.map.cell_size_m))
                soft_via_info = {
                    "landmark": via_name,
                    "point": list(via_point),
                    "accepted": True,
                    "extra_cost": round(via_cost - direct_cost, 2),
                }
                notes.append(
                    f"已按需求途经{landmark_label(via_name)}"
                    + (f"(多绕约 {extra_m} 米)" if extra_m > 0 else "(顺路)")
                )
            elif via_point is not None:
                soft_via_info = {
                    "landmark": via_name,
                    "point": list(via_point),
                    "accepted": False,
                    "extra_cost": round(via_cost - direct_cost, 2) if via_cost is not None else None,
                }
                notes.append(
                    f"顺路绕行到{landmark_label(via_name)}要多走不少路,本次未纳入路线;可到站后按指引前往"
                )

        route_points = [tuple(p) for p in route]
        tags_on_path: Dict[str, int] = {}
        for tag in self.map.cell_tags:
            cells = self.map.tag_cells(tag)
            count = sum(1 for p in route_points if p in cells)
            if count:
                tags_on_path[tag] = count

        return {
            "route": [list(p) for p in route],
            "length": len(route),
            "cost": round(cost, 2),
            "meters": int(round((len(route) - 1) * self.map.cell_size_m)),
            "degraded": degraded,
            "tags_on_path": tags_on_path,
            "soft_via": soft_via_info,
            "notes": notes,
            "direct_cost": round(direct_cost, 2),
            "direct_length": direct_length,
        }

    # ------------------------------------------------------------ A* 实现

    def _astar(self, start: Point, goal: Point, cost_spec: Optional[RouteCostSpec] = None) -> List[Point]:
        return self._astar_with_cost(start, goal, cost_spec)[0]

    def _astar_with_cost(
        self,
        start: Point,
        goal: Point,
        cost_spec: Optional[RouteCostSpec] = None,
    ) -> Tuple[List[Point], float]:
        if self.map is None:
            raise RuntimeError("导航图未加载。")
        if not self.map.is_valid(start):
            raise ValueError(f"非法起点: {start}")
        if not self.map.is_valid(goal):
            raise ValueError(f"非法终点: {goal}")

        penalties: Dict[str, float] = dict(cost_spec.tag_penalties) if cost_spec else {}
        forbidden: set = set()
        if cost_spec:
            for tag in cost_spec.forbidden_tags or []:
                forbidden |= self.map.tag_cells(tag)
        tag_sets = {tag: self.map.tag_cells(tag) for tag in penalties}

        def step_cost(node: Point) -> float:
            total = 1.0
            for tag, cells in tag_sets.items():
                if node in cells:
                    total += penalties[tag]
            return total

        open_heap: List[Tuple[float, Point]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Point, Optional[Point]] = {start: None}
        g_score: Dict[Point, float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct(came_from, current), g_score[current]

            for neighbor in self._neighbors(current):
                if neighbor in forbidden:
                    continue
                tentative = g_score[current] + step_cost(neighbor)
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score = tentative + self._manhattan(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        raise RuntimeError(f"无法规划路径: {start} -> {goal}")

    def _best_soft_via_route(
        self,
        start: Point,
        goal: Point,
        spec: RouteCostSpec,
    ) -> Tuple[Optional[List[Point]], Optional[float], Optional[Point], str]:
        """在软必经地标组中找代价最低的绕行方案(只试离起终点最近的几个候选)。"""
        assert self.map is not None
        best: Tuple[Optional[List[Point]], Optional[float], Optional[Point], str] = (None, None, None, "")
        for via_name in spec.soft_via_landmarks:
            candidates = self.map.landmarks_by_prefix(via_name)
            if not candidates:
                continue
            candidates.sort(key=lambda item: self._manhattan(start, item[1]) + self._manhattan(item[1], goal))
            for name, point in candidates[:3]:
                try:
                    head = self._astar_with_cost(start, point, spec)
                    tail = self._astar_with_cost(point, goal, spec)
                except (RuntimeError, ValueError):
                    continue
                total_cost = head[1] + tail[1]
                if best[1] is None or total_cost < best[1]:
                    route = head[0] + tail[0][1:]
                    best = (route, total_cost, point, name)
        return best

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
