from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .. import config as cfg
from .navigation import NavigationMap, Point


@dataclass
class PassengerGroup:
    """一组同起终点、同批次释放的乘客。"""

    name: str
    start: Point
    goal: Point
    via_landmarks: List[str]
    release_time: int
    passengers: int

    def to_payload(self) -> Dict:
        return {
            "name": self.name,
            "start": list(self.start),
            "goal": list(self.goal),
            "via_landmarks": list(self.via_landmarks),
            "release_time": self.release_time,
            "passengers": self.passengers,
        }


def load_scenarios(path: Optional[Path] = None) -> List[Dict]:
    """读取 configs/scenarios.json（地标引用尚未解析为坐标）。"""
    payload = cfg.load_json(Path(path) if path else cfg.paths.scenarios_config)
    return payload.get("groups", [])


def resolve_groups(payload: List[Dict], nav_map: Optional[NavigationMap] = None) -> List[PassengerGroup]:
    """把场景 payload（支持 landmark 引用或 [x,y] 坐标）解析为 PassengerGroup 列表。

    地标引用需要 nav_map 提供地标表；纯坐标场景可直接传入。
    支持 start/goal 或 start_landmark/goal_landmark 两种字段名。
    """
    groups: List[PassengerGroup] = []
    for item in payload:
        start = _resolve_point(item.get("start", item.get("start_landmark")), nav_map)
        goal = _resolve_point(item.get("goal", item.get("goal_landmark")), nav_map)
        groups.append(
            PassengerGroup(
                name=item["name"],
                start=start,
                goal=goal,
                via_landmarks=list(item.get("via_landmarks", [])),
                release_time=int(item.get("release_time", 0)),
                passengers=int(item.get("passengers", 10)),
            )
        )
    return groups


def _resolve_point(value, nav_map: Optional[NavigationMap]) -> Point:
    if isinstance(value, str):
        if nav_map is None or value not in nav_map.landmarks:
            raise KeyError(f"未找到地标: {value}")
        return nav_map.landmarks[value]
    return (int(value[0]), int(value[1]))
