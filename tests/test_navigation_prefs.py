"""偏好感知规划测试:加权 A* / 设施硬约束 / 软必经点 / 旧 schema 兼容。"""

from __future__ import annotations

import json

import pytest

from zhishuxing import config as cfg
from zhishuxing.core.navigation import NavigationAdapter, RouteCostSpec
from zhishuxing.llm.profile import rule_parse_preferences


@pytest.fixture(scope="module")
def hub_nav() -> NavigationAdapter:
    adapter = NavigationAdapter()
    adapter.load_navigation(str(cfg.paths.navigation_config))
    return adapter


def test_default_plan_unchanged_without_preferences(hub_nav):
    plan = hub_nav.plan_with_preferences((1, 2), (28, 12), RouteCostSpec())
    assert plan["route"][0] == [1, 2] and plan["route"][-1] == [28, 12]
    assert plan["meters"] == (plan["length"] - 1) * 30
    assert plan["degraded"] is False


def test_prefer_elevator_routes_through_elevator(hub_nav):
    # A口 -> 高铁闸机:默认最短路走楼梯门洞;优先直梯应绕行直梯门洞
    default_plan = hub_nav.plan_with_preferences((1, 2), (28, 3), RouteCostSpec())
    assert default_plan["tags_on_path"].get("stairs", 0) >= 2

    spec = rule_parse_preferences("行李多优先直梯").to_cost_spec()
    plan = hub_nav.plan_with_preferences((1, 2), (28, 3), spec)
    assert plan["tags_on_path"].get("elevator", 0) >= 2
    assert plan["tags_on_path"].get("stairs", 0) == 0
    assert plan["meters"] > default_plan["meters"]  # 代价偏好换来更长的步行距离


def test_avoid_stairs_hard_constraint_forbids_stairs(hub_nav):
    spec = rule_parse_preferences("轮椅出行").to_cost_spec()
    plan = hub_nav.plan_with_preferences((1, 2), (28, 3), spec)
    assert plan["tags_on_path"].get("stairs", 0) == 0
    assert plan["degraded"] is False


def test_hard_constraint_degrades_when_no_path(hub_nav):
    # 把换乘通道三个门洞全部禁走:不可达时应降级为重罚模式而不是抛错
    spec = RouteCostSpec(
        tag_penalties={"stairs": 2.0},
        forbidden_tags=["stairs", "escalator", "elevator"],
    )
    plan = hub_nav.plan_with_preferences((1, 2), (28, 12), spec)
    assert plan["degraded"] is True
    assert any("放宽" in note for note in plan["notes"])


def test_need_restroom_soft_via_accepted_when_detour_small(hub_nav):
    spec = rule_parse_preferences("去趟卫生间").to_cost_spec()
    plan = hub_nav.plan_with_preferences((1, 2), (28, 3), spec)  # 卫生间A几乎顺路
    assert plan["soft_via"].get("accepted") is True
    assert plan["soft_via"].get("landmark", "").startswith("restroom")
    assert any("卫生间" in note for note in plan["notes"])


def test_need_restroom_soft_via_rejected_when_detour_large(tmp_path):
    # 合成地图:顺路直行 6 步, restroom 挂在角落,绕行需 8 步——
    # 容忍度 1.4 时接受途经,容忍度 1.0 时拒绝绕行
    payload = {
        "width": 7,
        "height": 3,
        "blocked": [],
        "landmarks": {"a": [0, 1], "b": [6, 1], "restroom_x": [6, 0]},
        "cell_tags": {},
        "cell_size_m": 10,
    }
    map_file = tmp_path / "mini.json"
    map_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    adapter = NavigationAdapter()
    adapter.load_navigation(str(map_file))

    spec = RouteCostSpec(soft_via_landmarks=["restroom"], soft_via_tolerance=1.4)
    accepted = adapter.plan_with_preferences((0, 1), (6, 1), spec)
    assert accepted["soft_via"].get("accepted") is True
    assert accepted["soft_via"]["landmark"] == "restroom_x"

    strict = RouteCostSpec(soft_via_landmarks=["restroom"], soft_via_tolerance=1.0)
    rejected = adapter.plan_with_preferences((0, 1), (6, 1), strict)
    assert rejected["soft_via"].get("accepted") is False
    assert rejected["length"] == 7  # 回退为直达


def test_weighted_astar_prefers_penalized_detour_on_synthetic_map(tmp_path):
    # 构造最小地图:两条等长通道,一条带 crowd 标签——避开拥挤必须绕开它
    payload = {
        "width": 7,
        "height": 3,
        "blocked": [],
        "landmarks": {"a": [0, 1], "b": [6, 1]},
        "cell_tags": {"crowd": [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]},
        "cell_size_m": 10,
    }
    map_file = tmp_path / "mini.json"
    map_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    adapter = NavigationAdapter()
    adapter.load_navigation(str(map_file))

    neutral = adapter.plan_with_preferences((0, 1), (6, 1), RouteCostSpec())
    assert neutral["length"] == 7

    spec = RouteCostSpec(tag_penalties={"crowd": 5.0})
    routed = adapter.plan_with_preferences((0, 1), (6, 1), spec)
    assert routed["tags_on_path"].get("crowd", 0) == 0  # 绕行上下两排
    assert routed["cost"] > neutral["cost"]


def test_legacy_schema_still_loads(tmp_path):
    # 旧版 JSON(无 cell_tags/cell_size_m)必须照常加载,行为不变
    payload = {"width": 5, "height": 5, "blocked": [[2, 2]], "landmarks": {"x": [0, 0], "y": [4, 4]}}
    map_file = tmp_path / "legacy.json"
    map_file.write_text(json.dumps(payload), encoding="utf-8")
    adapter = NavigationAdapter()
    nav = adapter.load_navigation(str(map_file))
    assert nav.cell_tags == {}
    assert nav.cell_size_m == 30.0
    assert adapter.plan_path((0, 0), (4, 4))[-1] == (4, 4)
