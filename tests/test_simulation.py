from __future__ import annotations

import json
from pathlib import Path

import pytest

from zhishuxing.core.scenarios import load_scenarios, resolve_groups
from zhishuxing.core.simulation import HeuristicPolicy, run_guided_simulation


def test_heuristic_policy_action_semantics():
    policy = HeuristicPolicy()
    result = policy.act([[0.5, 0.0, 1.0, 0.0, 0.0] * 2])  # 目标在正前方
    actions = result["actions"]
    assert len(actions) == 1 and len(actions[0]) == 2
    forward, turn = actions[0]
    assert forward > 0.1          # 应前进
    assert abs(turn) <= 1.0       # 转向幅度受限于 [-1, 1]


def test_simulation_all_arrive_with_heuristic(navigation):
    payload = load_scenarios()
    groups = resolve_groups(payload, navigation.map)
    output = Path(__import__("zhishuxing").config.paths.outputs) / "test_simulation.png"
    result = run_guided_simulation(
        navigation=navigation,
        groups=groups,
        output_png=str(output),
        policy=HeuristicPolicy(),
        seed=42,
        max_steps=240,
        agents_per_group=3,
    )
    assert result["agents_total"] == 9
    assert result["agents_arrived"] == result["agents_total"]
    # 换乘步数应接近自由流基线（拥堵延误有限）
    assert result["avg_transfer_steps"] <= result["free_flow_baseline_steps"] + 15
    assert result["collision_events"] < result["agents_total"]
    assert Path(result["image"]).exists()


def test_simulation_metrics_fields(navigation):
    groups = resolve_groups(load_scenarios(), navigation.map)
    result = run_guided_simulation(
        navigation=navigation,
        groups=groups,
        output_png=str(Path(__import__("zhishuxing").config.paths.outputs) / "test_simulation2.png"),
        policy=HeuristicPolicy(),
        max_steps=100,
        agents_per_group=2,
    )
    for key in (
        "policy_source",
        "steps_executed",
        "agents_total",
        "agents_arrived",
        "avg_transfer_steps",
        "free_flow_baseline_steps",
        "congestion_peak",
        "congestion_mean",
        "groups",
        "image",
    ):
        assert key in result
    assert len(result["groups"]) == len(groups)


def test_simulation_requires_navigation():
    from zhishuxing.core.navigation import NavigationAdapter

    empty = NavigationAdapter()
    # 纯坐标群组（不依赖地标表），导航图未加载时应报错
    groups = resolve_groups(
        [
            {"name": "测试组", "start": [1, 2], "goal": [28, 12], "via_landmarks": [], "release_time": 0, "passengers": 3}
        ],
        None,
    )
    with pytest.raises(RuntimeError):
        run_guided_simulation(
            navigation=empty,
            groups=groups,
            output_png="/tmp/never.png",
        )


def test_scenario_payload_resolves_landmarks(navigation):
    groups = resolve_groups(load_scenarios(), navigation.map)
    assert len(groups) == 3
    names = {g.name for g in groups}
    assert names == {"A口进站->地铁", "B口进站->高铁", "A口进站->公交"}
    for group in groups:
        assert navigation.map.is_valid(group.start)
        assert navigation.map.is_valid(group.goal)
