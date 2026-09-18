from __future__ import annotations

import pytest

from zhishuxing.core.navigation import NavigationAdapter, NavigationMap


@pytest.fixture()
def simple_map():
    return NavigationMap(
        width=5,
        height=5,
        blocked=[(2, 2), (2, 3)],
        landmarks={"a": (0, 0), "b": (4, 4), "mid": (0, 4)},
    )


def test_astar_finds_shortest_path(simple_map):
    adapter = NavigationAdapter()
    adapter.map = simple_map
    path = adapter.plan_path((0, 0), (4, 4))
    # 曼哈顿最短路径长度 = 8 步（含起终点）
    assert len(path) == 8 + 1 - 1 + 1  # 9 个节点
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    # 全部节点合法
    assert all(simple_map.is_valid(p) for p in path)


def test_astar_routes_around_blocked(simple_map):
    adapter = NavigationAdapter()
    adapter.map = simple_map
    path = adapter.plan_path((1, 2), (3, 2))
    # (2,2) 被阻挡，路径必须绕行且不经过阻挡格
    assert (2, 2) not in path and (2, 3) not in path
    assert path[0] == (1, 2) and path[-1] == (3, 2)


def test_landmark_path_includes_via(simple_map):
    adapter = NavigationAdapter()
    adapter.map = simple_map
    path = adapter.plan_landmark_path(start=(4, 4), via=["mid"], goal=(0, 0))
    assert (0, 4) in path
    assert path[0] == (4, 4) and path[-1] == (0, 0)


def test_unknown_landmark_raises(simple_map):
    adapter = NavigationAdapter()
    adapter.map = simple_map
    with pytest.raises(KeyError):
        adapter.plan_landmark_path(start=(0, 0), via=["不存在"], goal=(4, 4))


def test_invalid_start_raises(simple_map):
    adapter = NavigationAdapter()
    adapter.map = simple_map
    with pytest.raises(ValueError):
        adapter.plan_path((9, 9), (4, 4))


def test_unreachable_raises():
    nav = NavigationMap(width=3, height=1, blocked=[(1, 0)])
    adapter = NavigationAdapter()
    adapter.map = nav
    with pytest.raises(RuntimeError):
        adapter.plan_path((0, 0), (2, 0))


def test_load_navigation_from_config():
    from zhishuxing import config as cfg

    adapter = NavigationAdapter()
    nav = adapter.load_navigation(str(cfg.paths.navigation_config))
    assert nav.width == 30 and nav.height == 16
    assert "entry_a" in nav.landmarks and "metro_gate" in nav.landmarks
