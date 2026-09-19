from __future__ import annotations

from zhishuxing import config as cfg
from zhishuxing.planning.amap import (
    build_route_details,
    build_route_tips,
    extract_od_locally,
    polyline_from_transit,
    resolve_strategy,
)


def test_extract_od_locally_from_to():
    result = extract_od_locally("从深圳北站A20出站口到五号线")
    assert result["origin_text"] == "深圳北站A20出站口"
    assert result["destination_text"] == "五号线"
    assert result["city"] == "深圳"


def test_extract_od_locally_fallback():
    result = extract_od_locally("随便说点什么")
    assert result["origin_text"] == "当前位置"
    assert result["destination_text"] == "目的地"


def test_polyline_parsing_flat_structure():
    """旧响应结构：segments[].steps[].polyline。"""
    transit = {
        "segments": [
            {"steps": [{"polyline": "114.05,22.55;114.06,22.56"}]},
            {"steps": [{"polyline": ""}, {"polyline": "114.07,22.57;114.08,22.58"}]},
        ]
    }
    coords = polyline_from_transit(transit)
    assert coords == [(114.05, 22.55), (114.06, 22.56), (114.07, 22.57), (114.08, 22.58)]


def test_polyline_parsing_nested_structure():
    """新响应结构：折线在 walking.steps / bus.buslines 子对象中（2026-09 实测）。"""
    transit = {
        "segments": [
            {
                "walking": {"steps": [{"polyline": "114.02,22.61;114.03,22.62"}]},
                "bus": {"buslines": [{"name": "地铁5号线", "polyline": "114.03,22.62;114.04,22.63"}]},
                "steps": [],
            }
        ]
    }
    coords = polyline_from_transit(transit)
    assert coords == [(114.02, 22.61), (114.03, 22.62), (114.03, 22.62), (114.04, 22.63)]


def test_resolve_strategy_applies_preferences():
    # 历史版本收集偏好但写死 strategy=0，现在偏好真正生效
    assert resolve_strategy({"strategy": "fast"}) == 0
    assert resolve_strategy({"strategy": "economy"}) == 1
    assert resolve_strategy({"strategy": "comfort"}) == 2
    assert resolve_strategy({"strategy": "balanced"}) == 0
    # 少步行优先于时间策略（映射到高德“最少步行”）
    assert resolve_strategy({"strategy": "fast", "walk": "short"}) == 3


def test_build_route_details_from_real_transit():
    transit = {
        "segments": [
            {"steps": [{"name": "地铁5号线"}, {"name": "步行"}]},
            {"steps": [{"name": "地铁11号线"}]},
        ]
    }
    details = build_route_details(transit)
    assert len(details) == 2
    assert "地铁5号线" in details[0]
    assert "地铁11号线" in details[1]


def test_build_route_details_nested_structure():
    """新结构：步行距离 + 公交线路名 + 上下车站点。"""
    transit = {
        "segments": [
            {
                "walking": {"distance": "773", "steps": [{"instruction": "步行"}]},
                "bus": {
                    "buslines": [
                        {
                            "name": "地铁5号线(环中线)(大剧院--赤湾)",
                            "departure_stop": {"name": "深圳北站"},
                            "arrival_stop": {"name": "前海湾"},
                        }
                    ]
                },
            }
        ]
    }
    details = build_route_details(transit)
    assert len(details) == 1
    assert "步行773米" in details[0]
    assert "地铁5号线(环中线)" in details[0]
    assert "深圳北站 → 前海湾" in details[0]


def test_build_route_tips_with_prefs():
    transit = {"duration": "1500", "walking_distance": "800"}
    tips = build_route_tips(transit, {"crowd": "avoid", "riskOn": True})
    assert any("25 分钟" in t for t in tips)
    assert any("避开拥堵" in t for t in tips)


def test_dotenv_loader(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 注释行\nTEST_DOTENV_KEY = abc123\nQUOTED = \"hello world\"\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    cfg._load_dotenv(env_file)

    import os

    assert os.environ["TEST_DOTENV_KEY"] == "abc123"
    assert os.environ["QUOTED"] == "hello world"

    monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
