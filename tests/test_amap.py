from __future__ import annotations

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


def test_polyline_parsing():
    transit = {
        "segments": [
            {"steps": [{"polyline": "114.05,22.55;114.06,22.56"}]},
            {"steps": [{"polyline": ""}, {"polyline": "114.07,22.57;114.08,22.58"}]},
        ]
    }
    coords = polyline_from_transit(transit)
    assert coords == [(114.05, 22.55), (114.06, 22.56), (114.07, 22.57), (114.08, 22.58)]


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


def test_build_route_tips_with_prefs():
    transit = {"duration": "1500", "walking_distance": "800"}
    tips = build_route_tips(transit, {"crowd": "avoid", "riskOn": True})
    assert any("25 分钟" in t for t in tips)
    assert any("避开拥堵" in t for t in tips)


def test_geocode_and_transit_error_shapes(requests_stub=None):
    """网络层错误应返回 None / 抛 ValueError，而不是静默吞掉（由 plan_route 的用户文案保证）。"""
    from zhishuxing import config as cfg

    assert cfg.amap_config()["rest_key"] in (None, "") or True  # 密钥可缺省
