"""乘客需求档案解析测试:规则解析 / LLM JSON 解析 / 合并 / 规划参数翻译。"""

from __future__ import annotations

import json

from zhishuxing.llm.profile import (
    PassengerProfile,
    parse_preferences,
    profile_from_api_prefs,
    rule_parse_preferences,
)


class _FakeRealAdapter:
    """伪装成真实适配器(mock=False),返回固定 JSON,用于测 LLM 解析分支。"""

    mock = False

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = []

    def chat(self, messages, *, json_mode=False, temperature=None, max_tokens=None):
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return self.payload


def test_rule_parse_full_needs():
    profile = rule_parse_preferences("带老人行李多,优先直梯,去地铁前先上趟卫生间")
    assert "need_restroom" in profile.hard
    assert "prefer_elevator" in profile.soft
    assert "elderly" in profile.persona and "luggage" in profile.persona
    # 画像隐含派生:老人/行李 → 优先直梯
    assert profile.summary().startswith("无") is False
    assert "优先直梯" in profile.summary()
    assert "途经卫生间" in profile.summary()


def test_rule_parse_rushed_sets_time_priority():
    profile = rule_parse_preferences("我赶时间,从A口到地铁")
    assert "rushed" in profile.persona
    assert profile.has_priority_hint
    assert profile.priorities["time"] >= 0.5


def test_rule_parse_wheelchair_implies_hard_no_stairs():
    profile = rule_parse_preferences("轮椅乘客怎么走")
    assert "avoid_stairs" in profile.hard  # mobility 隐含硬约束
    assert "prefer_elevator" in profile.soft


def test_rule_parse_empty_returns_empty_profile():
    profile = rule_parse_preferences("从深圳北站到宝安机场")
    assert profile.is_empty()


def test_parse_preferences_rules_win_without_llm_call():
    adapter = _FakeRealAdapter("not even called")
    profile = parse_preferences("优先直梯,途经卫生间", adapter)
    assert adapter.calls == []  # 规则命中即返回,不消耗 LLM 调用
    assert "prefer_elevator" in profile.soft


def test_llm_parse_fallback_for_colloquial_text():
    payload = json.dumps(
        {
            "priorities": {"time": 0.7},
            "hard": ["need_restroom"],
            "soft": ["prefer_elevator"],
            "persona": ["mobility"],
        },
        ensure_ascii=False,
    )
    adapter = _FakeRealAdapter(f"```json\n{payload}\n```")
    # 全口语表达:不含任何规则关键词,规则未命中才触发 LLM 解析
    profile = parse_preferences("我腿脚不太方便,想找个地方解决一下内急,麻烦照顾下", adapter)
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["json_mode"] is True
    assert "need_restroom" in profile.hard
    assert "mobility" in profile.persona
    assert profile.source == "llm"
    assert profile.priorities["time"] >= 0.5


def test_llm_parse_failure_falls_back_to_rule():
    class _BrokenAdapter:
        mock = False

        def chat(self, *args, **kwargs):
            raise RuntimeError("网络错误")

    profile = parse_preferences("帮我看看怎么走", _BrokenAdapter())
    assert profile.is_empty()


def test_merge_profiles_accumulates_across_turns():
    first = rule_parse_preferences("优先直梯,从A口出发")
    second = rule_parse_preferences("我现在有点赶时间")
    merged = first.merge(second)
    assert "prefer_elevator" in merged.soft
    assert "rushed" in merged.persona
    assert merged.has_priority_hint
    assert merged.source == "merged"


def test_to_cost_spec_neutral_for_empty_profile():
    spec = PassengerProfile().to_cost_spec()
    assert spec.tag_penalties == {}
    assert spec.forbidden_tags == []
    assert spec.soft_via_landmarks == []


def test_to_cost_spec_translates_preferences():
    spec = rule_parse_preferences("行李多优先直梯").to_cost_spec()
    assert spec.tag_penalties["stairs"] >= 6.0
    assert spec.tag_penalties["escalator"] >= 1.5
    assert spec.forbidden_tags == []

    spec_hard = rule_parse_preferences("轮椅出行,不走楼梯").to_cost_spec()
    assert "stairs" in spec_hard.forbidden_tags


def test_to_cost_spec_restroom_soft_via_and_rushed_tolerance():
    spec = rule_parse_preferences("去趟卫生间").to_cost_spec()
    assert spec.soft_via_landmarks == ["restroom"]
    assert spec.soft_via_tolerance > 1.2

    spec_rushed = rule_parse_preferences("要上卫生间,赶时间").to_cost_spec()
    assert spec_rushed.soft_via_tolerance < spec.soft_via_tolerance


def test_profile_from_api_prefs_back_compat():
    profile = profile_from_api_prefs(
        {"strategy": "fast", "walk": "short", "crowd": "avoid", "note": "优先直梯"}
    )
    assert "prefer_elevator" in profile.soft
    assert "least_walk" in profile.soft
    assert "avoid_crowd" in profile.soft
    assert profile.priorities["time"] >= 0.5

    # 旧版 PWA 只发结构化键,note 缺省也能解析
    legacy = profile_from_api_prefs({"strategy": "balanced", "walk": "normal", "crowd": "normal"})
    assert legacy.is_empty()
