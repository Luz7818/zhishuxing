"""智能体动作协议:动作注册表 / 规则短路 / 动作信封解析。"""

from zhishuxing.llm.actions import (
    TAB_ACTIONS,
    match_tab_intent,
    parse_action_envelope,
)


def test_match_tab_intent_direct_commands():
    assert match_tab_intent("打开导航") == ("navigation", "枢纽导航")
    assert match_tab_intent("切换到客流面板") == ("flow", "客流面板")
    assert match_tab_intent("打开 RL 智能体") == ("rl", "RL 智能体")
    assert match_tab_intent("去设置") == ("settings", "设置")
    assert match_tab_intent("打开分析报告看看") == ("reports", "分析报告")


def test_match_tab_intent_ignores_normal_requests():
    assert match_tab_intent("带老人行李多,优先直梯,去地铁前先上趟卫生间") is None
    assert match_tab_intent("从深圳北站到宝安机场,赶时间") is None
    assert match_tab_intent("") is None


def test_parse_action_envelope_valid():
    raw = '{"say": "好的,已为你打开", "action": {"type": "switch_tab", "tab": "flow"}}'
    say, action = parse_action_envelope(raw)
    assert say == "好的,已为你打开"
    assert action == {"type": "switch_tab", "tab": "flow"}


def test_parse_action_envelope_rejects_non_whitelisted_tab():
    raw = '{"say": "试图越权", "action": {"type": "switch_tab", "tab": "admin"}}'
    say, action = parse_action_envelope(raw)
    assert say == "试图越权"
    assert action is None                      # 白名单外动作不下发


def test_parse_action_envelope_plain_text_passthrough():
    say, action = parse_action_envelope("普通自然语言回答,不是 JSON。")
    assert say == "普通自然语言回答,不是 JSON。"
    assert action is None


def test_parse_action_envelope_malformed_json_is_not_lost():
    raw = '{"say": 123, "action": broken'
    say, action = parse_action_envelope(raw)
    assert say == raw                          # 非法信封按原文返回,回复不丢
    assert action is None


def test_tab_actions_match_frontend_tabs():
    assert set(TAB_ACTIONS) == {"overview", "plan", "navigation",
                                "flow", "rl", "reports", "settings"}
