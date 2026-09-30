"""智能体动作协议:让对话助手能触发界面动作(类 MCP 工具调用的最小实现)。

- 动作注册表(白名单):模型只能调用这里登记的动作,服务端校验后才下发,
  前端按 type 执行(如 switch_tab 由看板切换视图);
- 触发有两条通道:
  1. 规则短路 —— "打开/切换 + 板块名" 这类明确指令不经 LLM 直接命中,
     离线确定性可演示(与全仓「LLM 只有增强、必须可降级」的口径一致);
  2. LLM 动作信封 —— 自然语言泛化请求由真实模型输出 JSON 信封
     {"say": "...", "action": {...}},解析并校验后下发;解析失败按普通回复处理。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

# 动作白名单:tab id 与看板 data-tab 一一对应(前端 switchTab 认的键)
TAB_ACTIONS: Dict[str, str] = {
    "overview": "概览",
    "plan": "路线规划",
    "navigation": "枢纽导航",
    "flow": "客流面板",
    "rl": "RL 智能体",
    "reports": "分析报告",
    "settings": "设置",
}

# 规则短路:明确指令正则(打开/切换/跳转/去 + 板块名)
_INTENT_RE = re.compile(r"(?:打开|切换|跳转|进入|去)(?:一下|到)?\s*(?P<name>[A-Za-z]{0,4}\s*[「「'\"]?[概览路线规划枢纽导航客流面板智能体分析报告设置]{2,6}[」」'\"]?)")


def match_tab_intent(message: str) -> Optional[Tuple[str, str]]:
    """规则短路:消息明确要求打开某板块时返回 (tab, 板块中文名);否则 None。"""
    text = (message or "").strip()
    if not text or len(text) > 40:
        return None
    m = _INTENT_RE.search(text)
    if not m:
        return None
    name = m.group("name")
    for tab, label in TAB_ACTIONS.items():
        if label in name or name in label:
            return tab, label
        if tab in ("rl",) and ("RL" in name.upper() or "智能体" in name):
            return tab, label
    return None


def action_envelope_instructions() -> str:
    """注入合成系统提示词的动作调用说明(工具清单 + JSON 信封格式)。"""
    tools = ";".join(f"{tab}={label}" for tab, label in TAB_ACTIONS.items())
    return (
        "你可以调用一个界面动作工具 switch_tab(打开看板板块),可用 tab: "
        f"{tools}。仅当用户要求打开/切换/查看某板块时调用:此时把整个回复输出为"
        'JSON 对象 {"say": "给用户看的一句话", "action": {"type": "switch_tab", "tab": "<tab id>"}};'
        "其余情况正常用自然语言回答,不要输出 JSON。"
    )


def parse_action_envelope(reply: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """从模型回复里解析动作信封。

    返回 (展示文本, 动作或 None)。信封必须满足:整体是 JSON 对象、含字符串 say、
    action.type == "switch_tab" 且 tab 在白名单内 —— 任一不满足就按普通文本原样返回,
    绝不因为模型输出非法 JSON 而丢掉回复。
    """
    text = (reply or "").strip()
    if not text.startswith("{"):
        return text, None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text, None
    if not isinstance(data, dict) or not isinstance(data.get("say"), str):
        return text, None
    action = data.get("action")
    if not isinstance(action, dict) or action.get("type") != "switch_tab":
        return str(data.get("say")), None
    tab = action.get("tab")
    if tab not in TAB_ACTIONS:
        return str(data.get("say")), None
    say = data["say"].strip() or f"已打开「{TAB_ACTIONS[tab]}」。"
    return say, {"type": "switch_tab", "tab": tab}
