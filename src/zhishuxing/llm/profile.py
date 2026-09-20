"""乘客需求档案:把自然语言需求解析为结构化换乘偏好。

PassengerProfile 是"懂乘客"的核心数据结构:
- priorities: 时间/距离/舒适/拥挤四维归一化优先级权重;
- hard: 硬约束(途经卫生间、少走楼梯),规划必须满足,不可行时由规划层自动降级并注明;
- soft: 软偏好(优先直梯、少步行、避开拥挤),规划通过代价加权尽量满足;
- persona: 乘客画像(行李/老人/儿童/轮椅/赶时间),按 PERSONA_IMPLICATIONS 派生隐含偏好。

解析级联:规则关键词(确定性、离线可用)优先;规则完全未命中且存在真实 LLM 适配器时,
再用 LLM JSON 模式解析口语表达。任何异常都不阻断主流程,最坏返回空档案(按均衡需求规划)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.navigation import RouteCostSpec

PRIORITY_KEYS = ("time", "distance", "comfort", "crowd")
HARD_KEYS = ("need_restroom", "avoid_stairs")
SOFT_KEYS = ("prefer_elevator", "least_walk", "avoid_crowd")
PERSONA_KEYS = ("luggage", "elderly", "child", "mobility", "rushed")

DEFAULT_PRIORITIES: Dict[str, float] = {
    "time": 0.25,
    "distance": 0.25,
    "comfort": 0.25,
    "crowd": 0.25,
}

# 画像 → 隐含偏好:老幼行李轮椅等画像自动派生规划约束与优先级
PERSONA_IMPLICATIONS: Dict[str, Dict[str, Any]] = {
    "luggage": {"soft": ["prefer_elevator"]},
    "elderly": {"soft": ["prefer_elevator", "avoid_crowd"]},
    "child": {"soft": ["prefer_elevator"]},
    "mobility": {"hard": ["avoid_stairs"], "soft": ["prefer_elevator"]},
    "rushed": {"priority": "time"},
}

KEY_LABELS: Dict[str, str] = {
    "need_restroom": "途经卫生间",
    "avoid_stairs": "少走楼梯",
    "prefer_elevator": "优先直梯",
    "least_walk": "少步行",
    "avoid_crowd": "避开拥挤",
    "luggage": "行李较多",
    "elderly": "携老人",
    "child": "携儿童",
    "mobility": "轮椅/行动不便",
    "rushed": "赶时间",
}

PRIORITY_LABELS: Dict[str, str] = {
    "time": "时间优先",
    "distance": "距离优先",
    "comfort": "舒适优先",
    "crowd": "少拥挤优先",
}

# (维度, 键, 触发词)。dimension: hard/soft/persona/priority;persona+rush 同时置时间优先。
_RULE_PATTERNS: List[tuple] = [
    ("hard", "need_restroom", ["卫生间", "洗手间", "厕所", "wc"]),
    ("hard", "avoid_stairs", ["少走楼梯", "不走楼梯", "避免楼梯", "别走楼梯", "少爬楼", "不爬楼", "爬楼梯"]),
    ("soft", "prefer_elevator", ["直梯", "电梯", "升降梯", "无障碍电梯"]),
    ("soft", "least_walk", ["少走路", "少步行", "步行短", "避免长通道", "近道", "别走太远"]),
    ("soft", "avoid_crowd", ["人少", "避开拥挤", "避开拥堵", "不挤", "避开人流", "别太挤"]),
    ("persona", "luggage", ["行李", "拉杆箱", "箱子多", "大包小包"]),
    ("persona", "elderly", ["老人", "父母", "爸妈", "长辈", "老年人", "奶奶", "爷爷"]),
    ("persona", "child", ["小孩", "孩子", "儿童", "宝宝"]),
    ("persona", "mobility", ["轮椅", "行动不便", "腿脚不便", "腿脚不好"]),
    ("persona+rush", "rushed", ["赶时间", "时间优先", "最快", "尽快", "着急", "时间短", "省时间", "越快越好"]),
    ("priority", "comfort", ["舒适", "轻松", "省力", "舒服"]),
]

_LLM_PROFILE_PROMPT = (
    "你是交通枢纽换乘助手的乘客需求解析器。从乘客的话中抽取个性化换乘需求,输出 JSON,"
    "schema:{\"priorities\":{\"time\":0-1,\"distance\":0-1,\"comfort\":0-1,\"crowd\":0-1},"
    "\"hard\":[\"need_restroom\"|\"avoid_stairs\"],"
    "\"soft\":[\"prefer_elevator\"|\"least_walk\"|\"avoid_crowd\"],"
    "\"persona\":[\"luggage\"|\"elderly\"|\"child\"|\"mobility\"|\"rushed\"]}。"
    "hard 是必须满足的约束(如要途经卫生间、不能走楼梯),soft 是尽量满足的偏好(如优先直梯、少步行、避开拥挤),"
    "persona 是乘客画像(行李多/老人/儿童/轮椅行动不便/赶时间)。"
    "只填用户明确表达的需求,没提到的不要编造;priorities 只在用户表达了明确优先级时给出最大项。"
    "示例:带老人行李多,优先直梯,转地铁前上趟卫生间 →"
    " {\"hard\":[\"need_restroom\"],\"soft\":[\"prefer_elevator\"],\"persona\":[\"elderly\",\"luggage\"]}"
)


@dataclass
class PassengerProfile:
    priorities: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIORITIES))
    hard: List[str] = field(default_factory=list)
    soft: List[str] = field(default_factory=list)
    persona: List[str] = field(default_factory=list)
    raw_text: str = ""
    source: str = "rule"  # rule | llm | merged
    has_priority_hint: bool = False

    # ------------------------------------------------------------ 编辑操作

    def set_priority(self, key: str, weight: float = 0.55) -> None:
        """置顶某个优先级维度,其余维度均分剩余权重。"""
        if key not in PRIORITY_KEYS:
            return
        self.has_priority_hint = True
        weight = min(max(weight, 0.3), 0.8)
        rest = round(1.0 - weight, 2)
        others = [k for k in PRIORITY_KEYS if k != key]
        self.priorities = {key: round(weight, 2), **{k: round(rest / len(others), 2) for k in others}}

    def normalize(self) -> "PassengerProfile":
        """去重 + 重新应用画像隐含偏好(合并/外部修改后调用)。"""
        self.hard = _unique(self.hard)
        self.soft = _unique(self.soft)
        self.persona = _unique(self.persona)
        _apply_persona_implications(self)
        self.hard = _unique(self.hard)
        self.soft = _unique(self.soft)
        self.persona = _unique(self.persona)
        return self

    def is_empty(self) -> bool:
        return not (self.hard or self.soft or self.persona or self.has_priority_hint)

    def merge(self, other: "PassengerProfile") -> "PassengerProfile":
        """多轮会话增量合并:新档案的优先级覆盖旧档案,约束/偏好/画像取并集。"""
        merged = PassengerProfile(
            priorities=dict(other.priorities if other.has_priority_hint else self.priorities),
            hard=_unique(self.hard + other.hard),
            soft=_unique(self.soft + other.soft),
            persona=_unique(self.persona + other.persona),
            raw_text="\n".join(t for t in (self.raw_text, other.raw_text) if t),
            source="merged",
            has_priority_hint=self.has_priority_hint or other.has_priority_hint,
        )
        return merged.normalize()

    # ------------------------------------------------------------ 输出

    def summary(self) -> str:
        chips = [KEY_LABELS.get(k, k) for k in (*self.hard, *self.soft, *self.persona)]
        if self.has_priority_hint:
            top = max(self.priorities, key=lambda k: self.priorities[k])
            chips.insert(0, PRIORITY_LABELS.get(top, top))
        ordered = list(dict.fromkeys(chips))
        return "、".join(ordered) if ordered else "无特殊需求"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "priorities": self.priorities,
            "hard": self.hard,
            "soft": self.soft,
            "persona": self.persona,
            "raw_text": self.raw_text,
            "source": self.source,
            "summary": self.summary(),
            "labels": {
                "hard": [{"key": k, "label": KEY_LABELS.get(k, k)} for k in self.hard],
                "soft": [{"key": k, "label": KEY_LABELS.get(k, k)} for k in self.soft],
                "persona": [{"key": k, "label": KEY_LABELS.get(k, k)} for k in self.persona],
            },
        }

    # ------------------------------------------------------------ 规划参数翻译

    def to_cost_spec(self) -> RouteCostSpec:
        """翻译成加权 A* 参数。空档案返回中性参数(与默认等价路径)。"""
        if self.is_empty():
            return RouteCostSpec()

        tag_penalties: Dict[str, float] = {}
        forbidden: List[str] = []
        weak_leg = bool({"elderly", "luggage", "child", "mobility"} & set(self.persona))

        if "avoid_stairs" in self.hard:
            forbidden.append("stairs")
        elif weak_leg or "prefer_elevator" in self.soft:
            # 楼梯重罚:对行李/老幼/轮椅人群,爬楼的费力程度远高于平地多走一段路
            tag_penalties["stairs"] = 8.0
        # 明确优先直梯时扶梯同样重罚(推行李/轮椅上扶梯不便),仅弱腿画像未明说时轻罚
        if "prefer_elevator" in self.soft:
            tag_penalties["escalator"] = 4.0
        elif weak_leg:
            tag_penalties["escalator"] = 1.5

        crowd_priority = float(self.priorities.get("crowd", 0.25))
        if "avoid_crowd" in self.soft or crowd_priority > 0.3:
            crowd_penalty = 1.0 + 2.0 * crowd_priority
            if "avoid_crowd" in self.soft:
                crowd_penalty = max(crowd_penalty, 2.5)
            tag_penalties["crowd"] = round(crowd_penalty, 2)

        tolerance = 1.3
        if "rushed" in self.persona or float(self.priorities.get("time", 0.25)) >= 0.5:
            tolerance = 1.15

        soft_via = ["restroom"] if "need_restroom" in self.hard else []
        return RouteCostSpec(
            tag_penalties=tag_penalties,
            forbidden_tags=forbidden,
            soft_via_landmarks=soft_via,
            soft_via_tolerance=tolerance,
        )


# ------------------------------------------------------------ 解析入口

def parse_preferences(text: str, adapter: Any = None) -> PassengerProfile:
    """自然语言 → 需求档案。规则优先;规则未命中且存在真实适配器时用 LLM 兜底。"""
    rule_profile = rule_parse_preferences(text)
    if not rule_profile.is_empty():
        return rule_profile
    llm_profile = llm_parse_preferences(text, adapter)
    if llm_profile is not None:
        return llm_profile
    return rule_profile


def rule_parse_preferences(text: str) -> PassengerProfile:
    """关键词规则解析:确定性、离线可用,覆盖演示与常见显式表达。"""
    profile = PassengerProfile(raw_text=text or "", source="rule")
    lowered = (text or "").lower()
    for dimension, key, words in _RULE_PATTERNS:
        if not any(word in lowered for word in words):
            continue
        if dimension == "hard":
            profile.hard.append(key)
        elif dimension == "soft":
            profile.soft.append(key)
        elif dimension == "persona":
            profile.persona.append(key)
        elif dimension == "persona+rush":
            profile.persona.append(key)
            if not profile.has_priority_hint:
                profile.set_priority("time")
        elif dimension == "priority" and not profile.has_priority_hint:
            profile.set_priority(key)
    return profile.normalize()


def llm_parse_preferences(text: str, adapter: Any) -> Optional[PassengerProfile]:
    """真实 LLM 解析口语表达;Mock/未加载/失败一律返回 None(调用方回退规则结果)。"""
    if adapter is None or getattr(adapter, "mock", False):
        return None
    try:
        raw = adapter.chat(
            [
                {"role": "system", "content": _LLM_PROFILE_PROMPT},
                {"role": "user", "content": text or ""},
            ],
            json_mode=True,
            temperature=0,
            max_tokens=256,
        )
        data = json.loads(_extract_json_object(raw))
        profile = PassengerProfile(raw_text=text or "", source="llm")

        priorities = data.get("priorities") or {}
        values = {}
        for key in PRIORITY_KEYS:
            if key in priorities:
                try:
                    value = float(priorities[key])
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    values[key] = value
        if values:
            top = max(values, key=lambda k: values[k])
            if values[top] >= 0.4:
                profile.set_priority(top, weight=values[top])

        profile.hard = [k for k in HARD_KEYS if k in (data.get("hard") or [])]
        profile.soft = [k for k in SOFT_KEYS if k in (data.get("soft") or [])]
        profile.persona = [k for k in PERSONA_KEYS if k in (data.get("persona") or [])]
        if profile.is_empty():
            return None
        return profile.normalize()
    except Exception:
        return None


def profile_from_api_prefs(
    prefs: Optional[Dict[str, Any]],
    adapter: Any = None,
    extra_text: str = "",
) -> PassengerProfile:
    """把 /api 请求的 prefs 结构化键 + 自由文本(note/消息)合并成需求档案。

    兼容历史前端:walk=short → 少步行,crowd=avoid → 避开拥挤,strategy=fast/comfort →
    对应优先级。自由文本中的显式表达优先于结构化键。
    """
    prefs = prefs or {}
    texts = [str(prefs.get("note") or ""), extra_text or ""]
    profile = parse_preferences("。".join(t for t in texts if t.strip()), adapter)

    if prefs.get("need_restroom"):
        profile.hard.append("need_restroom")
    if prefs.get("avoid_stairs"):
        profile.hard.append("avoid_stairs")
    if prefs.get("prefer_elevator"):
        profile.soft.append("prefer_elevator")
    if str(prefs.get("walk") or "") == "short":
        profile.soft.append("least_walk")
    if str(prefs.get("crowd") or "") == "avoid":
        profile.soft.append("avoid_crowd")
    strategy = str(prefs.get("strategy") or "")
    if strategy == "fast" and not profile.has_priority_hint:
        profile.set_priority("time")
    if strategy == "comfort" and not profile.has_priority_hint:
        profile.set_priority("comfort")
    return profile.normalize()


# ------------------------------------------------------------ 内部工具

def _apply_persona_implications(profile: PassengerProfile) -> None:
    for persona in profile.persona:
        implication = PERSONA_IMPLICATIONS.get(persona)
        if not implication:
            continue
        profile.hard.extend(implication.get("hard", []))
        profile.soft.extend(implication.get("soft", []))
        priority = implication.get("priority")
        if priority and not profile.has_priority_hint:
            profile.set_priority(priority)


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(item for item in items if item))


def _extract_json_object(raw: str) -> str:
    """从模型输出中抠出第一个 JSON 对象(容忍 ```json 包裹)。"""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
