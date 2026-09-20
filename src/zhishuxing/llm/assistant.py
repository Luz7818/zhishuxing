"""对话式换乘助手:需求档案 → 偏好规划 → 站内经验检索 → 个性化引导合成。

编排流程(Mock 与真实 LLM 均可完整运行):
1. parse_preferences 解析本轮消息(与会话档案增量合并);
2. 起终点命中枢纽地标 → 枢纽内偏好规划(加权 A* + 软必经点);否则走高德引擎
   (委托 service.plan_route,内部含 LLM/正则 OD 提取);
3. TransferKB 检索 top-k 站内换乘经验;
4. 真实 LLM 用 chat 合成个性化回答(为什么这样走符合需求);Mock 模式用确定性
   模板拼接,保证无密钥时离线演示闭环。

会话状态(历史/需求档案)保存在内存,进程重启即失效——演示项目不做持久化。
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ..core.navigation import NavigationMap, TAG_LABELS, landmark_label, resolve_hub_landmark
from ..planning.amap import extract_od_locally
from .kb import TransferKB
from .profile import PassengerProfile, profile_from_api_prefs

HISTORY_LIMIT = 12  # 会话保留的最近消息条数(一问一答各算一条)

SYNTHESIS_SYSTEM_PROMPT = (
    "你是大型综合交通枢纽的智慧换乘引导助手「智枢星」,真正懂乘客。"
    "根据乘客需求档案、已规划路线与站内换乘经验,用简明中文回答:"
    "1)先用一句话回应乘客的核心需求;"
    "2)分步说明怎么走,内容与给定路线详情一致,不要编造路线外设施;"
    "3)点明为什么这样走符合 TA 的需求(优先直梯/途经卫生间/少走路/避开拥挤等);"
    "4)引用站内经验时注明来源标题;"
    "5)没有可用路线时,说明缺什么信息并给出可行建议。"
)


class TransferAssistant:
    """无状态服务类:依赖 service 提供导航适配器、LLM 适配器与高德规划。"""

    def __init__(self, service: Any, kb: Optional[TransferKB] = None) -> None:
        self._service = service
        self._kb = kb or TransferKB.empty()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------ 对外接口

    def reset(self, session_id: str) -> Dict[str, Any]:
        self._sessions.pop(session_id or "", None)
        return {"session_id": session_id, "reset": True}

    def handle(
        self,
        message: str,
        session_id: Optional[str] = None,
        prefs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        message = (message or "").strip()
        if not message:
            raise ValueError("缺少 message")

        session_key = session_id or uuid.uuid4().hex[:12]
        session = self._sessions.setdefault(session_key, {"history": [], "profile": None})

        adapter = self._llm_adapter()
        profile = profile_from_api_prefs(prefs, adapter, extra_text=message)
        if session.get("profile") is not None:
            profile = session["profile"].merge(profile)
        session["profile"] = profile

        route_payload: Dict[str, Any] = {}
        engine: Optional[str] = None
        od = extract_od_locally(message)
        nav_map = self._nav_map()
        origin_point = resolve_hub_landmark(od.get("origin_text", ""), nav_map) if nav_map else None
        goal_point = resolve_hub_landmark(od.get("destination_text", ""), nav_map) if nav_map else None
        od_ready = od.get("origin_text") not in ("当前位置",) and od.get("destination_text") not in ("目的地",)

        if origin_point and goal_point:
            engine = "hub"
            route_payload = self._plan_hub(nav_map, profile, origin_point, goal_point, od)
        elif od_ready:
            engine = "amap"
            route_payload = self._plan_amap(message, prefs)
        else:
            # 起终点不完整(如"我有点赶时间"):本轮只更新需求档案,引导乘客补充 OD
            engine = None
            route_payload = {
                "incomplete_od": True,
                "origin_text": od.get("origin_text", ""),
                "destination_text": od.get("destination_text", ""),
            }

        kb_refs = self._search_kb(message)
        reply = self._synthesize(message, profile, route_payload, kb_refs, session)

        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "assistant", "content": reply})
        session["history"] = session["history"][-HISTORY_LIMIT:]

        payload: Dict[str, Any] = {
            "session_id": session_key,
            "reply": reply,
            "profile": profile.to_payload(),
            "engine": engine,
            "kb_refs": kb_refs,
        }
        if route_payload.get("error"):
            payload["route_error"] = route_payload["error"]
        elif route_payload.get("incomplete_od"):
            payload["od_incomplete"] = True
        elif route_payload:
            payload["route"] = route_payload
        return payload

    # ------------------------------------------------------------ 各环节实现

    def _llm_adapter(self) -> Any:
        system = getattr(self._service, "system", None)
        return getattr(system, "llm", None)

    def _nav_map(self) -> Optional[NavigationMap]:
        system = getattr(self._service, "system", None)
        return getattr(system, "nav_map", None)

    def _plan_hub(
        self,
        nav_map: NavigationMap,
        profile: PassengerProfile,
        origin_point,
        goal_point,
        od: Dict[str, str],
    ) -> Dict[str, Any]:
        plan = self._service.system.navigation.plan_with_preferences(
            origin_point, goal_point, profile.to_cost_spec()
        )
        details = [f"枢纽内步行约 {plan['meters']} 米({plan['length']} 格)"]
        details.extend(plan["notes"])
        facility_bits = [
            f"{TAG_LABELS.get(tag, tag)}×{count}"
            for tag, count in plan["tags_on_path"].items()
        ]
        if facility_bits:
            details.append("路径经过设施:" + "、".join(facility_bits))
        if plan["soft_via"].get("landmark"):
            status = "已纳入" if plan["soft_via"].get("accepted") else "未纳入"
            details.append(
                f"{landmark_label(plan['soft_via']['landmark'])}:{status}路线"
            )

        tips = ["请跟随站内引导标识与电子屏指引前往目标检票口。"]
        if plan.get("degraded"):
            tips.append("您的通行约束较严格,已自动放宽规划,请注意脚下设施类型。")
        return {
            "engine": "hub",
            "od_source": "landmark",
            "origin_text": od.get("origin_text", ""),
            "destination_text": od.get("destination_text", ""),
            "route": plan["route"],
            "length": plan["length"],
            "meters": plan["meters"],
            "details": details,
            "tips": tips,
            "tags_on_path": plan["tags_on_path"],
            "soft_via": plan["soft_via"],
        }

    def _plan_amap(self, message: str, prefs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return dict(self._service.plan_route(question=message, engine="amap", prefs=prefs or {}))
        except Exception as exc:  # 无高德 Key/网络失败时仍要给出可用的对话回答
            return {"error": f"实时路线获取失败({exc})"}

    def _search_kb(self, message: str) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        if len(self._kb) == 0:
            return refs
        try:
            for hit in self._kb.search(message, hub="shenzhen_north", top_k=3):
                refs.append(
                    {
                        "title": hit["doc"].get("title", ""),
                        "source": hit["doc"].get("source", ""),
                        "score": round(float(hit["score"]), 3),
                    }
                )
        except Exception:
            return []
        return refs

    def _synthesize(
        self,
        message: str,
        profile: PassengerProfile,
        route_payload: Dict[str, Any],
        kb_refs: List[Dict[str, Any]],
        session: Dict[str, Any],
    ) -> str:
        adapter = self._llm_adapter()
        if adapter is not None and not getattr(adapter, "mock", False):
            try:
                user_prompt = (
                    f"乘客需求档案:{profile.summary()}\n"
                    f"已规划路线:{self._format_route_for_prompt(route_payload)}\n"
                    f"站内经验参考:{self._format_kb_for_prompt(kb_refs)}\n"
                    f"最近对话:{self._format_history(session.get('history', []))}\n"
                    f"乘客最新消息:{message}"
                )
                reply = adapter.chat(
                    [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=600,
                )
                if reply and reply.strip():
                    return reply.strip()
            except Exception:
                pass  # LLM 失败降级模板,不中断对话
        return self._template_reply(profile, route_payload, kb_refs)

    def _format_route_for_prompt(self, route_payload: Dict[str, Any]) -> str:
        if not route_payload:
            return "无"
        if route_payload.get("incomplete_od"):
            return "乘客未给出完整起终点,请先确认需求并引导补充(如'从A口到地铁闸机')"
        if route_payload.get("error"):
            return f"获取失败:{route_payload['error']}"
        lines = [f"引擎={route_payload.get('engine', '')}"]
        lines.extend(str(item) for item in route_payload.get("details", [])[:8])
        tips = route_payload.get("tips") or []
        lines.extend(f"提示:{tip}" for tip in tips[:4])
        return " | ".join(lines)

    @staticmethod
    def _format_kb_for_prompt(kb_refs: List[Dict[str, Any]]) -> str:
        if not kb_refs:
            return "无"
        return " ;".join(f"《{ref['title']}》(来源:{ref['source']})" for ref in kb_refs)

    @staticmethod
    def _format_history(history: List[Dict[str, Any]]) -> str:
        if not history:
            return "无"
        parts = []
        for item in history[-6:]:
            role = "乘客" if item["role"] == "user" else "助手"
            content = str(item["content"])[:80]
            parts.append(f"{role}:{content}")
        return " / ".join(parts)

    def _template_reply(
        self,
        profile: PassengerProfile,
        route_payload: Dict[str, Any],
        kb_refs: List[Dict[str, Any]],
    ) -> str:
        """Mock/降级模板:确定性输出,完整演示"理解需求 → 路线 → 经验引用"闭环。"""
        lines: List[str] = []
        if profile.is_empty():
            lines.append("好的,我来帮您规划换乘。")
        else:
            lines.append(f"已理解您的需求:{profile.summary()}。")

        if route_payload.get("incomplete_od"):
            lines.append("还差一点信息:请告诉我从哪儿出发、要到哪儿(比如「从A口到地铁闸机」),我就能给出完整路线。")
        elif route_payload.get("error"):
            lines.append(f"暂时拿不到实时路线({route_payload['error']});建议先在站内引导屏确认目标方向。")
        elif route_payload:
            for detail in route_payload.get("details", [])[:6]:
                lines.append(str(detail))
            for tip in route_payload.get("tips", [])[:3]:
                lines.append(f"提示:{tip}")

        if kb_refs:
            titles = "、".join(f"《{ref['title']}》" for ref in kb_refs)
            lines.append(f"参考站内经验:{titles}。")
        lines.append("(当前为本地模板模式;配置 SILICONFLOW_API_KEY 后可获得更自然的对话回答。)")
        return "\n".join(lines)
