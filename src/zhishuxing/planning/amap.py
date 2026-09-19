"""真实路线规划：LLM 提取 OD + 高德地理编码/公交换乘（逻辑迁自 UI/jiaohu.py）。

改进点：
- 偏好参数真正传入规划（历史版本收集了偏好但 strategy 写死为 0）；
- 路线详情/提醒从真实 transit 数据构建，不再是写死话术；
- 密钥一律走环境变量（config.amap_config / config.siliconflow_config）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from .. import config as cfg

# 高德公交换乘 strategy：0-推荐 1-最经济 2-最少换乘 3-最少步行 5-不乘地铁
STRATEGY_MAP = {
    "fast": 0,
    "economy": 1,
    "comfort": 2,
    "balanced": 0,
}

WALK_MAP = {
    "short": 3,   # 少步行 → 最少步行优先
    "normal": None,
    "long": None,
}


def extract_od_locally(question: str) -> Dict[str, str]:
    """正则兜底 OD 提取（LLM 不可用时）。"""
    cleaned = question.strip().replace("乘坐地铁", "")
    cleaned = cleaned.replace("乘坐公交", "")
    origin = ""
    destination = ""
    city = ""

    match = None
    for pattern in (
        r"从(.+?)到(.+?)(?:$|，|。|,)",
        r"(.+?)到(.+?)(?:$|，|。|,)",
    ):
        match = re.search(pattern, cleaned)
        if match:
            origin = match.group(1).strip()
            destination = match.group(2).strip()
            break

    if not origin or not destination:
        origin = "当前位置"
        destination = "目的地"

    if any(keyword in cleaned for keyword in ("深圳", "深圳北", "宝安机场", "前海湾", "五号线", "5号线", "11号线")):
        city = "深圳"

    return {
        "origin_text": origin,
        "destination_text": destination,
        "city": city,
    }


def call_llm_extract_od(
    question: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    """OpenAI 兼容聊天模型提取起点/终点/城市；失败回退本地正则。"""
    try:
        from openai import OpenAI
    except Exception:
        return extract_od_locally(question)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        system_prompt = (
            "从用户的问题中提取起点、终点和可选的城市。返回JSON格式，键为origin_text, destination_text, city（city可为空）。"
            "起点和终点应该是地点名称，如火车站、机场等。城市是地点所在的城市，如果未指定则为空。"
            "示例：问题“从深圳北站A20出站口到五号线”，返回{\"origin_text\":\"深圳北站A20出站口\", \"destination_text\":\"五号线\", \"city\":\"深圳\"}。"
            "另一个示例：问题“从深圳北站到宝安机场”，返回{\"origin_text\":\"深圳北站\", \"destination_text\":\"宝安机场\", \"city\":\"深圳\"}。"
        )
        completion = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        return {
            "origin_text": data.get("origin_text", ""),
            "destination_text": data.get("destination_text", ""),
            "city": data.get("city", ""),
        }
    except Exception:
        return extract_od_locally(question)


def geocode(address: str, amap_key: str, timeout: float = 10.0) -> Optional[Tuple[float, float, str]]:
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"address": address, "key": amap_key}
    resp = requests.get(url, params=params, timeout=timeout)
    payload = resp.json()
    if payload.get("status") != "1" or not payload.get("geocodes"):
        return None
    geocode0 = payload["geocodes"][0]
    location = geocode0.get("location", "")
    if "," not in location:
        return None
    lng_str, lat_str = location.split(",", 1)
    city = geocode0.get("city", "")
    return float(lng_str), float(lat_str), city


def transit_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    amap_key: str,
    city: str = "",
    strategy: int = 0,
    timeout: float = 15.0,
) -> Optional[Dict]:
    """高德公交换乘规划（integrated），返回首条方案。"""
    url = "https://restapi.amap.com/v3/direction/transit/integrated"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "key": amap_key,
        "city": city,
        "strategy": strategy,
        "nightflag": 0,
        "show_fields": "polyline",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    payload = resp.json()
    if payload.get("status") != "1":
        return None
    routes = payload.get("route", {}).get("transits", [])
    if not routes:
        return None
    return routes[0]


def _iter_segment_steps(segment: Dict) -> List[Dict]:
    """兼容新旧两种响应结构：扁平 steps[] 或 walking/bus/railway 子对象。"""
    steps: List[Dict] = list(segment.get("steps") or [])
    for container_key in ("walking", "bus", "railway", "taxi"):
        obj = segment.get(container_key)
        if isinstance(obj, dict):
            steps.extend(obj.get("steps") or [])
            steps.extend(obj.get("buslines") or [])
    return steps


def polyline_from_transit(transit: Dict) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for segment in transit.get("segments", []):
        for step in _iter_segment_steps(segment):
            polyline = step.get("polyline", "")
            pairs = polyline.split(";") if polyline else []
            for pair in pairs:
                if "," not in pair:
                    continue
                lng_str, lat_str = pair.split(",", 1)
                coords.append((float(lng_str), float(lat_str)))
    return coords


def resolve_strategy(prefs: Optional[Dict[str, Any]]) -> int:
    """把偏好映射为高德 strategy 参数（历史版本写死 0，现在真正生效）。"""
    prefs = prefs or {}
    walk_value = WALK_MAP.get(prefs.get("walk", "normal"))
    if walk_value is not None:
        return walk_value
    return STRATEGY_MAP.get(prefs.get("strategy", "balanced"), 0)


def build_route_details(transit: Dict) -> List[str]:
    """从真实 transit 数据构建路线详情（兼容新旧响应结构）。"""
    details: List[str] = []
    for index, segment in enumerate(transit.get("segments", []), start=1):
        parts: List[str] = []

        walking = segment.get("walking") or {}
        walking_steps = walking.get("steps") or []
        if walking_steps:
            distance = walking.get("distance")
            parts.append(f"步行{distance}米" if distance else "步行")

        for line in (segment.get("bus") or {}).get("buslines") or []:
            name = (line.get("name") or "").strip()
            departure = ((line.get("departure_stop") or {}).get("name") or "").strip()
            arrival = ((line.get("arrival_stop") or {}).get("name") or "").strip()
            entry = name + (f"（{departure} → {arrival}）" if departure and arrival else "")
            if entry and entry not in parts:
                parts.append(entry)

        railway = segment.get("railway") or {}
        if railway:
            parts.append((railway.get("name") or "城际铁路").strip())

        if segment.get("taxi"):
            parts.append("打车")

        if not parts:  # 旧扁平结构回退
            for step in _iter_segment_steps(segment):
                name = (step.get("name") or step.get("vehicle") or "").strip()
                if name and name not in parts:
                    parts.append(name)

        if parts:
            details.append(f"{index}. " + " → ".join(parts))

    if not details:
        details.append("未解析到分段详情，请查看地图折线。")
    return details


def build_route_tips(transit: Dict, prefs: Optional[Dict[str, Any]] = None) -> List[str]:
    """从真实数据与偏好生成出行提醒。"""
    tips: List[str] = []
    duration = transit.get("duration")
    walking = transit.get("walking_distance")
    if duration:
        try:
            minutes = int(float(duration) / 60)
            tips.append(f"预计全程约 {minutes} 分钟，请预留余量。")
        except (TypeError, ValueError):
            pass
    if walking:
        tips.append("本方案含步行路段，行李较多建议优先直梯。")
    if (prefs or {}).get("crowd") == "avoid":
        tips.append("已按“避开拥堵”偏好规划；高峰期请跟随站内引导分流。")
    if (prefs or {}).get("riskOn", True):
        tips.append("如遇地铁延误，可改用枢纽内其他接驳方式并咨询站内志愿者。")
    if not tips:
        tips.append("请跟随站内指示牌与引导标识前行。")
    return tips


def plan_route(
    question: str,
    prefs: Optional[Dict[str, Any]] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """完整真实规划链路：OD 提取 → 地理编码 → 换乘规划 → 详情/提醒构建。

    返回 JSON 友好结构；任一环节失败抛出 ValueError（消息面向最终用户）。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题必填。")

    amap = cfg.amap_config()
    amap_key = amap["rest_key"]
    if not amap_key:
        raise ValueError("未配置 AMAP_REST_KEY 环境变量，无法使用真实路线规划。")

    llm_conf = cfg.siliconflow_config()
    llm_key = llm_api_key or llm_conf["api_key"]
    if llm_key:
        od_data = call_llm_extract_od(
            question,
            api_key=llm_key,
            model=llm_model or llm_conf["model"],
            base_url=llm_base_url or llm_conf["base_url"],
        )
        od_source = "llm"
    else:
        od_data = extract_od_locally(question)
        od_source = "regex"

    origin_text = od_data.get("origin_text", "").strip()
    dest_text = od_data.get("destination_text", "").strip()
    city = od_data.get("city", "").strip()

    if not origin_text or not dest_text:
        raise ValueError("未能提取起点和终点文本。")

    origin = geocode(origin_text, amap_key, timeout=float(amap["timeout"]))
    destination = geocode(dest_text, amap_key, timeout=float(amap["timeout"]))
    if not origin:
        raise ValueError(f"地理编码起点失败：{origin_text}")
    if not destination:
        raise ValueError(f"地理编码终点失败：{dest_text}")

    origin_lng, origin_lat, origin_city = origin
    dest_lng, dest_lat, dest_city = destination
    if not city and origin_city == dest_city:
        city = origin_city

    transit = transit_route(
        (origin_lng, origin_lat),
        (dest_lng, dest_lat),
        amap_key,
        city=city,
        strategy=resolve_strategy(prefs),
        timeout=float(amap["timeout"]),
    )
    if not transit:
        raise ValueError("高德地图未返回交通规划。")

    coords = polyline_from_transit(transit)
    return {
        "engine": "amap",
        "od_source": od_source,
        "origin_text": origin_text,
        "destination_text": dest_text,
        "city": city,
        "origin": [origin_lng, origin_lat],
        "destination": [dest_lng, dest_lat],
        "strategy": resolve_strategy(prefs),
        "duration_sec": transit.get("duration"),
        "cost": transit.get("cost"),
        "segments": len(transit.get("segments", [])),
        "polyline": [[lng, lat] for lng, lat in coords],
        "details": build_route_details(transit),
        "tips": build_route_tips(transit, prefs),
    }
