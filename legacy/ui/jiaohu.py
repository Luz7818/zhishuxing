import json
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
import config_direct


st.set_page_config(
	page_title="“智枢星”—动态客流下的综合性交通枢纽智慧换乘引导系统",
	layout="wide",
	initial_sidebar_state="expanded",
)


ASSET_DIR = Path(__file__).resolve().parent


def asset_path(filename: str) -> str:
	return str(ASSET_DIR / filename)


def extract_od_locally(question: str) -> Dict[str, str]:
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
	model: str = "gpt-4o-mini",
	base_url: Optional[str] = None,
) -> Dict[str, str]:
	"""Use an OpenAI-compatible chat model to extract origin/destination text."""
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


def geocode(address: str, amap_key: str) -> Optional[Tuple[float, float, str]]:
	url = "https://restapi.amap.com/v3/geocode/geo"
	params = {"address": address, "key": amap_key}
	resp = requests.get(url, params=params, timeout=10)
	payload = resp.json()
	if payload.get("status") != "1" or not payload.get("geocodes"):
		return None
	geocode = payload["geocodes"][0]
	location = geocode.get("location", "")
	if "," not in location:
		return None
	lng_str, lat_str = location.split(",", 1)
	city = geocode.get("city", "")
	return float(lng_str), float(lat_str), city


def transit_route(
	origin: Tuple[float, float],
	destination: Tuple[float, float],
	amap_key: str,
	city: str = "",
) -> Optional[Dict]:
	url = "https://restapi.amap.com/v3/direction/transit/integrated"
	params = {
		"origin": f"{origin[0]},{origin[1]}",
		"destination": f"{destination[0]},{destination[1]}",
		"key": amap_key,
		"city": city,
		"strategy": 0,
		"nightflag": 0,
		"show_fields": "polyline",
	}
	resp = requests.get(url, params=params, timeout=15)
	payload = resp.json()
	if payload.get("status") != "1":
		return None
	routes = payload.get("route", {}).get("transits", [])
	if not routes:
		return None
	return routes[0]


def polyline_from_transit(transit: Dict) -> List[Tuple[float, float]]:
	coords: List[Tuple[float, float]] = []
	for segment in transit.get("segments", []):
		for step in segment.get("steps", []):
			polyline = step.get("polyline", "")
			pairs = polyline.split(";") if polyline else []
			for pair in pairs:
				if "," not in pair:
					continue
				lng_str, lat_str = pair.split(",", 1)
				coords.append((float(lng_str), float(lat_str)))
	return coords


def render_amap(coords: List[Tuple[float, float]], js_key: str, labels: Dict[str, str]) -> None:
	if not coords:
		st.warning("没有可用于在地图上渲染的折线。")
		return

	center = coords[len(coords) // 2]
	path_js = json.dumps([[lng, lat] for lng, lat in coords])
	html = f"""
	<!DOCTYPE html>
	<html>
	<head>
	  <meta charset=\"utf-8\" />
	  <style>
		html, body, #map {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
	  </style>
	  <script src=\"https://webapi.amap.com/maps?v=2.0&key={js_key}&plugin=AMap.ToolBar\"></script>
	</head>
	<body>
	  <div id=\"map\"></div>
	  <script>
		const map = new AMap.Map('map', {{
		  center: [{center[0]}, {center[1]}],
		  zoom: 12,
		}});
		map.addControl(new AMap.ToolBar());
		const path = {path_js};
		const polyline = new AMap.Polyline({{
		  path: path.map(p => new AMap.LngLat(p[0], p[1])),
		  strokeColor: '#3366FF',
		  strokeWeight: 5,
		  showDir: true,
		}});
		map.add(polyline);
		map.setFitView([polyline]);
		const markers = [
		  {{position: path[0], label: {json.dumps(labels.get('origin', 'O'))}}},
		  {{position: path[path.length - 1], label: {json.dumps(labels.get('destination', 'D'))}}},
		];
		markers.forEach(m => {{
		  const mk = new AMap.Marker({{
			position: new AMap.LngLat(m.position[0], m.position[1]),
			label: {{ content: m.label, direction: 'top' }},
		  }});
		  map.add(mk);
		}});
	  </script>
	</body>
	</html>
	"""
	st.components.v1.html(html, height=500)


def init_ui_state() -> None:
	if "user" not in st.session_state:
		st.session_state["user"] = {"account": ""}
	if "logged_in" not in st.session_state:
		st.session_state["logged_in"] = False
	if "route_history" not in st.session_state:
		st.session_state["route_history"] = []
	if "current_page" not in st.session_state:
		st.session_state["current_page"] = "首页"
	if "route_question" not in st.session_state:
		st.session_state["route_question"] = ""
	if "locate_status" not in st.session_state:
		st.session_state["locate_status"] = "未定位"
	if "last_plan" not in st.session_state:
		st.session_state["last_plan"] = "暂无"
	if "prefs" not in st.session_state:
		st.session_state["prefs"] = {
			"strategy": "balanced",
			"walk": "normal",
			"crowd": "normal",
			"pace": "normal",
			"note": "",
			"notifyOn": True,
			"riskOn": True,
			"defaultMode": "map",
			"darkMode": False,
		}

	if "plan_triggered" not in st.session_state:
		st.session_state["plan_triggered"] = False
	if "plan_data" not in st.session_state:
		st.session_state["plan_data"] = None


def preference_summary(prefs: Dict[str, str]) -> str:
	strategy_map = {
		"fast": "时间优先",
		"economy": "费用优先",
		"comfort": "舒适优先",
		"balanced": "综合均衡",
	}
	walk_map = {"short": "少步行", "normal": "步行普通", "long": "可多走"}
	crowd_map = {"avoid": "避开拥堵", "normal": "正常通过"}
	pace_map = {"relaxed": "宽松节奏", "normal": "适中节奏", "tight": "紧凑节奏"}

	s = strategy_map.get(prefs.get("strategy", "balanced"), "综合均衡")
	w = walk_map.get(prefs.get("walk", "normal"), "步行普通")
	c = crowd_map.get(prefs.get("crowd", "normal"), "正常通过")
	p = pace_map.get(prefs.get("pace", "normal"), "适中节奏")
	n = (prefs.get("note", "") or "").strip()
	note_part = f"；备注：{n}" if n else ""
	return f"当前偏好：{s}，{w}，{c}，{p}{note_part}。"


def save_history(route_text: str) -> None:
	history = st.session_state["route_history"]
	history.insert(
		0,
		{
			"text": route_text,
			"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		},
	)
	st.session_state["route_history"] = history[:20]


def run_planning(question: str, llm_key: str, llm_model: str, llm_base: str, amap_rest_key: str) -> bool:
	if not question.strip():
		st.error("问题必填。")
		return False

	if not llm_key:
		st.error("LLM API 密钥必填。")
		return False

	if not amap_rest_key:
		st.error("高德 REST 密钥必填。")
		return False

	with st.spinner("调用 LLM 提取 OD..."):
		od_data = call_llm_extract_od(question.strip(), api_key=llm_key, model=llm_model, base_url=llm_base or None)

	origin_text = od_data.get("origin_text", "").strip()
	dest_text = od_data.get("destination_text", "").strip()
	city = od_data.get("city", "").strip()

	if not origin_text or not dest_text:
		st.error("LLM 未返回起点和终点文本。")
		return False

	with st.spinner("通过高德地图对起点和终点进行地理编码..."):
		origin = geocode(origin_text, amap_rest_key)
		destination = geocode(dest_text, amap_rest_key)

	if not origin:
		st.error(f"地理编码起点失败：{origin_text}")
		return False
	if not destination:
		st.error(f"地理编码终点失败：{dest_text}")
		return False

	origin_lng, origin_lat, origin_city = origin
	dest_lng, dest_lat, dest_city = destination
	if not city and origin_city == dest_city:
		city = origin_city

	with st.spinner("从高德地图请求交通路线..."):
		transit = transit_route((origin_lng, origin_lat), (dest_lng, dest_lat), amap_rest_key, city=city)

	if not transit:
		st.error("高德地图未返回交通规划。")
		return False

	st.session_state["plan_data"] = {
		"origin_text": origin_text,
		"dest_text": dest_text,
		"city": city,
		"origin_lng": origin_lng,
		"origin_lat": origin_lat,
		"dest_lng": dest_lng,
		"dest_lat": dest_lat,
		"transit": transit,
	}
	st.session_state["last_plan"] = f"{origin_text}→{dest_text}"
	save_history(question.strip())
	st.toast("交通规划准备就绪。")
	return True


def main() -> None:
	init_ui_state()
	prefs = st.session_state["prefs"]

	if prefs.get("darkMode"):
		st.markdown(
			"""
			<style>
			.stApp { background: #0b1224; color: #f3f7ff; }
			</style>
			""",
			unsafe_allow_html=True,
		)

	st.markdown(
		"""
		<style>
		[data-testid="stHeader"] { display: none; }
		[data-testid="stToolbar"] { display: none; }
		.block-container { padding-top: 2.4rem; }
		[data-testid="stSidebar"] { min-width: 360px; }
		[data-testid="stSidebarContent"] { padding-top: 0.4rem; }
		.kiosk-sub { font-size: 17px; opacity: 0.85; margin-bottom: 12px; }
		.kiosk-card {
			padding: 16px 18px;
			border-radius: 14px;
			border: 1px solid rgba(80, 120, 180, 0.24);
			background: rgba(238, 244, 255, 0.45);
			margin-bottom: 12px;
		}
		.kiosk-route { font-size: 24px; font-weight: 700; }
		.kiosk-tip { font-size: 16px; line-height: 1.7; }
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.markdown(
		"""
		<div style="margin: 0 0 6px 0; padding: 2px 0 0 2px; color: var(--ink);">
		  <div style="font-size: 36px; font-weight: 900; line-height: 1.02; letter-spacing: 1px;">智枢星</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	st.markdown('<div class="kiosk-sub">你的换乘不折腾</div>', unsafe_allow_html=True)

	if not st.session_state["logged_in"]:
		st.subheader("控制端登录")
		login_col1, login_col2 = st.columns(2)
		with login_col1:
			login_account = st.text_input("手机号 / 用户名", key="login_account_input")
		with login_col2:
			login_password = st.text_input("登录密码", type="password", key="login_password_input")
		if st.button("登录并进入", type="primary", use_container_width=True):
			if not login_account or not login_password:
				st.error("请输入账号和密码")
			elif len(login_password) < 6:
				st.error("密码至少 6 位")
			else:
				st.session_state["logged_in"] = True
				st.session_state["user"] = {"account": login_account}
				st.toast("登录成功")
				st.rerun()
		st.caption("演示账号：zhishuxing.demo  密码：123456")
		return

	with st.sidebar:
		st.markdown("### 控制侧栏")
		st.caption(f"登录账号：{st.session_state['user'].get('account') or '未登录'}")
		st.caption(f"定位状态：{st.session_state['locate_status']}")

		st.markdown("**快捷操作**")
		quick_col1, quick_col2 = st.columns(2)
		with quick_col1:
			if st.button("📍 定位", use_container_width=True):
				st.session_state["locate_status"] = "定位成功"
				if not st.session_state["route_question"]:
					st.session_state["route_question"] = "从当前位置到宝安机场"
				st.toast("定位成功（演示模式）")
		with quick_col2:
			if st.button("▶ 规划", type="primary", use_container_width=True):
				st.session_state["plan_data"] = None
				st.session_state["plan_triggered"] = True
				st.rerun()

		quick_col3, quick_col4 = st.columns(2)
		with quick_col3:
			st.download_button(
				"⬇ 导出历史",
				data=json.dumps(st.session_state["route_history"], ensure_ascii=False, indent=2),
				file_name=f"zhishuxing_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
				mime="application/json",
				use_container_width=True,
			)
		with quick_col4:
			if st.button("⟳ 清空", use_container_width=True):
				st.session_state["route_history"] = []
				st.session_state["plan_data"] = None
				st.session_state["last_plan"] = "暂无"
				st.toast("本地缓存已清理")
				st.rerun()

		if st.button("退出登录", use_container_width=True):
			st.session_state["logged_in"] = False
			st.session_state["user"] = {"account": ""}
			st.session_state["plan_triggered"] = False
			st.rerun()

		with st.expander("乘客交互", expanded=True):
			st.text_area(
				"乘客诉求",
				key="route_question",
				placeholder="例如：从深圳北站A20出站口到五号线。",
				height=110,
			)
			if st.button("加入规划队列", use_container_width=True):
				st.session_state["plan_data"] = None
				st.session_state["plan_triggered"] = True
				st.rerun()

		with st.expander("偏好策略", expanded=True):
			prefs["strategy"] = st.selectbox(
				"路线策略",
				options=["balanced", "fast", "economy", "comfort"],
				index=["balanced", "fast", "economy", "comfort"].index(prefs.get("strategy", "balanced")),
				format_func=lambda v: {"fast": "时间优先", "economy": "费用优先", "comfort": "舒适优先", "balanced": "综合均衡"}[v],
			)
			prefs["walk"] = st.select_slider(
				"步行容忍度",
				options=["short", "normal", "long"],
				value=prefs.get("walk", "normal"),
				format_func=lambda v: {"short": "少步行", "normal": "普通", "long": "可多走"}[v],
			)
			prefs["crowd"] = st.radio(
				"人流偏好",
				options=["avoid", "normal"],
				index=["avoid", "normal"].index(prefs.get("crowd", "normal")),
				format_func=lambda v: {"avoid": "尽量避开拥堵", "normal": "正常通过"}[v],
			)
			prefs["note"] = st.text_area("备注", value=prefs.get("note", ""), height=70)
			st.caption(preference_summary(prefs))

		with st.expander("历史与系统", expanded=False):
			history = st.session_state["route_history"]
			if history:
				choice = st.selectbox("历史路线", options=list(range(len(history))), format_func=lambda i: f"{history[i]['time']} | {history[i]['text']}")
				if st.button("加载该历史并规划", use_container_width=True):
					st.session_state["route_question"] = history[choice]["text"]
					st.session_state["plan_data"] = None
					st.session_state["plan_triggered"] = True
					st.rerun()
			else:
				st.caption("暂无历史记录")

			prefs["notifyOn"] = st.toggle("行程通知", value=bool(prefs.get("notifyOn", True)))
			prefs["riskOn"] = st.toggle("拥堵预警", value=bool(prefs.get("riskOn", True)))
			prefs["darkMode"] = st.toggle("深色模式", value=bool(prefs.get("darkMode", False)))
			prefs["defaultMode"] = st.selectbox(
				"默认导航模式",
				options=["map", "ar"],
				index=0 if prefs.get("defaultMode") != "ar" else 1,
				format_func=lambda v: "3D地图" if v == "map" else "AR实景",
			)

			history_json = json.dumps(st.session_state["route_history"], ensure_ascii=False, indent=2)
			st.download_button(
				"导出历史记录(JSON)",
				data=history_json,
				file_name=f"zhishuxing_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
				mime="application/json",
				use_container_width=True,
			)
			if st.button("清空历史与缓存", use_container_width=True):
				st.session_state["route_history"] = []
				st.session_state["plan_data"] = None
				st.session_state["last_plan"] = "暂无"
				st.toast("本地缓存已清理")
				st.rerun()

		with st.expander("服务配置", expanded=False):
			llm_key = st.text_input("LLM API 密钥", value=os.getenv("LLM_API_KEY", config_direct.DEEPSEEK_API_KEY), type="password")
			llm_base = st.text_input("LLM 基础 URL（可选）", value=os.getenv("LLM_BASE_URL", config_direct.DEEPSEEK_API_URL))
			llm_model = st.text_input("LLM 模型", value=os.getenv("LLM_MODEL", config_direct.DEEPSEEK_MODEL))
			amap_rest_key = st.text_input("高德 REST 密钥", value=os.getenv("AMAP_REST_KEY", config_direct.AMAP_REST_KEY), type="password")
			amap_js_key = st.text_input("高德 JS 密钥", value=os.getenv("AMAP_JS_KEY", config_direct.AMAP_JS_KEY), type="password")

	if st.session_state["plan_triggered"]:
		ok = run_planning(
			st.session_state["route_question"],
			llm_key=llm_key,
			llm_model=llm_model,
			llm_base=llm_base,
			amap_rest_key=amap_rest_key,
		)
		st.session_state["plan_triggered"] = False
		if ok:
			st.rerun()
	status_col1, status_col2, status_col3, status_col4 = st.columns(4)
	with status_col1:
		st.metric("定位状态", st.session_state["locate_status"])
	with status_col2:
		st.metric("最近规划", st.session_state["last_plan"])
	with status_col3:
		st.metric("历史记录", str(len(st.session_state["route_history"])))
	with status_col4:
		st.metric("通知状态", "开启" if prefs.get("notifyOn", True) else "关闭")

	plan_data = st.session_state["plan_data"]
	if not plan_data:
		st.markdown('<div class="kiosk-card"><div class="kiosk-route">请在左侧交互区输入乘客诉求并点击“开始规划”', unsafe_allow_html=True)
		return

	origin_text = plan_data["origin_text"]
	dest_text = plan_data["dest_text"]
	city = plan_data["city"]
	origin_lng = plan_data["origin_lng"]
	origin_lat = plan_data["origin_lat"]
	dest_lng = plan_data["dest_lng"]
	dest_lat = plan_data["dest_lat"]
	transit = plan_data["transit"]

	st.markdown(
		f'<div class="kiosk-card"><div class="kiosk-route">{origin_text} → {dest_text}</div><div class="kiosk-tip">城市：{city or "未指定"} ｜ 策略：{preference_summary(prefs)}</div></div>',
		unsafe_allow_html=True,
	)

	main_col1, main_col2 = st.columns([3, 2])
	with main_col1:
		nav_mode = "AR实景导航" if prefs.get("defaultMode") == "ar" else "3D地图导航"
		nav_mode = st.segmented_control("导航模式", options=["3D地图导航", "AR实景导航"], default=nav_mode)
		if nav_mode == "AR实景导航":
			st.image(asset_path("VR.png"), use_container_width=True)
		else:
			st.image(asset_path("地图.png"), use_container_width=True)

		coords = polyline_from_transit(transit)
		render_amap(
			coords,
			js_key=amap_js_key,
			labels={"origin": origin_text, "destination": dest_text},
		)

		with st.expander("高德网页嵌入", expanded=False):
			nav_params = {
				"type": "bus",
				"from[name]": origin_text,
				"to[name]": dest_text,
				"from[lnglat]": f"{origin_lng},{origin_lat}",
				"to[lnglat]": f"{dest_lng},{dest_lat}",
			}
			nav_query = urllib.parse.urlencode(nav_params, safe=",")
			nav_url = f"https://ditu.amap.com/dir?{nav_query}"
			st.components.v1.html(
				f'<iframe src="{nav_url}" style="width:100%;height:760px;border:0;" scrolling="yes"></iframe>',
				height=780,
			)

	with main_col2:
		st.subheader("路线详情")
		st.markdown(
			"""
			1. 高铁到站后，从22/23站台旁的扶梯下楼（优先选电梯，避开楼梯），直接走向铁路到达西8口（位于1F出入口南侧）出站。
			2. 下扶梯到地铁层，刷手机NFC/乘车码直接进站（无需排队购票，支付宝/微信领深圳地铁乘车码即可）。
			3. 进站后直走，避开罗森、一鸣真鲜奶吧等购物点；按“5号线大剧院往赤湾方向”指示牌乘车，坐十三站到前海湾站下车。
			4. 换乘11号线（机场线）往碧头方向，乘坐三站到达宝安机场站。
			5. 抵达宝安机场站后，出站按引导前往航站楼。
			"""
		)

		st.subheader("出行提醒")
		st.markdown(
			"""
			- 认准5号线与11号线，明确乘车方向，避免坐反。
			- 行李多可走直梯，高峰期优先轻装乘扶梯；赶时间勿购物，避免误车。
			- 错过本班地铁，10分钟后有下一班；若地铁延误，可前往深圳北站东广场乘机场大巴（不堵车约90分钟直达）。
			- 不确定路线时，可咨询现场红马甲志愿者，跟随指示牌箭头前行即可。
			"""
		)

		metric_col1, metric_col2, metric_col3 = st.columns(3)
		with metric_col1:
			if "cost" in transit:
				st.metric("预计费用", transit.get("cost"))
		with metric_col2:
			if "duration" in transit:
				st.metric("预计用时(秒)", transit.get("duration"))
		with metric_col3:
			st.metric("换乘次数", str(len(transit.get("segments", []))))

		share_text = f"智枢星换乘：{origin_text}→{dest_text}"
		if st.button("生成分享文本", use_container_width=True):
			st.code(share_text)

		with st.expander("调试日志", expanded=False):
			st.json(transit.get("segments", []))


if __name__ == "__main__":
	main()
