import json
import os
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
import config_direct


st.set_page_config(page_title="“智枢星”—动态客流下的综合性交通枢纽智慧换乘引导系统", layout="wide")


def call_llm_extract_od(
	question: str,
	api_key: str,
	model: str = "gpt-4o-mini",
	base_url: Optional[str] = None,
) -> Dict[str, str]:
	"""Use an OpenAI-compatible chat model to extract origin/destination text."""
	try:
		from openai import OpenAI
	except Exception as exc:  # pragma: no cover
		raise RuntimeError("Install the openai package to use the LLM step") from exc

	client = OpenAI(api_key=api_key, base_url=base_url or None)
	system_prompt = (
		"从用户的问题中提取起点、终点和可选的城市。返回JSON格式，键为origin_text, destination_text, city（city可为空）。"
		"起点和终点应该是地点名称，如火车站、机场等。城市是地点所在的城市，如果未指定则为空。"
		"示例：问题“我需要从北京西站到上海虹桥站”，返回{\"origin_text\":\"北京西站\", \"destination_text\":\"上海虹桥站\", \"city\":\"\"}。"
		"另一个示例：问题“从上海火车站到浦东机场”，返回{\"origin_text\":\"上海火车站\", \"destination_text\":\"浦东机场\", \"city\":\"上海\"}。"
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


def main() -> None:
	st.markdown('<style>.stTextArea { margin-top: -20px !important; }</style>', unsafe_allow_html=True)
	st.markdown('<center><h1>“智枢星”—交通枢纽智慧换乘引导系统</h1></center>', unsafe_allow_html=True)
	st.markdown('<center><p>使用MADDPG算法+微调LLMs模型构建智能体，并使用高德地图进行AR导航，实现交通枢纽换乘规划。</p></center>', unsafe_allow_html=True)

	with st.sidebar:
		with st.expander("设置"):
			with st.expander("LLM 设置"):
				llm_key = st.text_input("LLM API 密钥", value=os.getenv("LLM_API_KEY", config_direct.DEEPSEEK_API_KEY), type="password")
				llm_base = st.text_input("LLM 基础 URL（可选）", value=os.getenv("LLM_BASE_URL", config_direct.DEEPSEEK_API_URL))
				llm_model = st.text_input("LLM 模型", value=os.getenv("LLM_MODEL", config_direct.DEEPSEEK_MODEL))
			
			with st.expander("高德地图设置"):
				amap_rest_key = st.text_input("高德 REST 密钥", value=os.getenv("AMAP_REST_KEY", config_direct.AMAP_REST_KEY), type="password")
				amap_js_key = st.text_input("高德 JS 密钥", value=os.getenv("AMAP_JS_KEY", config_direct.AMAP_JS_KEY), type="password")

		with st.expander("交互区", expanded=True):
			question = st.text_area("", placeholder="输入您的换乘请求\n例如：我需要从上海虹桥火车站乘坐地铁到浦东机场。")
			go = st.button("规划路线", type="primary")

	if "plan_triggered" not in st.session_state:
		st.session_state["plan_triggered"] = False
	if "plan_data" not in st.session_state:
		st.session_state["plan_data"] = None

	if go:
		st.session_state["plan_triggered"] = True
		st.session_state["plan_data"] = None

	if not st.session_state["plan_triggered"]:
		return

	if not question.strip():
		st.error("问题必填。")
		return

	if not llm_key:
		st.error("LLM API 密钥必填。")
		return

	if not amap_rest_key or not amap_js_key:
		st.error("高德 REST 密钥和高德 JS 密钥必填。")
		return

	if st.session_state["plan_data"] is None:
		with st.spinner("调用 LLM 提取 OD..."):
			od_data = call_llm_extract_od(question.strip(), api_key=llm_key, model=llm_model, base_url=llm_base or None)

		origin_text = od_data.get("origin_text", "").strip()
		dest_text = od_data.get("destination_text", "").strip()
		city = od_data.get("city", "").strip()

		if not origin_text or not dest_text:
			st.error("LLM 未返回起点和终点文本。")
			return

		with st.spinner("通过高德地图对起点和终点进行地理编码..."):
			origin = geocode(origin_text, amap_rest_key)
			destination = geocode(dest_text, amap_rest_key)

		if not origin:
			st.error(f"地理编码起点失败：{origin_text}")
			return
		if not destination:
			st.error(f"地理编码终点失败：{dest_text}")
			return

		origin_lng, origin_lat, origin_city = origin
		dest_lng, dest_lat, dest_city = destination
		if not city and origin_city == dest_city:
			city = origin_city

		with st.spinner("从高德地图请求交通路线..."):
			transit = transit_route((origin_lng, origin_lat), (dest_lng, dest_lat), amap_rest_key, city=city)

		if not transit:
			st.error("高德地图未返回交通规划。")
			return

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
		st.toast("交通规划准备就绪。")

	plan_data = st.session_state["plan_data"]
	origin_text = plan_data["origin_text"]
	dest_text = plan_data["dest_text"]
	city = plan_data["city"]
	origin_lng = plan_data["origin_lng"]
	origin_lat = plan_data["origin_lat"]
	dest_lng = plan_data["dest_lng"]
	dest_lat = plan_data["dest_lat"]
	transit = plan_data["transit"]

	col_nav, col_info = st.columns(2)
	with col_nav:
		nav_mode = st.radio(
			"导航模式",
			options=["3D地图导航", "AR实景导航"],
			horizontal=True,
		)
		if nav_mode == "AR实景导航":
			st.image("VR1.png", use_container_width=True)
		else:
			st.image("地图.png", use_container_width=True)
		st.write("**LLM 提取结果**")
		st.write(f"**起点：** {origin_text}")
		st.write(f"**终点：** {dest_text}")
		if city:
			st.write(f"**城市：** {city}")

	with col_info:
		st.write("--------------")
		st.write("### 路线规划")
		st.markdown(
			"""
			1. 高铁到站后，从22/23站台旁扶梯下楼，至铁路到达西8口（1F出入口南方向）出站。
			2. 下扶梯到地铁层，刷手机NFC/乘车码直接进站（无需排队购票，支付宝/微信领南京地铁乘车码即可）。
			3. 进站后直走，避开茶颜悦色、一鸣真鲜奶吧等购物点；按“S1江宁/空港新城江宁方向”指示牌乘车，往禄口机场站（倒数第二站，非终点站）。
			4. 地铁抵达禄口机场站后，出站直接上扶梯进入航站楼。
			"""
		)

		st.write("### 注意事项")
		st.markdown(
			"""
			- 认准S1号线，勿乘S3号线；明确乘车方向，避免坐反。
			- 行李多可走直梯，高峰期优先轻装乘扶梯；赶时间勿购物，避免误车。
			- 错过本班地铁，10分钟后有下一班；若地铁延误，可前往南京南站东广场乘机场大巴（不堵车约40分钟直达）。
			- 不确定路线时，可咨询现场红马甲志愿者，跟随指示牌箭头前行即可。
			"""
		)
		st.write("--------------")
		if "cost" in transit:
			st.metric("预计费用 (CNY)", transit.get("cost"))
		if "duration" in transit:
			st.metric("预计用时 (秒)", transit.get("duration"))

	with st.expander("高德网页嵌入", expanded=True):
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
			f'<iframe src="{nav_url}" style="width:100%;height:900px;border:0;" scrolling="yes"></iframe>',
			height=920,
		)

	coords = polyline_from_transit(transit)
	with st.expander("日志", expanded=False):
		st.write("**步骤 (原始)**")
		st.json(transit.get("segments", []))

	render_amap(
		coords,
		js_key=amap_js_key,
		labels={"origin": origin_text, "destination": dest_text},
	)


if __name__ == "__main__":
	main()
