import json
import os
import requests
import streamlit as st
import config_direct
from typing import Dict, List, Optional, Tuple

# ==========================================
# 1. 页面基本配置与高级深隧科技感 CSS 注入 
# ==========================================
st.set_page_config(
    page_title="“智枢星”— 动态客流智慧换乘系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* 1.1 全局强行单版面与深钛蓝灰大背景 */
    html, body, [data-testid="stAppViewContainer"] {
        max-height: 100vh !important;
        overflow: hidden !important;
        background: linear-gradient(135deg, #05070f 0%, #0b1329 100%) !important;
        color: #ffffff !important;
        -webkit-font-smoothing: antialiased;
    }
    
    /* 消灭系统自带边距与空白 */
    [data-testid="stHeader"], #MainMenu, footer, header { visibility: hidden !important; height: 0 !important; }
    .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; max-height: 100vh !important; }
    
    /* 主副大标题样式 */
    .glow-title {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, #ffffff, #e0f2fe, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 0px !important;
        margin-bottom: 2px !important;
    }
    .sub-title {
        color: #38bdf8 !important;
        text-align: center;
        font-size: 1.1rem !important;
        letter-spacing: 4px;
        margin-bottom: 12px !important;
    }
    
    /* 全页面标题死锁纯白 */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #ffffff !important; 
        font-weight: 800 !important;
        opacity: 1 !important;
    }
    
    /* 解算中心大标题 */
    .detail-main-title {
        font-size: 1.9rem !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        margin-top: 5px !important;
        margin-bottom: 12px !important;
        letter-spacing: 1px;
    }

    /* 模块卡片小标题 */
    .card-title {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-bottom: 12px !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 卡片内部正文文本 */
    .card-body-text, .crypto-card p {
        font-size: 1.2rem !important;
        line-height: 1.7 !important;
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    /* 列表美化 */
    .strategy-list {
        margin: 0 !important;
        padding-left: 5px !important;
        list-style-type: none !important;
    }
    .strategy-list li {
        font-size: 1.25rem !important;
        line-height: 1.7 !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* 科技卡片容器 */
    .crypto-card {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 2px solid #38bdf8 !important;
        border-radius: 12px;
        padding: 14px 22px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.1);
    }
    
    /* 单选框高亮白字穿透死锁 */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label div,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] div[role="radiogroup"] p,
    div[data-testid="stRadio"] div[role="radiogroup"] span {
        color: #ffffff !important; 
        font-size: 1.4rem !important; 
        font-weight: 800 !important; 
        opacity: 1 !important; 
    }
    
    /* 输入框组件小标签 */
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #ffffff !important; 
        font-size: 1.5rem !important; 
        font-weight: 800 !important; 
        opacity: 1 !important;
    }
    
    /* 输入框样式 */
    div[data-testid="stTextInput"] input {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        border: 2px solid #38bdf8 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        border: 2px solid #38bdf8 !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="select"] div {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    
    /* 提示条 */
    .top-tips-bar {
        background: rgba(6, 182, 212, 0.25) !important;
        border: 2px solid #06b6d4 !important;
        border-radius: 8px;
        padding: 8px 18px !important;
        margin-bottom: 10px !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    
    /* 方案页基础卡片文本 */
    .scheme-header-title { font-size: 1.4rem !important; font-weight: 900 !important; color: #ffffff !important; }
    .scheme-meta-row { margin-top: 4px !important; display: flex; gap: 20px; font-size: 1.3rem !important; color: #ffffff !important; font-weight: 900 !important; }
    .scheme-meta-change { color: #f59e0b !important; font-size: 1rem !important; font-weight: 900; background: rgba(245, 158, 11, 0.1) !important; padding: 1px 6px; border-radius: 4px; border: 1px solid #f59e0b !important; }

    /* 按钮样式 */
    div.stButton > button { font-size: 1rem !important; font-weight: 900 !important; height: 38px !important; }
    div.stButton > button[kind="secondary"] { color: #000000 !important; background-color: #ffffff !important; border: 2px solid #ffffff !important; }
    div.stButton > button[kind="primary"] { background: linear-gradient(90deg, #ef4444, #dc2626) !important; color: #ffffff !important; border: none !important; }
    
    h1, h2, h3, h4, p, span, label, li { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 状态机初始化与公共标题
# ==========================================
if "page_stage" not in st.session_state: st.session_state["page_stage"] = "HOME"
if "search_params" not in st.session_state: st.session_state["search_params"] = {}
if "selected_scheme_idx" not in st.session_state: st.session_state["selected_scheme_idx"] = 0

st.markdown('<div class="glow-title">“智枢星”— 综合交通枢纽智慧换乘引导系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">DYNAMIC PASSENGER FLOW INTELLIGENT TRANSFER GUIDANCE SYSTEM</div>', unsafe_allow_html=True)

# ==========================================
# 【PAGE 1】系统主首页 
# ==========================================
if st.session_state["page_stage"] == "HOME":
    st.markdown("""
    <div class="crypto-card">
        <div class="card-title">🪐 平台功能概述</div>
        <p style="margin-bottom:0; font-size:1.15rem; line-height:1.6;">
            “智枢星”将为您提供智能化、一站式的综合交通枢纽出行指引。只需输入您的出发地与目的地，系统便能根据您的个人出行偏好（如时间最快、步行最少、携带大件行李等），实时为您规划并推荐最合适的路线方案，并配备清晰直观的AR实景流线与3D精细地图引导，助您轻松避开拥堵人群，享受无感、顺畅的智慧出行体验。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('### 📥 请根据您的实际情况选择')
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        origin_input = st.text_input("📍 出发地", value="")
        pref_opt = st.selectbox("🎯 偏好选择", ["⚡ 时间最短", "🚶 步行少", "🗺️ 换乘少"])
    with col_in2:
        dest_input = st.text_input("🏁 目的地", value="")
        expected_time = st.slider("⏳ 最大容忍晚点时间", 0, 60, 50)
    
    col_btn_l, col_btn_c, col_btn_r = st.columns([2, 2, 2])
    with col_btn_c:
        if st.button("🚀 开始为您规划", type="primary", use_container_width=True):
            st.session_state["search_params"] = {"origin": origin_input, "destination": dest_input}
            st.session_state["page_stage"] = "SCHEMES"
            st.rerun()

# ==========================================
# 【PAGE 2】多方案对比页
# ==========================================
elif st.session_state["page_stage"] == "SCHEMES":
    col_back, col_route_title = st.columns([1.5, 8.5])
    with col_back:
        if st.button("⬅️ 返回", type="primary", use_container_width=True):
            st.session_state["page_stage"] = "HOME"
            st.rerun()
    with col_route_title:
        params = st.session_state["search_params"]
        st.markdown(f"<div style='font-size: 1.2rem; color: #38bdf8; font-weight: 800;'>🛰️ 正在展示：从 {params.get('origin')} 至 {params.get('destination')} 的多维决策链</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="top-tips-bar">💡 提示：点击“选定方案”查看地图轨迹；点击红色按钮进入AR导航系统。</div>', unsafe_allow_html=True)
    
    schemes = [
        {"title": "🚇 地铁S1号线", "time": "42分钟", "distance": "35.2公里", "cost": "7.0元", "change": "无需换乘"},
        {"title": "🚌 大巴", "time": "55分钟", "distance": "37.1公里", "cost": "20.0元", "change": "无需换乘"},
        {"title": "🚗 顺风车", "time": "36分钟", "distance": "36.5公里", "cost": "40元", "change": "无需换乘"}
    ]
    
    col_left_list, col_right_preview = st.columns([5.5, 4.5])
    with col_left_list:
        for idx, item in enumerate(schemes):
            is_selected = (st.session_state["selected_scheme_idx"] == idx)
            border_css = "border: 3px solid #38bdf8 !important;" if is_selected else "border: 1px solid rgba(148,163,184,0.4) !important;"
            st.markdown(f"""<div class="crypto-card" style="{border_css} padding: 8px 18px !important; margin-bottom: 8px !important;"><span class="scheme-header-title">{item['title']}</span><div class="scheme-meta-row"><span>⏳ {item['time']}</span> <span>📏 {item['distance']}</span> <span>💰 {item['cost']}</span> <span class="scheme-meta-change">{item['change']}</span></div></div>""", unsafe_allow_html=True)
            
            col_b1, col_b2 = st.columns([4, 6])
            with col_b1:
                if st.button(f"🎯 选定方案", key=f"sel_{idx}", use_container_width=True):
                    st.session_state["selected_scheme_idx"] = idx
                    st.rerun()
            with col_b2:
                if st.button(f"👁️ 进入该方案AR导航界面", key=f"go_detail_{idx}", type="primary", use_container_width=True):
                    st.session_state["selected_scheme_idx"] = idx
                    st.session_state["page_stage"] = "DETAIL"
                    st.rerun()
                        
    with col_right_preview:
        st.markdown("### 📡 大致轨迹预览")
        html_map = f"""<div style="height:310px; background:#020617; border-radius:12px; border:1px solid #38bdf8; display:flex; align-items:center; justify-content:center; color:#38bdf8;">[ 高德地图渲染窗口 - 方案 {st.session_state['selected_scheme_idx']+1} ]</div>"""
        st.components.v1.html(html_map, height=310)

# ==========================================
# 【PAGE 3】3D数字孪生与 AR 导航深度页 (重构规整版)
# ==========================================
elif st.session_state["page_stage"] == "DETAIL":
    col_back_detail, col_title_detail = st.columns([1.5, 8.5])
    with col_back_detail:
        if st.button("⬅️ 返回方案选择", key="back_to_schemes_page", type="primary", use_container_width=True):
            st.session_state["page_stage"] = "SCHEMES"
            st.rerun()
    with col_title_detail:
        curr = st.session_state["selected_scheme_idx"]
        st.markdown(f"<div class='detail-main-title'>🪐 AR实景导航 (方案 {curr + 1})</div>", unsafe_allow_html=True)
    
    # 主体精细双栏对等排版
    col_media, col_strategy = st.columns([5.2, 4.8])
    
    with col_media:
        st.markdown("<div class='card-title'>🔮 AR 实时视导空间画面</div>", unsafe_allow_html=True)
        # 核心视频文件流载入接口
        video_file = "AR_video.mp4"
        if os.path.exists(video_file):
            st.video(video_file, format="video/mp4", loop=True, autoplay=True, muted=True)
        else:
            st.markdown(f"""
            <div style="background:#020617; border:2px dashed #38bdf8; border-radius:12px; height:295px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; color:#38bdf8; padding: 24px;">
                <div style="font-size:1.35rem; font-weight:800; margin-bottom:10px;">[ 🛸 AR 实景智慧导航视频流接口 ]</div>
                <div style="color:#ffffff; font-size:1.05rem; max-width:85%; line-height:1.6;">请将实景流线视频命名为 <b style="color:#38bdf8;">AR_video.mp4</b> 放入项目根目录下，系统将自动激活全息空间数字路线。</div>
            </div>
            """, unsafe_allow_html=True)
                
    with col_strategy:
        # 右侧上卡片：关键点指引说明
        st.markdown("""
        <div class="crypto-card" style="margin-bottom: 12px !important;">
            <div class="card-title">📄 关键点指引说明</div>
            <ul class="strategy-list">
                <li>1. 高铁下扶梯由西8口快速出站。</li>
                <li>2. 地铁S1号线 3车厢4门精准候车。</li>
                <li>3. 机场站下车垂直直达连廊。</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 右侧下卡片：实时出行提醒
        st.markdown("""
        <div class="crypto-card">
            <div class="card-title">⚠️ 实时出行提醒</div>
            <ul class="strategy-list">
                <li>• 务必认准空港新城方向，切勿错乘S3线。</li>
                <li>• 下一班地铁动态演化间隔为 9 分钟。</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)