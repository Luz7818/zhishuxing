"""前端结构回归网（X2-02）：回读保护次序 + a11y 锚点。

出处：核查报告/_体检-2026-11-05/研究三轮-拍板项执行手册.md 第六节（X2-02）。
前置修复：ZX2-1——mobile_app.html finish() 的贴底测量已挪到 append 之前（与桌面
app.js appendChatMessage 同构），本文件把修复后的正确次序钉进门禁。
原则：只断言「结构存在与次序正确」，不断言样式细节——防脆断。
全部走 Flask client 取产物（跟随包路径，CI/安装态都成立），零新依赖。
"""
from __future__ import annotations

import re


def _static(client, path: str) -> str:
    return client.get(path).data.decode("utf-8")


def _fn_body(src: str, header: str, stop: str) -> str:
    """截取 header 起到 stop 首次出现之间的源码段，用于次序断言。"""
    i = src.index(header)
    return src[i : src.index(stop, i)]


# ---- 回读保护（X2-02 核心：测量必须先于 append） ----


def test_chat_autoscroll_measures_before_append_desktop(client):
    src = _static(client, "/static/app.js")
    body = _fn_body(src, "function appendChatMessage", "\nfunction renderProfileChips")
    assert body.index("const nearBottom") < body.index("box.appendChild(el)")
    assert 'nearBottom || role === "user"' in body  # 用户发送始终滚底


def test_chat_autoscroll_measures_before_append_mobile(client):
    """ZX2-1 修复的钉子：finish() 里 chatNearBottom() 测量必须先于 appendChild，
    且 append 之后不得再测（chatNearBottom() 在函数体内只允许出现这一次）——
    长回复落地后距底必 >80px，append 后再测会误判用户已翻阅、贴底跟随失效。"""
    src = _static(client, "/mobile")
    body = _fn_body(src, "const finish = (data) => {", "state.isPlanning = false;")
    assert body.index("chatNearBottom()") < body.index("appendChild(assistantMsgEl(entry))")
    assert body.count("chatNearBottom()") == 1  # 防止 append 后的二次测量回潮
    assert "if (nearBottom) scrollChatBottom();" in body  # 滚底判定用的是 append 前的测量值


# ---- a11y 锚点（2.11.0 实现的结构钉） ----


def test_chat_aria_live_desktop(client):
    html = client.get("/").data.decode("utf-8")
    assert re.search(r'id="chatMessages"[^>]*aria-live="polite"', html)
    assert 'id="toastHost" aria-live="polite"' in html


def test_chat_aria_live_mobile(client):
    html = _static(client, "/mobile")
    assert re.search(r'id="chatLog"[^>]*aria-live="polite"', html)


def test_gallery_keyboard_template(client):
    src = _static(client, "/static/app.js")
    for anchor in (
        'tabindex="0"',
        'role="button"',
        'aria-label="放大查看',
        '"Enter" || e.key === " "',
    ):
        assert anchor in src, f"画廊键盘可达性锚点缺失: {anchor}"


def test_canvas_role_img_labels(client):
    """桌面 5 张画布在 index.html 静态标记里；chatRouteCanvas 由 app.js:1404
    动态注入（模板里无此节点），其 role/aria 锚点在 app.js 源码里钉。"""
    html = client.get("/").data.decode("utf-8")
    for cid in (
        "navCanvas",
        "rewardCanvas",
        "planCanvas",
        "ftLossCanvas",
        "ftMetricCanvas",
    ):
        assert re.search(rf'id="{cid}"[^>]*role="img"[^>]*aria-label=', html), cid
    src = _static(client, "/static/app.js")
    assert re.search(
        r'id="chatRouteCanvas"[^>]*role="img"[^>]*aria-label="[^"]*"',
        src,
    ), "chatRouteCanvas"


def test_reduced_motion_fallbacks_present(client):
    css = _static(client, "/static/style.css")
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "必须显式落到终态" in css  # 空态插画 reduced 直出终态的哨兵注释(style.css:1371 区)
    assert "@media (prefers-reduced-motion: reduce)" in _static(client, "/mobile")
