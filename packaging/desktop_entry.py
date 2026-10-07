"""桌面模式冻结入口:pywebview 原生窗口,不打开浏览器(由 scripts/build_exe.py --mode onedir 使用)。

与 exe_entry(浏览器模式)共用 bootstrap 引导,最后一公里不同:
1. WebView2 运行时预检(winreg 只读),缺失时原生对话框给出安装指引,不弹黑控制台;
2. Windows 命名互斥体做单实例:重复双击 = 重开窗口连到既有服务(viewer),而不是起第二个后端;
3. 后台 daemon 线程复用 CLI 的 serve --production(waitress),主线程先开品牌 loading 窗口;
4. 端口就绪后 load_url 切入控制台;?desktop=1 让前端禁用右键/缩放等浏览器专属行为;
5. 窗口关闭 → webview.start 返回 → 进程退出,daemon 服务线程随之终止。

viewer 模式注意:后端归首实例进程所有,关闭首实例窗口会停止服务,其余 viewer 窗口随之失效。
"""
from __future__ import annotations

import ctypes
import json
import sys
import threading
import time

import webview

from bootstrap import PORT, bootstrap, port_open

MUTEX_NAME = r"Local\zhishuxing-desktop-singleton"
READY_TIMEOUT = 30.0
ERROR_ALREADY_EXISTS = 183
BASE_URL = f"http://127.0.0.1:{PORT}"
APP_URL = f"{BASE_URL}/?desktop=1"
WINDOW_TITLE = "智枢星 · 大型枢纽智慧换乘引导系统"
# WebView2 Evergreen 运行时的 EdgeUpdate 产品 GUID
WEBVIEW2_KEY = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
WEBVIEW2_DOWNLOAD = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

# 品牌 loading 页:深色侧栏同族渐变 + 主题靛蓝,避免白屏闪烁;
# setStatus 由 Python 端 evaluate_js 调用,汇报启动进度
LOADING_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>智枢星</title>
<style>
  html,body{margin:0;height:100%;}
  body{display:flex;align-items:center;justify-content:center;
       background:linear-gradient(135deg,#0b1020,#141b33);color:#eef1ff;
       font-family:"Segoe UI","Microsoft YaHei",sans-serif;user-select:none;overflow:hidden;}
  .box{text-align:center}
  .mark{width:76px;height:76px;margin:0 auto 22px;border-radius:22px;
        background:linear-gradient(135deg,#4f46e5,#7c3aed);
        display:flex;align-items:center;justify-content:center;font-size:36px;color:#fff;
        box-shadow:0 6px 16px -4px rgba(79,70,229,.45);}
  h1{margin:0 0 8px;font-size:26px;letter-spacing:3px}
  p{margin:0;color:#98a2b8;font-size:13px}
  .spin{width:22px;height:22px;margin:28px auto 0;border-radius:50%;
        border:2px solid rgba(238,241,255,.18);border-top-color:#8b8df2;
        animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  #status{margin-top:14px;color:#69758c;font-size:12px}
</style></head><body>
<div class="box">
  <div class="mark">✦</div>
  <h1>智枢星</h1>
  <p>综合交通枢纽智慧换乘引导系统</p>
  <div class="spin"></div>
  <div id="status">正在准备本地服务…</div>
</div>
<script>function setStatus(t){var e=document.getElementById("status");if(e){e.textContent=t;}}</script>
</body></html>"""

_MUTEX_HANDLE = []  # 持有互斥体句柄,进程存活期间不释放
_LOG_PATH = None  # workspace/zhishuxing_desktop.log,bootstrap 之后赋值


def log(message: str) -> None:
    """stdout 在冻结+重定向下是块缓冲的,排查启动卡点必须落盘且即时刷。"""
    if _LOG_PATH is None:
        return
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    except OSError:
        pass


def webview2_runtime_version() -> str | None:
    """只读查询 Evergreen WebView2 运行时是否已安装;返回版本号或 None。"""
    import winreg

    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for sub in (WEBVIEW2_KEY, rf"SOFTWARE\{WEBVIEW2_KEY}", rf"SOFTWARE\WOW6432Node\{WEBVIEW2_KEY}"):
            try:
                with winreg.OpenKey(root, sub) as key:
                    pv, _ = winreg.QueryValueEx(key, "pv")
                if pv and pv != "0.0.0.0":
                    return str(pv)
            except OSError:
                continue
    return None


def acquire_single_instance() -> bool:
    """True = 首实例(负责起服务);False = 已有实例在运行或启动中(转 viewer 模式)。"""
    kernel32 = ctypes.windll.kernel32
    _MUTEX_HANDLE.append(kernel32.CreateMutexW(None, False, MUTEX_NAME))
    return kernel32.GetLastError() != ERROR_ALREADY_EXISTS


def error_dialog(title: str, detail: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, detail, f"智枢星 - {title}", 0x10)  # MB_ICONERROR


def run_server() -> None:
    try:
        from zhishuxing.cli import main as cli_main

        code = cli_main(["serve", "--host", "127.0.0.1", "--port", str(PORT), "--production"])
        print(f"[desktop] server exited with code {code}")
    except Exception:
        import traceback

        traceback.print_exc()


def main() -> int:
    global _LOG_PATH
    if not getattr(sys, "frozen", False):
        print("本入口仅供打包后的 exe 使用;源码开发请执行 zhishuxing serve 后用 launcher.bat 调试")
        return 2
    try:
        workspace = bootstrap()
    except SystemExit as exc:
        error_dialog("初始化被中止", str(exc))
        return 1
    except Exception:
        import traceback

        error_dialog("发生未处理异常", traceback.format_exc())
        return 1

    _LOG_PATH = workspace / "zhishuxing_desktop.log"
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    log("=== desktop entry start ===")

    version = webview2_runtime_version()
    if version is None:
        log("WebView2 runtime NOT FOUND")
        error_dialog(
            "缺少 WebView2 运行时",
            "智枢星桌面版需要 Microsoft Edge WebView2 运行时(Windows 10/11 通常已内置)。\n\n"
            f"请下载安装 Evergreen 引导程序后重试:\n{WEBVIEW2_DOWNLOAD}",
        )
        return 1
    log(f"WebView2 runtime: {version}")

    viewer = not acquire_single_instance() or port_open(PORT)
    log(f"viewer={viewer}")
    if viewer:
        print("[desktop] 检测到已有实例,本次仅作为窗口连接既有服务")
    else:
        threading.Thread(target=run_server, daemon=True).start()

    window = webview.create_window(
        WINDOW_TITLE,
        html=LOADING_HTML,
        width=1440,
        height=900,
        min_size=(1200, 760),
        background_color="#0b1020",
    )
    log("window created, starting webview event loop")
    webview.start(_switch_when_ready, (window, str(workspace)), gui="edgechromium")
    log("webview event loop returned (window closed)")
    return 0


def _switch_when_ready(window: webview.Window, workspace: str) -> None:
    """webview 事件循环启动后执行:轮询端口就绪,把 loading 页切入控制台。

    pywebview 的 start(func, args) 会把元组解包成多参数,故签名必须是两个形参。
    """
    workspace = str(workspace)
    log("switcher thread started")

    def set_status(text: str) -> None:
        try:
            window.evaluate_js(f"setStatus({json.dumps(text)})")
        except Exception as exc:
            log(f"set_status failed: {exc!r}")

    deadline = time.monotonic() + READY_TIMEOUT
    ready = False
    while time.monotonic() < deadline:
        if port_open(PORT):
            ready = True
            break
        time.sleep(0.25)
    log(f"port ready={ready}")
    if not ready:
        log("READY TIMEOUT, showing error dialog")
        error_dialog(
            "启动失败",
            f"本地服务未在 {READY_TIMEOUT:.0f} 秒内就绪。\n\n"
            f"workspace: {workspace}\n"
            "可删除 workspace 后重试;若问题持续,请查看 zhishuxing_workspace 下的日志。",
        )
        window.destroy()
        return
    set_status("正在加载控制台…")
    try:
        window.load_url(APP_URL)
        log(f"load_url({APP_URL}) called OK")
    except Exception as exc:
        log(f"load_url FAILED: {exc!r}")


if __name__ == "__main__":
    raise SystemExit(main())
