"""浏览器模式冻结入口:PyInstaller 打包入口(由 packaging/build_exe.py --mode onefile 使用)。

启动本地服务后打开系统浏览器——保留为兼容回退与开发自检链路;
"不走浏览器"的桌面窗口模式见 desktop_entry.py(onedir 构建)。

冻结运行时的职责,顺序即正确性:
1. bootstrap 先选定可写 workspace 并设置 ZHISHUXING_WORKSPACE——config.py 在 import 时求值,
   任何 zhishuxing 导入都必须发生在设置之后;
2. 首次运行把内置资源解压到 workspace(版本戳一致则跳过,升级即重建);
3. 端口 7860 已被占用时视为已有实例,直接开浏览器退出(与 launcher.bat 同语义);
4. 后台线程轮询端口就绪后自动开浏览器;
5. 复用 CLI 的 serve --production(waitress);
6. 致命错误时现场分配控制台展示原因——构建用 --windowed,平时无黑窗口。
"""
from __future__ import annotations

import sys
import threading
import time
import webbrowser

from bootstrap import PORT, bootstrap, port_open, show_fatal_error


def main() -> int:
    if not getattr(sys, "frozen", False):
        print("本入口仅供打包后的 exe 使用;源码运行请执行 zhishuxing serve")
        return 2
    try:
        return _serve()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code:
            show_fatal_error("初始化被中止", str(exc))
        return code
    except Exception:
        import traceback

        show_fatal_error("发生未处理异常", traceback.format_exc())
        return 1


def _open_browser_when_ready(port: int) -> None:
    url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if port_open(port):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def _serve() -> int:
    bootstrap()
    if port_open(PORT):
        print(f"端口 {PORT} 已有服务在运行,直接打开浏览器;如需重启请先退出旧进程")
        webbrowser.open(f"http://127.0.0.1:{PORT}/")
        return 0
    from zhishuxing import licensing

    try:
        licensing.ensure_serve_allowed()
    except licensing.LicenseExpired as exc:
        show_fatal_error("授权已过期", f"{exc}\n\n续期后把新的 license.lic 放到 workspace 根目录,再重新启动。")
        return 3
    threading.Thread(target=_open_browser_when_ready, args=(PORT,), daemon=True).start()
    from zhishuxing.cli import main as cli_main

    return cli_main(["serve", "--host", "127.0.0.1", "--port", str(PORT), "--production"])


if __name__ == "__main__":
    raise SystemExit(main())
