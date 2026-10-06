"""单 exe 运行引导:PyInstaller 打包入口(由 scripts/build_exe.py 使用)。

冻结运行时的职责,顺序即正确性:
1. 先选定可写 workspace 并设置 ZHISHUXING_WORKSPACE——config.py 在 import 时求值,
   任何 zhishuxing 导入都必须发生在设置之后;
2. 首次运行把内置资源解压到 workspace(版本戳一致则跳过,升级即重建);
3. 端口 7860 已被占用时视为已有实例,直接开浏览器退出(与 启动器.bat 同语义);
4. 后台线程轮询端口就绪后自动开浏览器;
5. 复用 CLI 的 serve --production(waitress);
6. 致命错误时现场分配控制台展示原因——构建用 --windowed,平时无黑窗口。
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

PORT = 7860
STAMP = ".build_version"
# 打包进 _MEIPASS 的随行资源,解压到 workspace 后由 config.paths 锚定
RESOURCES = ("configs", "web/mobile", "data/transfer_kb", "data/samples")


def _writable(candidate: Path) -> bool:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _choose_workspace() -> Path:
    # 优先 exe 旁(可见、删除即卸载);Program Files 等只读位置回退 %LOCALAPPDATA%
    candidates = [Path(sys.executable).resolve().parent / "zhishuxing_workspace"]
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "zhishuxing")
    for candidate in candidates:
        if _writable(candidate):
            return candidate
    raise SystemExit("未找到可写目录:请把 exe 放到可写位置,或检查 %LOCALAPPDATA% 权限")


def _extract_resources(workspace: Path) -> None:
    # sys._MEIPASS 只在 PyInstaller 冻结运行时存在
    src = Path(sys._MEIPASS)
    version = (src / "_build_version.txt").read_text(encoding="utf-8").strip()
    stamp = workspace / STAMP
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == version:
        return
    for rel in RESOURCES:
        if not (src / rel).exists():
            continue
        target = workspace / rel
        if target.exists():
            shutil.rmtree(target)  # 升级即重建;workspace 内对内置资源的本地改动会被覆盖
        shutil.copytree(src / rel, target)
    stamp.write_text(version, encoding="utf-8")


def _port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout).close()
        return True
    except OSError:
        return False


def _open_browser_when_ready(port: int) -> None:
    url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _port_open(port):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def _show_fatal_error(title: str, detail: str) -> None:
    """windowed 构建平时无控制台:致命错误时现场分配一个,把原因说清楚再退出。"""
    import ctypes

    try:
        ctypes.windll.kernel32.AllocConsole()
        sys.stdout = open("CONOUT$", "w")
        sys.stderr = sys.stdout
        sys.stdin = open("CONIN$", "r")
        ctypes.windll.kernel32.SetConsoleTitleW(f"智枢星 - {title}")
        print(f"启动失败: {title}\n\n{detail}")
        input("\n按回车键关闭窗口...")
    except Exception:
        pass  # 控制台都分配不出来时无处可写,保证干净退出即可


def main() -> int:
    if not getattr(sys, "frozen", False):
        print("本入口仅供打包后的 exe 使用;源码运行请执行 zhishuxing serve")
        return 2
    try:
        return _serve()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code:
            _show_fatal_error("初始化被中止", str(exc))
        return code
    except Exception:
        import traceback

        _show_fatal_error("发生未处理异常", traceback.format_exc())
        return 1


def _serve() -> int:
    workspace = _choose_workspace()
    _extract_resources(workspace)
    os.environ["ZHISHUXING_WORKSPACE"] = str(workspace)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(f"workspace: {workspace}")
    if _port_open(PORT):
        print(f"端口 {PORT} 已有服务在运行,直接打开浏览器;如需重启请先退出旧进程")
        webbrowser.open(f"http://127.0.0.1:{PORT}/")
        return 0
    threading.Thread(target=_open_browser_when_ready, args=(PORT,), daemon=True).start()
    from zhishuxing.cli import main as cli_main

    return cli_main(["serve", "--host", "127.0.0.1", "--port", str(PORT), "--production"])


if __name__ == "__main__":
    raise SystemExit(main())
