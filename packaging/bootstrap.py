"""打包引导共享模块:desktop_entry(桌面模式)与 exe_entry(浏览器模式)的公共职责。

冻结运行时的引导顺序即正确性:
1. 先选定可写 workspace 并设置 ZHISHUXING_WORKSPACE——config.py 在 import 时求值,
   任何 zhishuxing 导入都必须发生在设置之后;
2. 首次运行把内置资源解压到 workspace(版本戳一致则跳过,升级即重建);
3. 端口探测(7860 已占用视为已有实例);
4. 致命错误的可视化兜底(--windowed 构建平时无黑窗口)。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PORT = 7860
STAMP = ".build_version"
# 打包进 _MEIPASS(_internal) 的随行资源,解压到 workspace 后由 config.paths 锚定
RESOURCES = ("configs", "web/mobile", "data/transfer_kb", "data/samples")


def writable(candidate: Path) -> bool:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def choose_workspace() -> Path:
    # 优先 exe 旁(可见、删除即卸载);Program Files 等只读位置回退 %LOCALAPPDATA%
    candidates = [Path(sys.executable).resolve().parent / "zhishuxing_workspace"]
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "zhishuxing")
    for candidate in candidates:
        if writable(candidate):
            return candidate
    raise SystemExit("未找到可写目录:请把 exe 放到可写位置,或检查 %LOCALAPPDATA% 权限")


def extract_resources(workspace: Path) -> None:
    # sys._MEIPASS 在 PyInstaller onefile/onedir 冻结运行时都存在(onefile 指向解压临时目录)
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


def port_open(port: int, timeout: float = 0.5) -> bool:
    import socket

    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout).close()
        return True
    except OSError:
        return False


def show_fatal_error(title: str, detail: str) -> None:
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


def bootstrap() -> Path:
    """选定 workspace、解压内置资源、设置环境变量;返回 workspace 供上层展示。"""
    workspace = choose_workspace()
    extract_resources(workspace)
    os.environ["ZHISHUXING_WORKSPACE"] = str(workspace)
    os.environ.setdefault("MPLBACKEND", "Agg")
    print(f"workspace: {workspace}")
    return workspace
