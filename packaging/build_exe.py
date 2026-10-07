"""构建 Windows 桌面交付物:python packaging/build_exe.py [--mode onedir|onefile]

两种模式:
- onedir(默认,桌面模式):入口 packaging/desktop_entry.py,pywebview 原生窗口、不打开浏览器;
  产物 dist/zhishuxing/ 目录,供 packaging/package_zip.py 组装便携 zip。
  pythonnet/WebView2 的 DLL 与运行时配置依赖 collect-all 完整收集,onedir 布局最稳、启动最快。
- onefile(浏览器模式):入口 packaging/exe_entry.py,起服务后打开系统浏览器;
  产物 dist/zhishuxing.exe 单文件,保留为兼容回退。

前置:pip install pyinstaller + (onedir 模式) pip install pywebview;
图标取 web/mobile/icon-512.png(需 pillow,dev 依赖),失败不阻断;
openai 已装则真实 LLM 能力进包,未装则 exe 只有 Mock 模式(运行时如实标注)。
torch/tensorboard/mlagents_envs 永远排除:训练链路需要 Unity 环境,交付物只做运行态。
两种模式均构建为 --windowed(平时无黑窗口);致命错误由各入口现场可视化说明。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "zhishuxing"
ENTRY_BROWSER = ROOT / "packaging" / "exe_entry.py"
ENTRY_DESKTOP = ROOT / "packaging" / "desktop_entry.py"

# (源相对路径, 包内目标路径);运行时解压到 workspace 的清单在 bootstrap.RESOURCES
ADD_DATA = [
    ("src/zhishuxing/webapp/templates", "zhishuxing/webapp/templates"),
    ("src/zhishuxing/webapp/static", "zhishuxing/webapp/static"),
    ("configs", "configs"),
    ("web/mobile", "web/mobile"),
    ("data/transfer_kb", "data/transfer_kb"),
    ("data/samples", "data/samples"),
]
EXCLUDES = ("torch", "tensorboard", "mlagents_envs")
ICON_CANDIDATES = ("web/mobile/icon-512.png", "web/mobile/icon-192.png")
# pythonnet 的 `clr` 由 pywebview 运行时才 import,静态分析补一份保险
DESKTOP_HIDDEN = ("clr",)
DESKTOP_COLLECT = ("webview", "clr_loader", "pythonnet")


def _make_icon() -> Path | None:
    for rel in ICON_CANDIDATES:
        src = ROOT / rel
        if not src.exists():
            continue
        target = ROOT / "packaging" / "app.ico"
        try:
            from PIL import Image

            Image.open(src).save(
                target,
                sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
            )
            print(f"[build_exe] 图标: {target}")
            return target
        except Exception as exc:  # 图标生成失败不阻断构建
            print(f"[build_exe] 图标跳过({exc})")
            return None
    print("[build_exe] 未找到图标源 PNG,构建无图标版本")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="构建智枢星 Windows 交付物")
    parser.add_argument(
        "--mode",
        choices=("onedir", "onefile"),
        default="onedir",
        help="onedir=桌面模式(pywebview 窗口,默认);onefile=浏览器模式(兼容回退)",
    )
    args = parser.parse_args()
    desktop_mode = args.mode == "onedir"

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if importlib.util.find_spec("PyInstaller") is None:
        print("[build_exe] 缺少 PyInstaller: pip install pyinstaller")
        return 2
    if desktop_mode and importlib.util.find_spec("webview") is None:
        print("[build_exe] 桌面模式缺少 pywebview: pip install pywebview")
        return 2
    if importlib.util.find_spec("openai") is None:
        print("[build_exe] 未安装 openai: exe 将只有 Mock LLM(如需真实 LLM 先 pip install openai 再构建)")

    stamp = ROOT / "packaging" / "_build_version.txt"
    stamp.write_text(version, encoding="utf-8")

    entry = ENTRY_DESKTOP if desktop_mode else ENTRY_BROWSER
    sep = os.pathsep  # Windows 下 --add-data 的源/目标分隔符是 ';'
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--windowed",
        "--name", NAME,
        "--paths", str(ROOT / "src"),
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "exe"),
        "--specpath", str(ROOT / "build" / "exe"),  # spec 藏进已被 .gitignore 的 build/
    ]
    cmd += ["--onedir"] if desktop_mode else ["--onefile"]
    for rel, dest in ADD_DATA:
        src = ROOT / rel
        if not src.exists():
            print(f"[build_exe] 跳过不存在的资源: {rel}")
            continue
        cmd += ["--add-data", f"{src}{sep}{dest}"]
    cmd += ["--add-data", f"{stamp}{sep}."]
    for module in EXCLUDES:
        cmd += ["--exclude-module", module]
    if desktop_mode:
        for module in DESKTOP_HIDDEN:
            cmd += ["--hidden-import", module]
        for pkg in DESKTOP_COLLECT:
            cmd += ["--collect-all", pkg]
    icon = _make_icon()
    if icon:
        cmd += ["--icon", str(icon)]
    cmd.append(str(entry))

    print("[build_exe] " + subprocess.list2cmdline(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("[build_exe] 构建失败。若报非 ASCII 路径错误:subst X: <本项目所在盘符路径> 后从 X: 构建")
        return result.returncode
    artifact = ROOT / "dist" / (NAME if desktop_mode else f"{NAME}.exe")
    print(f"[build_exe] 产物: {artifact}")
    if desktop_mode:
        total = sum(p.stat().st_size for p in artifact.rglob("*") if p.is_file())
        print(f"[build_exe] 桌面模式目录总大小: {total / 1048576:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
