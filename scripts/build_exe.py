"""构建 Windows 单文件 exe:python scripts/build_exe.py

产物 dist/zhishuxing.exe。前置:pip install pyinstaller;
图标取 web/mobile/icon-512.png(需 pillow,dev 依赖),失败不阻断;
openai 已装则真实 LLM 能力进包,未装则 exe 只有 Mock 模式(运行时如实标注)。
torch/tensorboard/mlagents_envs 永远排除:训练链路需要 Unity 环境,exe 只做运行态。
构建为 --windowed(平时无黑窗口);致命错误由 packaging/exe_entry.py 现场弹控制台说明。
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "zhishuxing"
ENTRY = ROOT / "packaging" / "exe_entry.py"

# (源相对路径, 包内目标路径);运行时解压到 workspace 的清单在 exe_entry.RESOURCES
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
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if importlib.util.find_spec("PyInstaller") is None:
        print("[build_exe] 缺少 PyInstaller: pip install pyinstaller")
        return 2
    if importlib.util.find_spec("openai") is None:
        print("[build_exe] 未安装 openai: exe 将只有 Mock LLM(如需真实 LLM 先 pip install openai 再构建)")

    stamp = ROOT / "packaging" / "_build_version.txt"
    stamp.write_text(version, encoding="utf-8")

    sep = os.pathsep  # Windows 下 --add-data 的源/目标分隔符是 ';'
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--windowed", "--clean", "--noconfirm",
        "--name", NAME,
        "--paths", str(ROOT / "src"),
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "exe"),
        "--specpath", str(ROOT / "build" / "exe"),  # spec 藏进已被 .gitignore 的 build/
    ]
    for rel, dest in ADD_DATA:
        src = ROOT / rel
        if not src.exists():
            print(f"[build_exe] 跳过不存在的资源: {rel}")
            continue
        cmd += ["--add-data", f"{src}{sep}{dest}"]
    cmd += ["--add-data", f"{stamp}{sep}."]
    for module in EXCLUDES:
        cmd += ["--exclude-module", module]
    icon = _make_icon()
    if icon:
        cmd += ["--icon", str(icon)]
    cmd.append(str(ENTRY))

    print("[build_exe] " + subprocess.list2cmdline(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("[build_exe] 构建失败。若报非 ASCII 路径错误:subst X: <本项目所在盘符路径> 后从 X: 构建")
        return result.returncode
    exe = ROOT / "dist" / f"{NAME}.exe"
    print(f"[build_exe] 产物: {exe} ({exe.stat().st_size / 1048576:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
