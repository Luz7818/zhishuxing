"""组装便携 zip 分发包:python packaging/package_zip.py [--skip-build]

流程:
1. (默认)先执行 packaging/build_exe.py --mode onedir 构建桌面模式产物;
2. 把 dist/zhishuxing/ 整目录 + 使用说明.txt + LICENSE 压成
   dist/zhishuxing-<version>-win64-portable.zip,解压即用。

使用说明以 UTF-8 BOM 写入,保证旧版记事本打开不乱码。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "dist" / "zhishuxing"


def _usage_text(version: str) -> str:
    return f"""智枢星 · 综合交通枢纽智慧换乘引导系统
便携版 v{version} (Windows 64 位)
构建日期:{datetime.now().strftime("%Y-%m-%d")}

【快速开始】
1. 把整个文件夹解压到任意可写位置(如桌面、D:\\;不要放在需要管理员权限的目录);
2. 双击 zhishuxing\\zhishuxing.exe,稍候出现启动画面,随后自动进入主界面;
3. 关闭窗口即完全退出。

【系统要求】
- Windows 10/11 64 位;
- Microsoft Edge WebView2 运行时(Win10/11 通常已内置;缺失时程序会弹窗给出去官方下载的链接);
- 无需安装 Python 或任何依赖。

【数据存哪里】
- 首次运行会在 exe 旁创建 zhishuxing_workspace 文件夹,存放:
  对话记录(SQLite)、导航与仿真产物(PNG/CSV/GIF)、内置资源副本、日志;
- 升级版本:用新 exe 覆盖旧 exe 即可,workspace 中的内置资源会自动按版本重建,
  你的对话与产出数据不会被删除;把整个文件夹删掉即完成卸载。

【联网说明】
- 完全离线可用:地图自动降级为内置 Canvas 线路图,对话使用内置模板回复;
- 可选联网增强:
  · 高德地图底图与真实公交路线(需在设置页配置高德 Key);
  · 大模型对话(需在设置页配置 OpenAI 兼容接口与 Key);
  配置保存在 zhishuxing_workspace\\.env,也可用记事本直接编辑。

【常见问题】
- 双击无反应或被杀毒软件拦截:单文件交付常见误报,请在杀毒软件中选择"信任/允许";
- 提示"本地服务未就绪":确认 7860 端口未被其他程序占用,或删除 zhishuxing_workspace 后重试;
- 再次双击 exe 不会启动第二个后端,而是新开一个窗口连接已运行的服务。

【版本信息】
- 版本:{version}
- 技术栈:Python + Flask(waitress) + MADDPG 多智能体强化学习 + pywebview(WebView2)
- 本软件仅供学习与演示使用,详见随附 LICENSE 文件。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="组装智枢星便携 zip 分发包")
    parser.add_argument("--skip-build", action="store_true", help="跳过构建,直接打包现有 dist/zhishuxing/")
    args = parser.parse_args()

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

    if not args.skip_build:
        print("[package_zip] 先构建桌面模式产物...")
        build = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_exe.py"), "--mode", "onedir"])
        if build.returncode != 0:
            print("[package_zip] 构建失败,终止打包")
            return build.returncode

    exe = APP_DIR / "zhishuxing.exe"
    if not exe.exists():
        print(f"[package_zip] 未找到 {exe},请先构建(--mode onedir)")
        return 2

    out = ROOT / "dist" / f"zhishuxing-{version}-win64-portable.zip"
    print(f"[package_zip] 打包 -> {out}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(APP_DIR.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(APP_DIR)
            # zhishuxing_workspace 是本机运行时数据(日志/会话库),绝不能进交付物
            if rel.parts[0] == "zhishuxing_workspace":
                continue
            zf.write(path, arcname=str(Path("zhishuxing") / rel))
        zf.writestr("使用说明.txt", _usage_text(version))
        license_path = ROOT / "LICENSE"
        if license_path.exists():
            zf.writestr("LICENSE", license_path.read_text(encoding="utf-8"))
    print(f"[package_zip] 完成: {out} ({out.stat().st_size / 1048576:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
