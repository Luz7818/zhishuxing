# packaging/ —— Windows 交付物的打包资产

> 用途：说明 PyInstaller 打包用的入口脚本与两份生成资产各是什么、为什么入库。

本目录服务于 `packaging/build_exe.py` 的两种构建模式：

- **桌面模式（onedir，默认）**：`desktop_entry.py` 是 PyInstaller 打包入口，负责冻结运行时的
  引导（定 workspace → 解压内置资源 → WebView2 预检 → 单实例互斥 → 品牌 loading 窗口 →
  后台起 waitress 服务 → 窗口切入控制台）。双击 exe 弹出**原生应用窗口，不打开浏览器**，
  pywebview 走 Windows 自带的 Edge WebView2 渲染。
- **浏览器模式（onefile，兼容回退）**：`exe_entry.py` 是打包入口，起服务后自动打开系统
  浏览器，与 `launcher.bat` 同语义。保留用于最小交付与问题排查。

`app.ico` 与 `_build_version.txt` 是构建脚本写入的入库资产。两个入口共用 `bootstrap.py`
的引导逻辑（workspace 选择、资源解压、端口探测、致命错误兜底）。

## 文件清单

| 文件 | 干什么 | 备注 |
|---|---|---|
| `bootstrap.py` | 两模式共享的冻结引导：选可写 workspace 并设 `ZHISHUXING_WORKSPACE` → 解压内置资源（版本戳一致跳过，升级即重建）→ 端口探测 → 致命错误可视化 | 被两个入口 import，勿直接运行 |
| `desktop_entry.py` | 桌面模式入口（onedir 构建）：bootstrap → WebView2 运行时检测（缺失弹原生对话框给下载指引）→ `Local\zhishuxing-desktop-singleton` 互斥体单实例（重复双击 = 重开窗口连既有服务）→ daemon 线程跑 `serve --production` → pywebview 品牌 loading 页切入 `/?desktop=1`；关窗即退出 | 由 `packaging/build_exe.py --mode onedir` 指定，勿直接运行 |
| `exe_entry.py` | 浏览器模式入口（onefile 构建）：bootstrap → 端口占用视为已有实例直接开浏览器 → 后台轮询就绪后开浏览器 → 复用 `serve --production`（waitress） | 由 `packaging/build_exe.py --mode onefile` 指定，勿直接运行 |
| `build_exe.py` | PyInstaller 构建脚本：`--mode onedir`（默认）出桌面模式目录 `dist/zhishuxing/`，`--mode onefile` 出浏览器模式单文件 `dist/zhishuxing.exe`；顺带生成 `app.ico` 与 `_build_version.txt` | 需先 `pip install pyinstaller`（桌面模式另需 `pip install pywebview`）；见下面「构建与组装」 |
| `package_zip.py` | 把 onedir 产物 + `使用说明.txt` + LICENSE 组装为便携分发包 `dist/zhishuxing-<版本>-win64-portable.zip` | `--skip-build` 直接打包现有产物；见下面「构建与组装」 |
| `build_deploy.py` | 私有化离线交付包：docker build → `docker save` 后 gzip 成镜像包 → 组装 `dist/zhishuxing-deploy-<版本>.zip`（镜像 + compose + .env.example + LICENSE + eula-template + DEPLOY-PRIVATE + 交付说明） | 需要本机 docker 守护进程；目标机 `docker load` 免网络；见 `docs/DEPLOY-PRIVATE.md` |
| `deploy-compose.yml` | 交付包内的 compose 模板（镜像名按版本号替换）：数据三卷 + license.lic 挂载位 + env_file 注入 + healthcheck | 由 `build_deploy.py` 读取生成，勿直接运行 |
| `app.ico` | exe 图标，由 `web/mobile/icon-512.png` 经 pillow 转换生成 | 生成物，勿手改 |
| `_build_version.txt` | 内置资源版本戳（内容为打包时 `pyproject.toml` 的版本号），exe 首次运行据此决定是否重解压资源（版本戳一致则跳过，升级即重建） | 生成物，勿手改 |

## 构建与组装

在**仓库根目录**执行：

```bash
pip install pyinstaller
python packaging/build_exe.py                  # onedir 桌面模式(默认)
python packaging/build_exe.py --mode onefile   # onefile 浏览器模式(兼容回退)
python packaging/package_zip.py                # 先构建再打包
python packaging/package_zip.py --skip-build   # 直接打包现有 dist/zhishuxing/
```

两种模式均 `--windowed`（平时无黑窗口，致命错误由各自入口现场可视化说明）。共同行为要点：

- 图标取 `web/mobile/icon-512.png`，经 pillow 转成 `app.ico`，转换失败不阻断；
- 装了 `openai` 则真实 LLM 能力进包，没装则 exe 只有 Mock 模式（运行时如实标注）；
- `torch`/`tensorboard`/`mlagents_envs` 永远排除：训练链路需要 Unity 环境，交付物只做运行态。

便携 zip 解压到任意可写位置，双击 `zhishuxing\zhishuxing.exe` 即用；首启在 exe 旁建
`zhishuxing_workspace`，删除整个文件夹即卸载。

## 和谁打交道

- **上游**：`packaging/build_exe.py` 构建时写 `app.ico` 与 `_build_version.txt`，按模式把
  `desktop_entry.py`（onedir）或 `exe_entry.py`（onefile）指定为 PyInstaller 入口；
  桌面模式额外 `--collect-all webview/clr_loader/pythonnet`。
- **下游**：`dist/zhishuxing/`（桌面模式 onedir 目录，由 `packaging/package_zip.py` 组装便携
  zip）、`dist/zhishuxing.exe`（浏览器模式单文件，均为运行产物，不入库）。
- **改这里之后要跑**：`python packaging/build_exe.py` 重新打包并在 Windows 上双击冒烟一遍
  （启动画面 → 主界面 → 对话 → 报告 → 关窗退出 → 重复启动）；浏览器模式另跑
  `dist/zhishuxing.exe` 确认开浏览器语义未变。

## 别动

- 两个入口里「先设置 `ZHISHUXING_WORKSPACE` 再 import 任何 `zhishuxing` 模块」的顺序：
  `config.py` 在 import 时求值路径，提前 import 会把 workspace 定死到仓库目录，exe 在别人
  机器上就会往只读位置写数据。
- `_build_version.txt` 由 `build_exe.py` 从 `pyproject.toml` 的 `version` 写入，与 exe 内嵌
  资源配套；手改版本戳会让升级后的 exe 跳过资源重解压，出现旧语料配新代码的混跑。
- 桌面模式 `desktop_entry.py` 的单实例互斥体名字（`Local\zhishuxing-desktop-singleton`）与
  端口 7860 约定是配套的：改端口要连 `bootstrap.PORT` 一起改。
- 本目录不在 pyflakes 门禁路径里（门禁是 `python -m pyflakes src/ scripts/ tests/`），
  入口脚本的语法错误只会在下次打包时暴露——改完它们要手动跑一次
  `python -m pyflakes packaging/`。
