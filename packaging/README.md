# packaging/ —— Windows 单文件 exe 的打包资产

> 用途：说明 PyInstaller 打包用的入口脚本与两份生成资产各是什么、为什么入库。

本目录服务于 `scripts/build_exe.py` 的 exe 构建：`exe_entry.py` 是 PyInstaller 的打包入口，
负责冻结运行时的引导（定 workspace → 解压内置资源 → 端口检测 → 自动开浏览器）；
`app.ico` 与 `_build_version.txt` 是构建脚本写入的入库资产。没有这个目录，
`build_exe.py` 产不出可双击的 exe。

## 文件清单

| 文件 | 干什么 | 备注 |
|---|---|---|
| `exe_entry.py` | 单 exe 运行引导（PyInstaller 打包入口）：先选定可写 workspace 并设置 `ZHISHUXING_WORKSPACE`，再解压内置资源、检测 7860 端口占用、就绪后自动开浏览器、复用 `serve --production`（waitress） | 由 `scripts/build_exe.py` 指定为入口，勿直接运行 |
| `app.ico` | exe 图标，由 `web/mobile/icon-512.png` 经 pillow 转换生成 | 生成物，勿手改 |
| `_build_version.txt` | 内置资源版本戳（内容如 `2.3.0`），exe 首次运行据此决定是否重解压资源（版本戳一致则跳过，升级即重建） | 生成物，勿手改 |

## 和谁打交道

- **上游**：`scripts/build_exe.py` 构建时写 `app.ico` 与 `_build_version.txt`，并把
  `exe_entry.py` 指定为 PyInstaller 入口。
- **下游**：`dist/zhishuxing.exe`（运行产物，不入库）。
- **改这里之后要跑**：`python scripts/build_exe.py` 重新打包并在 Windows 上双击冒烟一遍
  （对话、报告、`/mobile` 三处）。

## 别动

- `exe_entry.py` 里「先设置 `ZHISHUXING_WORKSPACE` 再 import 任何 `zhishuxing` 模块」的顺序：
  `config.py` 在 import 时求值路径，提前 import 会把 workspace 定死到仓库目录，exe 在别人
  机器上就会往只读位置写数据。
- `_build_version.txt` 由 `build_exe.py` 从 `pyproject.toml` 的 `version` 写入，与 exe 内嵌
  资源配套；手改版本戳会让升级后的 exe 跳过资源重解压，出现旧语料配新代码的混跑。
- 本目录不在 pyflakes 门禁路径里（门禁是 `python -m pyflakes src/ scripts/ tests/`），
  `exe_entry.py` 的语法错误只会在下次打包时暴露——改完它要手动跑一次
  `python -m pyflakes packaging/`。
