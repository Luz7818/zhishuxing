# scripts/ —— 仓库级工具脚本

> 用途：说明这个目录两个脚本各做什么、产物落在哪、为什么它们不在 CI 里跑。

本目录只放**不参与运行时**的开发者工具。所有能被 `zhishuxing` 子命令做到的事都在这里，
不要往这个目录加功能脚本。

## 文件清单

| 文件 | 干什么 | 备注 |
|---|---|---|
| `render_brand_assets.py` | 从矢量设计参数直接绘制品牌 PNG：PWA 图标与 favicon | 入口脚本，无参数 |
| `build_exe.py` | 构建 Windows 单文件 exe（PyInstaller），产物 `dist/zhishuxing.exe`；顺带生成 `packaging/app.ico` 与 `packaging/_build_version.txt` | 入口脚本，需先 `pip install pyinstaller`；见下面「build_exe.py」 |

## render_brand_assets.py

在**仓库根目录**执行：

```bash
python scripts/render_brand_assets.py
```

它写 7 个文件，全部是入库的生成物：

| 输出 | 用途 |
|---|---|
| `web/mobile/icon-512.png`、`icon-192.png`、`icon-maskable-512.png` | PWA 安装图标，`manifest.webmanifest` 按名字引用 |
| `web/mobile/apple-touch-icon.png` | iOS 添加到主屏 |
| `web/mobile/splash-logo.png` | iOS 启动屏 |
| `src/zhishuxing/webapp/static/assets/favicon-48.png`、`favicon-32.png` | 控制台标签页图标 |

图标与 `logo-mark.svg` 用同一套几何定义、同一组品牌色，4 倍超采样后 LANCZOS 缩小。
改了 `logo-mark.svg` 的形状或品牌色，就要重跑这个脚本，否则 PNG 与 SVG 两套资产会不一致。

## build_exe.py

在**仓库根目录**执行：

```bash
pip install pyinstaller
python scripts/build_exe.py
```

产物 `dist/zhishuxing.exe`（`--onefile --windowed`：平时无黑窗口，致命错误由
`packaging/exe_entry.py` 现场弹控制台说明）。行为要点：

- 图标取 `web/mobile/icon-512.png`，经 pillow 转成 `packaging/app.ico`，转换失败不阻断；
- `packaging/_build_version.txt` 写入 `pyproject.toml` 的 `version`，exe 首次运行据此决定
  内置资源是否重解压（版本戳一致则跳过，升级即重建）；
- 装了 `openai` 则真实 LLM 能力进包，没装则 exe 只有 Mock 模式（运行时如实标注）；
- `torch`/`tensorboard`/`mlagents_envs` 永远排除：训练链路需要 Unity 环境，exe 只做运行态。

## 两个已知的坑

1. **它需要 Pillow，而 Pillow 没有写进任何依赖清单。** 脚本 `from PIL import Image`，
   `pyproject.toml` 的依赖里没有 `pillow`(它在 dev 组)（复核：
   `python -c "import importlib.metadata as m; print([r for r in m.requires('zhishuxing') or []])"`）。
   本机装了 Pillow 所以跑得通，干净环境里要先 `pip install pillow`。这是待补的声明缺口，
   不要因为它"在谁机器上都能跑"就当它不存在。
2. **它会覆盖入库文件**，且没有 `--dry-run`。跑之前先 `git status` 确认工作区干净，
   跑完用 `git diff --stat` 看改动是否符合预期。

## 为什么不在 CI 里跑

CI（`.github/workflows/ci.yml`）只做 pyflakes + pytest。这个脚本改的是二进制资产，
跑它等于让 CI 产生待提交的内容，不合适；它属于"人改完品牌再手动执行一次"的工具。
静态检查覆盖它：门禁命令里的 `scripts/` 就是这一目录，脚本必须保持零告警。

## 和谁打交道

- **上游**：`render_brand_assets.py` 没有输入文件，星形与渐变是脚本里的硬编码常量，按
  `web/mobile/logo-mark.svg`
  复刻同一套几何（复核：`sed -n '1,7p' scripts/render_brand_assets.py` 的模块 docstring；
  两份 `logo-mark.svg` 目前逐字节相同）；`build_exe.py` 以整个已安装的 `zhishuxing` 包为输入，
  打包入口取 `packaging/exe_entry.py`。
- **下游**：`web/mobile/` 的 5 个 PNG 被 `manifest.webmanifest` 的 3 个 `icons`、`sw.js` 的
  `ASSETS` 表和页面本体引用；`src/zhishuxing/webapp/static/assets/favicon-32.png` 被
  `templates/index.html` 引用（复核：`grep -n favicon src/zhishuxing/webapp/templates/index.html`）。
- **改这里之后要跑**（仓库根执行）：

```bash
python scripts/render_brand_assets.py
git status --porcelain web/mobile src/zhishuxing/webapp/static/assets
python -m pyflakes src/ scripts/ tests/
python -m pytest
```

第二条用来看脚本改写了哪些入库文件；跑之前先按「两个已知的坑」第 2 条确认工作区是干净的。
后两条就是本仓门禁，脚本本身必须保持零告警。

## 别动

- `web/mobile/splash-logo.png` 与 `src/zhishuxing/webapp/static/assets/favicon-48.png`：全仓没有
  任何页面引用它们，看着最像能删 —— 但删了下次跑脚本又生成。真要去掉得删脚本里那两行
  `render_icon`（172、173）。复核（只命中这两行）：
  `grep -rn "splash-logo\|favicon-48" --include="*.py" --include="*.html" --include="*.js" --include="*.webmanifest" web src/zhishuxing scripts`。
- `render_icon(180, False, MOBILE / "apple-touch-icon.png")` 的 **180**：iOS 添加到主屏的尺寸约定，
  不要"顺手对齐"成清单里的 192。复核：`sed -n '168,174p' scripts/render_brand_assets.py`。
- 脚本顶部的 `SS = 4`（4 倍超采样）与 `LANCZOS` 缩放是一对：只降 `SS` 会让小尺寸图标的边缘起锯齿。
- `.github/workflows/ci.yml` 里 `python -m pyflakes src/ scripts/ tests/` 的 `scripts/` 一项：
  本目录只有两个文件，看着可以从门禁参数里删掉，删了它们就没有任何东西再检查这两个脚本能否 import。
