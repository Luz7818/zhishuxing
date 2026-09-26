# scripts/ —— 仓库级工具脚本

> 用途：说明这个目录里唯一的脚本做什么、产物落在哪、为什么它不在 CI 里跑。

本目录只放**不参与运行时**的开发者工具。所有能被 `zhishuxing` 子命令做到的事都在这里，
不要往这个目录加功能脚本。

## 文件清单

| 文件 | 干什么 | 备注 |
|---|---|---|
| `render_brand_assets.py` | 从矢量设计参数直接绘制品牌 PNG：PWA 图标与 favicon | 入口脚本，无参数 |

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

## 两个已知的坑

1. **它需要 Pillow，而 Pillow 没有写进任何依赖清单。** 脚本 `from PIL import Image`，
   `pyproject.toml` 的 `dependencies` 与 `requirements.txt` 里都没有 `pillow`（复核：
   `python -c "import importlib.metadata as m; print([r for r in m.requires('zhishuxing') or []])"`）。
   本机装了 Pillow 所以跑得通，干净环境里要先 `pip install pillow`。这是待补的声明缺口，
   不要因为它"在谁机器上都能跑"就当它不存在。
2. **它会覆盖入库文件**，且没有 `--dry-run`。跑之前先 `git status` 确认工作区干净，
   跑完用 `git diff --stat` 看改动是否符合预期。

## 为什么不在 CI 里跑

CI（`.github/workflows/ci.yml`）只做 pyflakes + pytest。这个脚本改的是二进制资产，
跑它等于让 CI 产生待提交的内容，不合适；它属于"人改完品牌再手动执行一次"的工具。
静态检查覆盖它：门禁命令里的 `scripts/` 就是这一目录，脚本必须保持零告警。
