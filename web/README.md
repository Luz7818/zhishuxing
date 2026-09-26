# web/ —— 移动端 PWA 表层

> 用途：说明 `web/mobile/` 里每个文件管什么、它怎么被后端同源托管、哪些资源故意不入库。

这里只放**前端静态文件**，没有构建步骤：改完刷新页面即生效。整个目录由 Flask 以
`/mobile` 前缀同源提供（`webapp/app.py` 的 `mobile()` 与 `mobile_static()`），
所以 ServiceWorker 的 scope 能覆盖页面、也不需要考虑 CORS。

目录路径来自 `config.paths.mobile_dir`，也就是 **workspace 根下的 `web/mobile/`**。
用 `ZHISHUXING_WORKSPACE` 把 workspace 指到别处时，那边没有 `web/`，`/mobile` 会返回 404。

## 文件清单

| 文件 | 干什么 | 备注 |
|---|---|---|
| `mobile_app.html` | 单文件 PWA：对话式换乘引导、需求档案可视化、路线历史、深色模式 | 1885 行，样式与脚本内联；入口 URL 是 `/mobile` |
| `manifest.webmanifest` | 安装清单：名称、`start_url`、主题色 `#4f46e5`、3 个图标 | `start_url` 指向同目录的 `mobile_app.html` |
| `sw.js` | ServiceWorker：预缓存 9 个静态资源，`/api/` 与 `/outputs/` 走网络优先 | 缓存名 `zhishuxing-mobile-v5`，改了静态资源要一起升版本号 |
| `icon-192.png`、`icon-512.png`、`icon-maskable-512.png`、`apple-touch-icon.png`、`splash-logo.png` | 安装与启动屏图标 | **生成物**：`python scripts/render_brand_assets.py` |
| `logo-mark.svg`、`favicon.svg` | 矢量标识与标签页图标 | 手工维护，是上面 PNG 的设计来源 |
| `地图.png` | 「地图导航」模式下的站内示意图 | 被 `mobile_app.html` 与 `sw.js` 的预缓存列表引用 |

## 三个不入库的资源

| 文件 | 状态 | 为什么 |
|---|---|---|
| `AR.gif` | 被 `.gitignore` 的 `*.gif` 排除 | 「AR 实景导航」的演示素材，约 10 MB；缺失时页面显示"将 AR.gif 放入 web/mobile/ 即可启用"的占位说明而不是破图（复核：`web/mobile/mobile_app.html` 里的 `onerror` 分支） |
| `VR.png`、`图标.png` | 被 `.gitignore` 显式点名排除 | 旧版品牌大图，全仓无引用，已被 `render_brand_assets.py` 生成的图标替代 |

本机如果这三个文件在，`ls web/mobile` 会看到它们；`git ls-files web/mobile` 只有 11 个。
别把 `VR.png` 或 `图标.png` 再塞回清单，它们没有任何引用点。

## 后端不可用时的行为

`mobile_app.html` 只调两个接口：`POST /api/chat` 与 `GET /api/settings`。
请求失败时页面回退到内置的本地演示数据，并在状态栏写明「演示数据」与
「离线演示模式：未连接后端服务」，同时弹提示「后端不可用，已回退演示数据」。
这条降级路径是有意为之：PWA 装到手机上后要能脱离服务演示。

需求档案与路线历史存在浏览器 `localStorage`（键 `zsx_prefs`、`routeHistory`、`zsx_user`），
没有账号体系，换浏览器即清空。

## 怎么起

```bash
zhishuxing serve --host 127.0.0.1 --port 7860
# 本机浏览器打开 http://127.0.0.1:7860/mobile
```

要在另一台设备（真机）上试：`serve` 默认监听 `0.0.0.0`，用局域网地址访问即可；
但 iOS/Android 的系统「添加到主屏」对非 HTTPS 地址会限制 ServiceWorker，
真机验证以 Chrome + `http://<本机IP>:7860/mobile` 为准。

## 改这里之后要跑

```bash
python -m pytest tests/test_api.py -k mobile -o addopts="" -q
```

该用例只断言 `/mobile` 与 `/mobile/<file>` 返回 200，页面内部行为没有自动化覆盖，
需要手工在浏览器里过一遍对话与安装流程。
