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
| `mobile_app.html` | 单文件 PWA：对话流（气泡消息/需求档案/经验引用/动作信封）、结构化路线、站内导航 Canvas、客流提示、三态深色模式 | 2058 行（复核：`wc -l web/mobile/mobile_app.html`），样式与脚本内联；入口 URL 是 `/mobile` |
| `manifest.webmanifest` | 安装清单：名称、`start_url`、主题色 `#4f46e5`、3 个图标 | `start_url` 指向同目录的 `mobile_app.html` |
| `sw.js` | ServiceWorker：预缓存 9 个静态资源，`/api/` 与 `/outputs/` 走网络优先 | 缓存名 `zhishuxing-mobile-v8`，改了静态资源要一起升版本号 |
| `icon-192.png`、`icon-512.png`、`icon-maskable-512.png`、`apple-touch-icon.png`、`splash-logo.png` | 安装与启动屏图标 | **生成物**：`python scripts/render_brand_assets.py` |
| `logo-mark.svg`、`favicon.svg` | 矢量标识与标签页图标 | 手工维护，是上面 PNG 的设计来源 |
| `map.png` | 「地图导航」模式下的站内示意图（原中文名 `地图.png`，2026-10-07 随文件名 ASCII 化改为现名） | 被 `mobile_app.html` 与 `sw.js` 的预缓存列表引用 |

## 子目录

| 子目录 | 负责 |
|---|---|
| `mobile/` | 移动端 PWA 的全部文件：页面本体、清单、ServiceWorker、图标与站内示意图。本目录的另一半 |

`web/` 下只有 `mobile/` 这一个二级目录，没有构建产物目录（复核：
`find web -mindepth 1 -maxdepth 1 -type d`）。磁盘上 `web/mobile/` 是 12 个文件，入库 11 个，
差额就是下面「不入库的资源」里的 `AR.gif`（复核：`ls web/mobile | wc -l` 与
`git ls-files web/mobile | wc -l`）。

## 不入库的资源

| 文件 | 状态 | 为什么 |
|---|---|---|
| `AR.gif` | 被 `.gitignore` 的 `*.gif` 排除 | 「AR 实景导航」的演示素材，约 10 MB；缺失时页面显示"将 AR.gif 放入 web/mobile/ 即可启用"的占位说明而不是破图（复核：`web/mobile/mobile_app.html` 里的 `onerror` 分支） |
| `VR.png`、`图标.png` | 已从磁盘删除；`.gitignore` 仍显式点名排除 | 旧版品牌大图，全仓无引用，已被 `render_brand_assets.py` 生成的图标替代；留着排除规则是防止它们再被塞回目录 |

本机如果 `AR.gif` 在，`ls web/mobile` 会看到它；`git ls-files web/mobile` 没有它。
别把 `VR.png` 或 `图标.png` 再塞回清单，它们没有任何引用点。

## 后端不可用时的行为

`mobile_app.html` 调 6 个接口：`POST /api/chat`（+`/api/chat/reset` 新会话）、`GET /api/settings`、
`GET /api/navigation/grid`、`POST /api/navigation/plan`、`GET /api/scenarios`、`POST /api/dashboard/run`。
对话与配置请求失败时页面回退到内置的本地演示数据，在消息流里挂「演示数据」横幅，
同时弹提示「后端不可用，已回退演示数据」；站内导航与客流卡失败则就地显示错误或静默收起（非关键路径）。
这条降级路径是有意为之：PWA 装到手机上后要能脱离服务演示。

会话、需求档案与路线历史存在浏览器 `localStorage`（键 `zsx_prefs`、`routeHistory`、`zsx_user`、
`zsx_session`、`zsx_chat_log`、`zsx_theme`；服务端会话另有 SQLite 持久化），
没有账号体系，换浏览器即清空。

## 怎么起

```bash
zhishuxing serve --host 127.0.0.1 --port 7860
# 本机浏览器打开 http://127.0.0.1:7860/mobile
```

要在另一台设备（真机）上试：`serve` 默认监听 `0.0.0.0`，用局域网地址访问即可；
但 iOS/Android 的系统「添加到主屏」对非 HTTPS 地址会限制 ServiceWorker，
真机验证以 Chrome + `http://<本机IP>:7860/mobile` 为准。

## 和谁打交道

- **上游**：Flask 用 `GET /mobile` 与 `GET /mobile/<path:filename>` 把本目录当静态根直接送出去，
  根路径取自 `cfg.paths.mobile_dir`（复核：`grep -n "mobile" src/zhishuxing/webapp/app.py`）；
  5 个图标 PNG 来自 `python scripts/render_brand_assets.py`。
- **下游**：装了 PWA 的手机浏览器。页面调 6 个接口：`POST /api/chat`（+`/api/chat/reset`）、
  `GET /api/settings`、`GET /api/navigation/grid`、`POST /api/navigation/plan`、`GET /api/scenarios`、
  `POST /api/dashboard/run`（复核：`grep -noE "'/api/[a-z/]*'" web/mobile/mobile_app.html`），
  不读 `data/outputs/` 里的产物图（复核：`grep -c "outputs" web/mobile/mobile_app.html` 输出 `0`）。
- **改这里之后要跑**：

```bash
python -m pytest tests/test_api.py -k mobile -o addopts="" -q
```

该用例只断言 `/mobile` 与 `/mobile/<file>` 返回 200，页面内部行为没有自动化覆盖，
需要手工在浏览器里过一遍对话与安装流程。

## 别动

- `map.png`（2026-10-07 前为中文名 `地图.png`，改名时三处引用同步）：`mobile_app.html`
  第 746 行的 `<img>`、第 1581 行的 `navImage.src` 赋值，加上 `sw.js` 第 5 行的预缓存项，
  **三处**都写死了它。再改这个文件名要同时改三处并升 `CACHE_NAME`，
  漏一处的表现是破图或 SW 缓存 miss。复核：`grep -n "map.png" web/mobile/mobile_app.html web/mobile/sw.js`。
- `sw.js` 第 1 行的 `CACHE_NAME`（当前 `zhishuxing-mobile-v8`）：增删静态资源不升这个版本号，
  已经装到手机上的 PWA 会一直命中旧缓存，表现是"改了没生效"。复核：`sed -n '1p' web/mobile/sw.js`。
- `AR.gif`：本机文件、不入库，却被第 1581 行当正常资源引用，靠 `onerror` 分支（1571–1579 行）
  出占位说明。删本机那个文件没影响，删 `onerror` 那段就是新克隆破图。复核：
  `sed -n '1570,1582p' web/mobile/mobile_app.html`。
- `tests/test_api.py` 第 260 行的 `assert "VR.png" not in sw_text`：`VR.png`、`图标.png`
  这两张本机图唯一的"引用点"就是这条**反向**断言，防止它们被重新塞进预缓存清单。
  别把它当冗余断言删掉。
- 5 个图标 PNG 与 `logo-mark.svg`、`favicon.svg`：前者是脚本产物（别手改，改设计要回脚本），
  后两者是手工维护的设计母版，删了 PNG 就失去对照物。复核：
  `git ls-files web/mobile` 能看到它们都在跟踪列表里。
