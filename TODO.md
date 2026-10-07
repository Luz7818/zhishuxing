# TODO —— 开发计划与当前进度

> 用途：给开发者与 AI。当前在做哪个任务、每个任务按什么标准验收、接下来做什么。
> 完成一项就勾一项并写明下一项；历史性记录写进 [HISTORY.md](HISTORY.md)，这里只留计划。

## 当前进度

- 正在做：无（2.5.0 可发售私有化 v1 已落地并通过 Docker 真包实测，见 HISTORY.md 对应条目）
- 下一个：多枢纽接入工具（任务 6）

## 任务计划

| # | 任务 | 验收标准（可验证） | 状态 |
|---|---|---|---|
| 1 | Docker 部署首次实测（2.3 建了 Dockerfile/compose，本机无 Docker，镜像内契约只做过静态校验） | 按 `docs/DEPLOY.md` 完成 `docker compose up --build`，`/health` 通过、控制台与 `/mobile` 可用 | 完成（2026-10-07，随任务 4 一并实测：镜像构建 + compose up 25s healthy + `/health`、`/`、`/mobile` 全 200） |
| 2 | 移动端 PWA 真机走查（本机浏览器已过，真机未验） | 真机上对话流/站内导航/主题切换各验一遍，结论登记 HISTORY | 待开始 |
| 3 | `unity/` 路线 3.0 的录屏判据在真实 Unity 环境验证 | 按路线 3.0 判据录屏一次并归档（场景由使用者自建） | 待开始 |
| 4 | 私有化交付包在有 Docker 的机器实测 | `python packaging/build_deploy.py` 出真包 → 目标机 `docker load` + `compose up -d` → 容器内 `verify --url` 全 PASS（允许 WARN），含 license 激活与 ADMIN_PASSWORD 登录各走一遍 | 完成（2026-10-07，本机 Docker Desktop 29.8.2，详见 HISTORY 2.5.0 验证节） |
| 5 | 授权签名升级为非对称（Ed25519） | 签发私钥离厂商环境保管、交付物只含公钥校验；`licensing.py` 与 `make_license.py` 同步改造，存量 license 兼容策略写明 | 待开始 |
| 6 | 多枢纽接入工具（产品深度） | 新枢纽平面图/CAD → 导航图 JSON 的导入链路 + 文档，两套以上枢纽数据通过全量测试 | 待开始 |
| 7 | C 端小程序（第一个枢纽客户落地后） | 复用姊妹项目 miniprogram 架构对接 `/api/chat`、`/api/plan`，登录与管理动作不进小程序 | 待开始 |

状态取值：待开始 / 进行中 / 待确认 / 完成。

## 完成记录

- [x] 移动端 PWA 重写 1740 行落库 + 本机大文件清理约 220 MB —— 2026-10-05（核查报告整理轮）
- [x] 文档九件体系迁移 —— 2026-10-05（见 HISTORY.md 对应条目）
