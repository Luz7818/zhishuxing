# 私有化离线部署指南（DEPLOY-PRIVATE）

> 用途：拿到 `zhishuxing-deploy-<版本>.zip` 交付包的人。目标机**只需要 Docker**，
> 不需要外网、不需要 Python。云服务器/内网穿透等在线路线见 [DEPLOY.md](DEPLOY.md)，
> 本指南只讲离线私有化交付。命令在交付包解压目录里执行。

## 交付包内容

| 文件 | 用途 |
|---|---|
| `zhishuxing-<版本>-image.tar.gz` | `docker save` 导出的镜像，`docker load` 免网络导入 |
| `docker-compose.yml` | 部署编排，镜像名与 tar 对号入座 |
| `.env.example` | 密钥与管理口令配置模板（复制为 `.env` 后填写） |
| `LICENSE` / `eula-template.md` | 许可条款（正式条款以双方签署的合同为准） |
| `交付说明.md` | 一页速览：三步起服务 |
| `DEPLOY-PRIVATE.md` | 本指南 |

## 目标机要求

- Linux x86_64（或任何 Docker 可用主机），Docker ≥ 20 与 Docker Compose v2；
- 磁盘 ≥ 2 GB（镜像约 400 MB，解压后翻倍；数据卷随产出增长）；
- 内存 ≥ 2 GB（客流面板/报告渲染是 CPU 密集型瞬时任务）。

## 首次部署（五步）

1. **导入镜像**：`docker load -i zhishuxing-<版本>-image.tar.gz`
2. **配置 `.env`**：`cp .env.example .env` 后按注释填写——
   - 必填 3 项：`AMAP_REST_KEY`、`AMAP_JS_KEY`、`AMAP_SECURITY_CODE`（不填则对应在线能力降级，
     离线演示不受影响，与控制台/`/mobile` 的降级标注一致）；
   - 可选 4 项：LLM 三项（不配走 Mock 确定性模板）与 `AMAP_TIMEOUT`；
   - **强烈建议设置 `ADMIN_PASSWORD`**（管理端登录口令，见下文安全模型）。
3. **（可选）挂载授权**：未放置 `license.lic` 时为试用模式（全功能可用，界面标注"试用版"）。
   拿到授权文件后放到 compose 同目录，取消 `docker-compose.yml` 中挂载行的注释；
4. **启动**：`docker compose up -d`，健康检查就绪后（约 20 秒）浏览器打开 `http://<服务器IP>:7860`；
5. **验收**：
   ```bash
   docker compose exec zhishuxing zhishuxing verify --url http://127.0.0.1:7860
   ```
   逐项 `PASS`（允许 `WARN`：试用模式、可选密钥缺失、镜像未随附参考样本）即部署合格，
   末行 `VERIFY PASS`；markdown 验收报告写进容器 `/app/data/outputs/`，
   用 `http://<服务器IP>:7860/outputs/<报告文件名>` 下载归档签字。

## 端口与访问形态

- **Web 控制台** `http://<IP>:7860/`：管理动作（换导航图/微调/RL 权重/跑报告/改配置）需要
  管理员会话；未登录时后端返回 401，登录入口在 `/admin/login`；
- **乘客端** `http://<IP>:7860/mobile`：免登录。PWA「安装」需要 HTTPS——纯 HTTP 内网先
  用浏览器书签/添加到主屏幕，HTTPS 由前置反代终止后自动恢复安装能力；
- 生产建议：compose 的 `ports` 收窄为 `"127.0.0.1:7860:7860"`，由本机 Nginx 反代 + HTTPS 对外
  （参考 [DEPLOY.md](DEPLOY.md) 路线 B 的 Nginx 配置）。

## 安全模型（照 `src/zhishuxing/webapp/auth.py` 实现，不要凭印象改）

- **口令必须由部署者自己提供**：`.env` 里设置 `ADMIN_PASSWORD` 即启用管理端鉴权；
  未设置 = 鉴权关闭，管理端点对所在网络开放——仅限本机或纯内网演示，生产部署必设；
- **分域而不是整站登录**：只有管理动作端点（换导航图、微调、RL 权重、跑报告、改配置）
  要登录；乘客对话/规划、`/mobile`、演示只读接口永远公开，`/health` 契约不变；
- 口令 PBKDF2-SHA256 校验（不落盘），会话为 HMAC 签名令牌（Cookie 与 `Authorization: Bearer`
  是同一枚，12 小时有效），登录失败 5 次锁 10 分钟（进程内计数；单进程部署成立，多实例请在
  最外层网关限流）；
- `POST /api/settings` 另有 loopback / 生产模式双闸，管理端鉴权不放宽它；
- 授权(license)是另一层：HMAC 签名 + 有效期，过期超 14 天宽限后服务拒绝启动（见排障）。

## 升级与回滚

- **升级**：`docker load` 新版本 tar → 把 `docker-compose.yml` 的 `image:` 改成新 tag →
  `docker compose up -d`。三个数据卷（outputs/model/runs）不动，会话、报告、RL 权重全保留；
- **回滚**：保留上一版 tar（或 tag），`image:` 改回旧版本 → `docker compose up -d`。
  镜像版本与 compose 一一对应，这是「换 tag 即回滚」的前提。

## 排障

| 现象 | 处置 |
|---|---|
| 容器反复重启/退出 | `docker logs zhishuxing`；退出码 3 = 授权过期超宽限，`docker compose exec zhishuxing zhishuxing license` 看详情，续期后重新挂载 `license.lic` |
| healthcheck unhealthy | `curl http://127.0.0.1:7860/health` 看返回；启动期 20 秒内未就绪属正常 |
| 改了 `.env` 不生效 | `env_file` 在容器创建时注入：`docker compose up -d`（重建容器），`restart` 不够 |
| 管理操作提示需要登录 | 浏览器开 `/admin/login` 登录；脚本/CI 走登录接口拿 token 后带 `Authorization: Bearer <令牌>` |
| 忘记 ADMIN_PASSWORD | 改 `.env` 里的值后 `docker compose up -d` 重建（口令随环境变量注入，改完即换） |
| 授权状态查看 | `docker compose exec zhishuxing zhishuxing license` |
