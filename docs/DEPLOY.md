# 部署指南(2.3 起以 Docker 为标准路线)

> 离线私有化交付（交付包 `zhishuxing-deploy-<版本>.zip`、目标机免外网、部署验收、授权激活、
> 升级回滚）见 **[DEPLOY-PRIVATE.md](DEPLOY-PRIVATE.md)**;本篇讲从源码出发的部署路线。

## 路线 A:Docker(推荐)

前置:安装 Docker;`.env` 按 `.env.example` 填好密钥(全部留空也能跑,全链路离线降级)。

```bash
docker compose up -d --build
docker logs -f zhishuxing        # 看到 "[llm] 已自动挂载真实适配器" 或 Mock 提示即就绪
curl http://127.0.0.1:7860/health
```

- 服务只监听 `127.0.0.1:7860`,公网请前置 Nginx/Caddy 反代 + HTTPS(证书用 certbot 自动签续);
- 数据都在卷里(`outputs` / `model` / `runs`),镜像重建不丢;**会话库 `sessions.db`
  在 `runs` 卷中,服务重启后对话历史照常恢复**(30 天不活跃自动清理);
- 更新 = `git pull && docker compose up -d --build`;回滚 = 换镜像重建;
- **单实例边界**:进程内锁不跨进程,`docker compose` 里不要 scale 副本(AGENTS 约定)。

## 路线 B:裸机(内网演示)

```bash
pip install -e .            # Windows 先 set PYTHONIOENCODING=utf-8
zhishuxing serve --port 7860            # 默认 127.0.0.1;局域网用 --host 0.0.0.0 --production
```

生产托管 waitress 由 `--production` 启用;密钥写接口按监听地址裁决(非回环默认关闭,
逃生门 `--allow-remote-settings`),详见 `zhishuxing doctor` 输出。

## 移动端 PWA 上线(微信)

HTTPS + 备案域名 + 小程序后台配置 `request` 合法域名,三项缺一不可;
`/mobile` 同源托管,反代后即可真机安装。

## 健康与巡检

`GET /health` 200 即在线;卷挂载核对:`docker compose exec zhishuxing ls -la /app/data`。
LLM 状态:启动日志 `[llm]` 行(自动挂载成功 / Mock 原因)。

## 备份

会话与运行产物在卷里,备份即备份卷:

```bash
docker run --rm -v zhishuxing_runs:/data -v %cd%:/backup alpine tar czf /backup/runs.tgz -C /data .
```
