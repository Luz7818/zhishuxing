# 智枢星 · 单容器部署
# 构建:docker build -t zhishuxing .
# 运行:见 docker-compose.yml(推荐)或单条:
#   docker run -d -p 7860:7860 -v zhishuxing_data:/app/data --env-file .env zhishuxing
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# 非 root 运行(UID 1000,与宿主机挂载卷权限对齐)
RUN useradd --create-home --uid 1000 zsx

WORKDIR /app

# 先装依赖(层缓存:代码变更不触发重装)
# 境内构建 pypi.org 常被限速:docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# 源码与前端资产(package-data 已含 webapp/templates 与 static)
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY data/transfer_kb ./data/transfer_kb
COPY configs ./configs
# .[llm] 把 openai 装进镜像:缺它时适配器 import 失败会静默降级 Mock,
# .env 里配了 SILICONFLOW_API_KEY 也不生效
RUN pip install --no-cache-dir ".[llm]"

# 运行期可写目录:报告图 / 模型 / 会话库(挂卷持久化)
RUN mkdir -p data/outputs data/model data/runs && chown -R zsx:zsx /app
USER zsx

# 内置枢纽导航图与前端资产就位即可对外服务
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health',timeout=4).status==200 else 1)"

# waitress 生产托管;密钥经 .env / 环境变量注入(见 docs/getting-started.md 第 2 节)
CMD ["python", "-c", "from zhishuxing.cli import main;import sys;sys.argv=['zhishuxing','serve','--host','0.0.0.0','--port','7860','--production'];main()"]
