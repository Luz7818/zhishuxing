"""构建私有化离线交付包:python scripts/build_deploy.py

产物 dist/zhishuxing-deploy-<版本>.zip,内含:
  zhishuxing-<版本>-image.tar.gz   docker save 出的镜像(目标机 docker load 免网络)
  docker-compose.yml               部署编排(镜像名与 tar 对号入座)
  .env.example                     密钥与管理口令配置模板
  LICENSE / eula-template.md / DEPLOY-PRIVATE.md / 交付说明.md

前置:本机 docker 守护进程在运行。镜像按 Dockerfile 清单最小 COPY(不含源码
仓库的 scripts/tests/docs),签发工具 make_license.py 留在厂商侧、绝不进交付包。
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DELIVERY_NOTE = """# 智枢星 私有化部署包 · 交付说明

这是离线交付包,目标机**不需要外网与 Python**,只需要 Docker。

## 三步起服务

1. `docker load -i zhishuxing-<版本>-image.tar.gz`
2. `cp .env.example .env`,按其中注释填入高德/LLM 密钥与 `ADMIN_PASSWORD`(管理端口令)
3. `docker compose up -d`,浏览器打开 `http://<服务器IP>:7860`(乘客端 `/mobile`)

## 验收

```bash
docker compose exec zhishuxing zhishuxing verify --url http://127.0.0.1:7860
```

全部检查 PASS(允许 WARN)即部署合格;报告在容器 /app/data/outputs,可用
`http://<服务器IP>:7860/outputs/<报告文件名>` 下载归档。

## 授权

未放置 `license.lic` 时为试用模式(全功能可用,界面标注"试用版")。商业授权文件
由厂商签发:放到本目录、在 docker-compose.yml 里取消挂载行注释后 `docker compose up -d`。

详细步骤、升级与回滚、排障见 **DEPLOY-PRIVATE.md**;许可条款见 LICENSE 与 eula-template.md。
"""


def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    print("[build_deploy]", subprocess.list2cmdline(cmd))
    return subprocess.run(cmd, **kwargs)


def main() -> int:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    image = f"zhishuxing:{version}"

    try:
        _run(["docker", "version", "--format", "{{.Server.Version}}"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[build_deploy] 需要 docker 守护进程在运行(检测失败: {exc})")
        return 2

    stage = ROOT / "dist" / "docker" / f"zhishuxing-deploy-{version}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    build = _run(["docker", "build", "-t", image, str(ROOT)], cwd=ROOT)
    if build.returncode != 0:
        print("[build_deploy] docker build 失败(若报非 ASCII 路径错误:subst X: <项目所在盘符路径> 后从 X: 构建)")
        return build.returncode

    image_tar = stage / f"zhishuxing-{version}-image.tar.gz"
    with gzip.open(image_tar, "wb", compresslevel=6) as fh:
        save = _run(["docker", "save", image], stdout=fh)
    if save.returncode != 0:
        return save.returncode

    compose_text = (ROOT / "packaging" / "deploy-compose.yml").read_text(encoding="utf-8").replace("{VERSION}", version)
    (stage / "docker-compose.yml").write_text(compose_text, encoding="utf-8")
    shutil.copyfile(ROOT / ".env.example", stage / ".env.example")
    shutil.copyfile(ROOT / "LICENSE", stage / "LICENSE")
    docs = ROOT / "docs"
    shutil.copyfile(docs / "eula-template.md", stage / "eula-template.md")
    shutil.copyfile(docs / "DEPLOY-PRIVATE.md", stage / "DEPLOY-PRIVATE.md")
    (stage / "交付说明.md").write_text(DELIVERY_NOTE, encoding="utf-8")

    zip_path = ROOT / "dist" / f"zhishuxing-deploy-{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(stage.parent))
    print(f"[build_deploy] 交付包: {zip_path} ({zip_path.stat().st_size / 1048576:.0f} MB)")
    print("[build_deploy] 镜像工作目录(含未压缩清单):", stage)
    print("[build_deploy] 下一步:把 zip 发给客户,按包内 DEPLOY-PRIVATE.md 部署。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
