"""集中管理路径、密钥与运行参数。

约定：
- 所有产物路径锚定 workspace 根目录，不再依赖进程 CWD；
- 所有密钥一律走环境变量，代码中不存在任何真实密钥；
- 路径既支持以包源码定位（开发/安装后运行），也支持 ZHISHUXING_WORKSPACE 覆盖。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _find_workspace_root() -> Path:
    """定位工作区根目录：优先环境变量，其次从包源码位置向上推导（src/zhishuxing -> 根）。"""
    override = os.environ.get("ZHISHUXING_WORKSPACE")
    if override:
        return Path(override).resolve()
    package_root = Path(__file__).resolve().parent
    candidate = package_root.parent.parent
    if (candidate / "configs").exists():
        return candidate
    return Path.cwd().resolve()


WORKSPACE_ROOT = _find_workspace_root()
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class Paths:
    root: Path = field(default_factory=lambda: WORKSPACE_ROOT)
    configs: Path = field(default_factory=lambda: WORKSPACE_ROOT / "configs")
    data: Path = field(default_factory=lambda: WORKSPACE_ROOT / "data")
    webapp_dir: Path = field(default_factory=lambda: PACKAGE_DIR / "webapp")
    mobile_dir: Path = field(default_factory=lambda: WORKSPACE_ROOT / "web" / "mobile")

    @property
    def samples(self) -> Path:
        return self.data / "samples"

    @property
    def outputs(self) -> Path:
        return self.data / "outputs"

    @property
    def model_dir(self) -> Path:
        return self.data / "model"

    @property
    def runs_dir(self) -> Path:
        return self.data / "runs"

    @property
    def navigation_config(self) -> Path:
        return self.configs / "hub_default.json"

    @property
    def scenarios_config(self) -> Path:
        return self.configs / "scenarios.json"

    @property
    def training_config(self) -> Path:
        return self.configs / "training.json"

    def ensure_runtime_dirs(self) -> None:
        for directory in (self.outputs, self.model_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)


paths = Paths()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_training_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_json(path or paths.training_config)


def load_scenarios(path: Optional[Path] = None) -> Dict[str, Any]:
    return load_json(path or paths.scenarios_config)


# ---------------------------------------------------------------- 密钥（环境变量，绝无默认值）

def siliconflow_config() -> Dict[str, Optional[str]]:
    """SiliconFlow（OpenAI 兼容）LLM 服务配置。"""
    return {
        "api_key": os.environ.get("SILICONFLOW_API_KEY"),
        "base_url": os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "model": os.environ.get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    }


def amap_config() -> Dict[str, Optional[str]]:
    """高德开放平台配置：Web 服务 Key（REST）与 JS Key 分开管理。"""
    return {
        "rest_key": os.environ.get("AMAP_REST_KEY"),
        "js_key": os.environ.get("AMAP_JS_KEY", os.environ.get("AMAP_REST_KEY")),
        "timeout": float(os.environ.get("AMAP_TIMEOUT", "10")),
    }
