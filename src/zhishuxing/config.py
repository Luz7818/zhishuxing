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


def env_file_path() -> Path:
    """当前生效的 .env 路径（锚定 workspace 根目录，不随进程 CWD 变化）。"""
    return WORKSPACE_ROOT / ".env"


def parse_env_text(text: str) -> Dict[str, str]:
    """按项目约定解析 .env 文本：只认 KEY=VALUE 与整行 # 注释，值去空白与外层引号。"""
    parsed: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        parsed[key] = value.strip().strip('"').strip("'")
    return parsed


def _load_dotenv(env_file: Optional[Path] = None, override: bool = False) -> None:
    """加载 .env（键=值/行注释）。

    项目不依赖 python-dotenv：仅支持 KEY=VALUE 与 # 注释两种行，满足本地密钥配置需求。
    默认读取 workspace 根目录的 .env。

    override=False（默认，也是 import 期唯一用法）：保持原有语义，**不覆盖已存在的环境变量**，
    系统环境变量始终优先于 .env。
    override=True：仅供「设置」页保存 .env 后的热重载调用，用文件内容强制刷新进程环境变量，
    否则改完 .env 不重启服务就看不到新值。
    """
    env_file = env_file or env_file_path()
    if not env_file.exists():
        return
    try:
        parsed = parse_env_text(env_file.read_text(encoding="utf-8"))
    except OSError:
        return
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value


def reload_env(env_file: Optional[Path] = None) -> Dict[str, str]:
    """强制按 .env 内容刷新进程环境变量（写入配置后的热重载入口）。

    与默认加载的唯一区别是 override=True：只影响 .env 里出现的键，
    .env 未涉及的环境变量一律不动，因此不会牵连系统里的其他配置。
    返回解析后的键值，供调用方核对结果。
    """
    env_file = env_file or env_file_path()
    if not env_file.exists():
        return {}
    _load_dotenv(env_file, override=True)
    return parse_env_text(env_file.read_text(encoding="utf-8"))


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

_load_dotenv()


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

    @property
    def kb_dir(self) -> Path:
        return self.data / "transfer_kb"

    @property
    def kb_corpus(self) -> Path:
        """入库后的换乘经验语料(JSONL)。"""
        return self.kb_dir / "corpus.jsonl"

    @property
    def kb_sources_dir(self) -> Path:
        """手工整理的换乘经验源文档目录(txt/md/html,kb-ingest 的默认输入)。"""
        return self.kb_dir / "shenzhen_north"

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
# 一律「空值视同未配置」：控制台「设置」视图会把清空的项写成 KEY=（空串），
# 若用 os.environ.get(key, default) 取默认值，空串会被当成有效值而绕掉兜底逻辑。


def _env_or_default(key: str, default: str) -> str:
    return os.environ.get(key) or default


def _env_timeout() -> float:
    """AMAP_TIMEOUT 可能被手改成非数字：回落默认值，而不是让整个控制台 500。"""
    try:
        return float(_env_or_default("AMAP_TIMEOUT", "10"))
    except ValueError:
        return 10.0


def siliconflow_config() -> Dict[str, Optional[str]]:
    """SiliconFlow（OpenAI 兼容）LLM 服务配置。"""
    return {
        "api_key": os.environ.get("SILICONFLOW_API_KEY") or None,
        "base_url": _env_or_default("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "model": _env_or_default("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    }


def amap_config() -> Dict[str, Optional[str]]:
    """高德开放平台配置：Web 服务 Key（REST）与 JS Key 分开管理。

    security_code 为 JS API 的安全密钥（2021-12 后申请的 Key 需要配合使用）。
    JS Key 留空时回退复用 REST Key（与 settings.effective_value 的判定保持一致）。
    """
    rest_key = os.environ.get("AMAP_REST_KEY") or None
    return {
        "rest_key": rest_key,
        "js_key": os.environ.get("AMAP_JS_KEY") or rest_key,
        "security_code": os.environ.get("AMAP_SECURITY_CODE") or None,
        "timeout": _env_timeout(),
    }
