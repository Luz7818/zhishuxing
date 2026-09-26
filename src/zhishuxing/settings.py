"""密钥配置的托管读写：注册表 + 状态掩码 + 写入校验 + .env 原子更新与热重载。

设计约束（与 README 的「无密钥也能全链路演示」承诺绑定，不可退化）：
- 对外响应**一律掩码**，任何接口/CLI 输出都不回显密钥明文；
- 写入 workspace 根目录 .env 前先备份、写入时原子替换，值拒绝换行与等号注入；
- 本模块只负责「配置」，不改变任何降级判断：留空时 LLM 仍走 Mock、地图仍回退 Canvas、
  规划仍走内置枢纽引擎，所有降级逻辑都在各自的适配器里。
"""

from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import config as cfg

# 单个值长度上限：密钥/URL 都远小于此，超出即视为误粘贴或攻击载荷
MAX_VALUE_LENGTH = 400
# .env 备份文件（被 .gitignore 的 .env.* 规则覆盖，不会入库）
BACKUP_NAME = ".env.bak"


class SettingError(ValueError):
    """配置写入的可预期失败：调用方按 400 返回 detail 即可，不当作服务端异常。"""


@dataclass(frozen=True)
class SettingItem:
    """一个托管配置项。文案同时供「设置」视图、doctor 子命令与移动端引导复用。"""

    key: str
    label: str
    purpose: str
    apply_entry: str
    degrades_to: str
    # required=True 表示该项属于「就绪判定」范围：缺失会让 doctor 退出码为 1
    # 并在界面上列为待办。留空不影响离线演示，只影响对应的在线能力。
    required: bool = True
    default: str = ""  # 留空时实际生效的默认值（仅用于提示，不参与降级判断）
    derived_from: str = ""  # 留空时可复用的兄弟项（如 JS Key 回退到 REST Key）
    numeric: bool = False  # 值必须是数字（如超时秒数），避免手滑写坏运行参数


SETTINGS: Tuple[SettingItem, ...] = (
    SettingItem(
        key="AMAP_REST_KEY",
        label="高德 Web 服务 Key",
        purpose="后端地理编码 + 公交换乘规划，决定 /api/plan 与 /api/chat 能否出真实路线。",
        apply_entry="https://lbs.amap.com/ → 控制台 → 应用管理 → 添加 Key，服务平台选「Web服务」。",
        degrades_to="真实路线规划不可用：引擎自动改用内置枢纽引擎，市际 OD 给出话术提示而不是报错。",
    ),
    SettingItem(
        key="AMAP_JS_KEY",
        label="高德 JS API Key",
        purpose="控制台路线规划页的真实地图底图（浏览器侧使用，属前端注入项）。",
        apply_entry="同一应用下再加一个 Key，服务平台必须选「Web端(JS API)」，与 Web服务 Key 不通用。",
        degrades_to="地图回退 Canvas 离线折线示意，并标注「未配置 AMAP_JS_KEY」。",
        derived_from="AMAP_REST_KEY",
    ),
    SettingItem(
        key="AMAP_SECURITY_CODE",
        label="高德 JS API 安全密钥",
        purpose="与 JS Key 配对的前端鉴权码；2021-12 之后申请的 Key 缺它就无法初始化底图。",
        apply_entry="JS Key 所在应用详情页的「安全密钥」字段，与 JS Key 成对下发，必须一起填。",
        degrades_to="高德底图初始化失败，前端捕获后同样回退 Canvas 折线（不影响规划结果）。",
    ),
    SettingItem(
        key="SILICONFLOW_API_KEY",
        label="SiliconFlow API Key",
        purpose="LLM 对话合成 / 需求档案解析 / OD 提取（OpenAI 兼容端点）。",
        apply_entry="https://cloud.siliconflow.cn/ → 登录 → 账号 → API 密钥 → 新建密钥。",
        degrades_to="走 MockLLMAdapter 确定性模板：规则关键词仍能解析需求档案，全链路离线可演示。",
        # 离线演示的兜底就是 Mock，因此不计入必需项；doctor --strict 会把它算作缺失
        required=False,
    ),
    SettingItem(
        key="SILICONFLOW_BASE_URL",
        label="LLM 兼容端点",
        purpose="OpenAI 兼容服务地址，可指向自建 vLLM / Ollama 等。",
        apply_entry="按所选服务商提供的 OpenAI 兼容 baseURL 填写，注意以 /v1 结尾。",
        degrades_to="使用内置默认端点（SiliconFlow 官方）。",
        required=False,
        default="https://api.siliconflow.cn/v1",
    ),
    SettingItem(
        key="SILICONFLOW_MODEL",
        label="LLM 默认模型",
        purpose="未在前端指定模型 ID 时使用的默认模型。",
        apply_entry="填服务商侧有调用权限的模型 ID；控制台对话面板里手填的模型 ID 优先级更高。",
        degrades_to="使用内置默认模型名（deepseek-ai/DeepSeek-R1-Distill-Qwen-7B）。",
        required=False,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ),
    SettingItem(
        key="AMAP_TIMEOUT",
        label="高德请求超时（秒）",
        purpose="后端调用高德接口的超时秒数，弱网可适当调大。",
        apply_entry="无需申请，按网络情况填一个正数即可。",
        degrades_to="使用默认 10 秒。",
        required=False,
        default="10",
        numeric=True,
    ),
)

MANAGED_KEYS: Tuple[str, ...] = tuple(item.key for item in SETTINGS)


SETTINGS_BY_KEY: Dict[str, SettingItem] = {item.key: item for item in SETTINGS}


# ---------------------------------------------------------------- 状态读取（一律掩码）


def effective_value(item: SettingItem) -> Tuple[str, bool]:
    """取运行时生效值（config 与各适配器看到的就是它），返回 (值, 是否来自复用兄弟项)。"""
    value = (os.environ.get(item.key) or "").strip()
    if value:
        return value, False
    if item.derived_from:
        parent = (os.environ.get(item.derived_from) or "").strip()
        if parent:
            return parent, True
    return "", False


def mask_value(value: str) -> str:
    """掩码：只暴露前 2 字符与长度，绝不回显明文。

    长度 ≤4 时连前缀也不给（否则等于把短值整份暴露出去）。
    """
    if not value:
        return ""
    if len(value) <= 4:
        return f"***(长度 {len(value)})"
    return f"{value[:2]}***(长度 {len(value)})"


def describe_item(item: SettingItem) -> Dict[str, Any]:
    value, reused = effective_value(item)
    return {
        "key": item.key,
        "label": item.label,
        "purpose": item.purpose,
        "apply_entry": item.apply_entry,
        "degrades_to": item.degrades_to,
        "required": item.required,
        "default": item.default,
        "configured": bool(value),
        "reused_from": item.derived_from if reused else "",
        "value_masked": mask_value(value),
        "length": len(value),
    }


def capabilities() -> Dict[str, Any]:
    """当前在线能力开关：只描述「有没有配」，实际降级由各适配器兜底。"""
    plan_ready = bool(effective_value(SETTINGS_BY_KEY["AMAP_REST_KEY"])[0])
    js_key, js_key_reused = effective_value(SETTINGS_BY_KEY["AMAP_JS_KEY"])
    security_ready = bool(effective_value(SETTINGS_BY_KEY["AMAP_SECURITY_CODE"])[0])
    llm_ready = bool(effective_value(SETTINGS_BY_KEY["SILICONFLOW_API_KEY"])[0])
    return {
        "amap_plan": plan_ready,
        "amap_map": bool(js_key) and security_ready,
        "amap_js_key_reused": js_key_reused,
        "amap_map_blocked_by_security_code": bool(js_key) and not security_ready,
        "llm_real": llm_ready,
    }


def read_state() -> Dict[str, Any]:
    """GET /api/settings 与 doctor 共用的状态视图（不含任何密钥明文）。"""
    env_file = cfg.env_file_path()
    items = [describe_item(item) for item in SETTINGS]
    missing = [item["key"] for item in items if item["required"] and not item["configured"]]
    return {
        "env_file": str(env_file),
        "env_file_exists": env_file.exists(),
        "template_file": ".env.example",
        "items": items,
        "missing": missing,
        "ready": not missing,
        "capabilities": capabilities(),
        "offline_demo_ready": True,
    }


# ---------------------------------------------------------------- 写入校验


def clean_value(item: SettingItem, raw: Any) -> str:
    """清洗并校验单个值：去空白、拒绝换行/等号注入、限制长度。留空是合法值（表示清除）。"""
    if raw is None:
        return ""
    key = item.key
    value = (raw if isinstance(raw, str) else str(raw)).strip()
    # 外层引号在读取时会被解析器剥掉，这里先去掉，保证「文件里的值」与「内存里的值」一致
    value = value.strip('"').strip("'")
    if any(ch in value for ch in ("\n", "\r", "\x00", "=")):
        raise SettingError(f"{key} 不能包含换行或等号（防止向 .env 注入额外配置项）。")
    if len(value) > MAX_VALUE_LENGTH:
        raise SettingError(f"{key} 超过 {MAX_VALUE_LENGTH} 字符，疑似粘贴了非密钥内容。")
    if item.numeric and value:
        try:
            number = float(value)
        except ValueError:
            raise SettingError(f"{key} 需要填数字（留空则用默认值 {item.default}）。") from None
        # 只认有限正数：float('nan')/float('inf') 能通过 float()，但超时填 inf
        # 会让请求永久挂着、填 nan 会立刻炸，二者都不该写进配置。
        if not math.isfinite(number) or number <= 0:
            raise SettingError(
                f"{key} 需要填有限的正数（收到 {value!r}；留空则用默认值 {item.default}）。"
            )
    return value


def validate_updates(payload: Mapping[str, Any]) -> Dict[str, str]:
    """只接受注册表内的键；未注册键一律拒绝，否则等于开放任意环境变量写入。"""
    if not isinstance(payload, Mapping):
        raise SettingError("请求体应为 {配置项: 值} 形式的 JSON 对象。")
    if not payload:
        raise SettingError("没有需要保存的配置项。")
    unknown = [str(key) for key in payload if key not in SETTINGS_BY_KEY]
    if unknown:
        raise SettingError(
            "不支持保存的配置项：" + ", ".join(unknown) + "。可保存项：" + ", ".join(MANAGED_KEYS)
        )
    # 按注册表顺序整理，保证写入 .env 与返回结果的顺序稳定
    return {key: clean_value(SETTINGS_BY_KEY[key], payload[key]) for key in MANAGED_KEYS if key in payload}


# ---------------------------------------------------------------- .env 更新与热重载


# .env 中托管项追加块的标记行：同一次保存只保留一份
APPEND_MARKER = "# 以下为控制台「设置」视图追加的配置项"


def _merge_env_text(text: str, updates: Mapping[str, str]) -> str:
    """在原 .env 文本上就地替换托管键，保留全部注释与其他行，缺失的键追加到末尾。"""
    placed = set()
    lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in updates:
                lines.append(f"{key}={updates[key]}")
                placed.add(key)
                continue
        lines.append(raw_line)

    pending = [(key, value) for key, value in updates.items() if key not in placed]
    if pending:
        if lines and lines[-1].strip():
            lines.append("")
        if APPEND_MARKER not in lines:
            lines.append(APPEND_MARKER)
        for key, value in pending:
            lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def _backup(target: Path, content: str) -> Optional[Path]:
    """把即将被覆盖的旧 .env 原样存一份（文件本就不存在时不需要备份）。"""
    if not target.exists():
        return None
    backup_path = target.with_name(BACKUP_NAME)
    backup_path.write_text(content, encoding="utf-8")
    return backup_path


def write_settings(updates: Mapping[str, str]) -> Dict[str, Any]:
    """原子写入 .env（写前备份）并热重载，使改动不重启服务即刻生效。

    原子性：先写同目录临时文件，再 os.replace 覆盖目标 —— 中途崩溃只会留下一个 .tmp，
    不会出现半截 .env 把整份配置毁掉。
    """
    path = cfg.env_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    merged = _merge_env_text(existing, updates)

    backup_path = _backup(path, existing)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(merged, encoding="utf-8")
    os.replace(tmp_path, path)
    cfg.reload_env(path)
    return {
        "written": sorted(updates),
        "env_file": str(path),
        "backup_file": str(backup_path) if backup_path else "",
        "reloaded": True,
    }


def save_settings(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """POST /api/settings 的入口：校验 → 落盘 → 热重载 → 返回掩码后的最新状态。"""
    updates = validate_updates(payload)
    result = write_settings(updates)
    # JS Key / 安全密钥属前端注入项（模板里的 ZSX_CONFIG），变了就要让浏览器重新拉一次
    result["client_config_stale"] = any(
        key in updates for key in ("AMAP_JS_KEY", "AMAP_SECURITY_CODE")
    )
    result["state"] = read_state()
    return result


# ---------------------------------------------------------------- 访问来源判定


def is_loopback(address: Optional[str]) -> bool:
    """是否本机回环地址。只看 socket 直连的 remote_addr，不信任可伪造的 X-Forwarded-For。"""
    if not address:
        return False
    text = str(address).strip().strip("[]")
    if text.lower() == "localhost":
        return True
    if text.lower().startswith("::ffff:"):
        text = text[7:]
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False
