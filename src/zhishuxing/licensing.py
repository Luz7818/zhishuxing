"""离线授权(license)机制:HMAC 签名 + 客户名 + 有效期,无证走试用模式。

设计边界(与交付文档口径一致,不可夸大):
- 签名是对称 HMAC,校验密钥随包分发,防「随手复制」不防专业逆向;升级路径是非对称
  签名(Ed25519),见 TODO.md;
- 无证不锁功能:试用模式全功能可用,只在界面与 doctor 如实标注「试用版」——售前
  演示零障碍,生产部署的约束靠交付合同与验收流程;
- 过期宽限 GRACE_DAYS 天:政企客户最反感「到期当天服务消失」,宽限期内横幅提醒。

license.lic 是 JSON 文件:{product, customer, issued_at, expires_at, signature},
签名覆盖 product|customer|issued_at|expires_at 四字段。文件路径在调用时求值
(锚定 workspace 根,与 env_file_path() 同款约定),ZHISHUXING_LICENSE_FILE 可覆盖。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

from . import config as cfg

PRODUCT = "zhishuxing"
LICENSE_FILENAME = "license.lic"
ENV_OVERRIDE = "ZHISHUXING_LICENSE_FILE"
EXPIRING_SOON_DAYS = 30   # 剩余 ≤30 天即提示续期
GRACE_DAYS = 14           # 过期后的宽限天数,超过即拒绝启动服务
LICENSE_SECRET = "c99709ee72248be10f880d7c27b5726682a93b6793a06bdd571e16a008d77560"


class LicenseExpired(RuntimeError):
    """授权过期且超出宽限期:serve 启动前必须拒绝,exe 引导与 CLI 都按此文案呈现。"""


@dataclass(frozen=True)
class LicenseState:
    status: str      # missing / invalid / valid / expiring / grace / expired
    customer: str
    issued_at: str
    expires_at: str
    days_left: int   # 负数表示已过期天数
    message: str     # 面向用户的一句话结论(UI/doctor 直接展示)
    file: str

    def to_public(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "customer": self.customer,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "days_left": self.days_left,
            "message": self.message,
            "file": self.file,
        }


def license_file_path() -> Path:
    """当前生效的授权文件路径(调用时求值;测试用 monkeypatch cfg.WORKSPACE_ROOT 重定向)。"""
    override = (os.environ.get(ENV_OVERRIDE) or "").strip()
    if override:
        return Path(override)
    return cfg.WORKSPACE_ROOT / LICENSE_FILENAME


def _signature(customer: str, issued_at: str, expires_at: str) -> str:
    payload = f"{PRODUCT}|{customer}|{issued_at}|{expires_at}"
    return hmac.new(LICENSE_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def write_license(customer: str, days: int, out: Path) -> Dict[str, str]:
    """签发一份授权(仅厂商侧 scripts/make_license.py 使用):days<=0 视为永久授权。"""
    customer = customer.strip()
    if not customer:
        raise ValueError("customer 不能为空")
    issued = date.today()
    expires = "" if days <= 0 else (issued + timedelta(days=days)).isoformat()
    record = {
        "product": PRODUCT,
        "customer": customer,
        "issued_at": issued.isoformat(),
        "expires_at": expires,
        "signature": _signature(customer, issued.isoformat(), expires),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def _evaluate(record: Dict[str, Any], file: str = "") -> "LicenseState":
    """对已通过签名与产品校验的记录做有效期判定(状态机唯一的日期逻辑入口)。"""
    customer = str(record.get("customer", ""))
    issued_at = str(record.get("issued_at", ""))
    expires_at = str(record.get("expires_at", ""))
    if not expires_at:
        return LicenseState("valid", customer, issued_at, "永久", 9999, f"授权版 · {customer}(永久授权)", file)
    try:
        days_left = (date.fromisoformat(expires_at) - date.today()).days
    except ValueError:
        return LicenseState("invalid", customer, issued_at, expires_at, 0, "授权文件日期无法解析,按试用模式运行", file)
    if days_left < -GRACE_DAYS:
        return LicenseState(
            "expired", customer, issued_at, expires_at, days_left,
            f"授权已过期 {abs(days_left)} 天(超出 {GRACE_DAYS} 天宽限),服务拒绝启动;请联系厂商续期",
            file,
        )
    if days_left < 0:
        return LicenseState(
            "grace", customer, issued_at, expires_at, days_left,
            f"授权已过期 {abs(days_left)} 天,宽限期剩余 {GRACE_DAYS + days_left} 天,请尽快续期",
            file,
        )
    if days_left <= EXPIRING_SOON_DAYS:
        return LicenseState(
            "expiring", customer, issued_at, expires_at, days_left,
            f"授权版 · {customer} · {expires_at} 到期(剩 {days_left} 天),请安排续期",
            file,
        )
    return LicenseState(
        "valid", customer, issued_at, expires_at, days_left,
        f"授权版 · {customer} · {expires_at} 到期",
        file,
    )


def read_license() -> LicenseState:
    """读取并判定当前授权。文件缺失/被篡改/损坏都如实标注,不抛异常(试用模式兜底)。"""
    path = license_file_path()
    if not path.exists():
        return LicenseState(
            "missing", "", "", "", 0,
            "试用模式(未放置授权文件,全功能可用;生产部署请向厂商索取授权文件)",
            str(path),
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LicenseState("invalid", "", "", "", 0, "授权文件损坏或不可读,按试用模式运行", str(path))
    if not isinstance(record, dict) or record.get("product") != PRODUCT:
        return LicenseState("invalid", "", "", "", 0, "授权文件与本产品不匹配,按试用模式运行", str(path))
    expected = _signature(str(record.get("customer", "")), str(record.get("issued_at", "")), str(record.get("expires_at", "")))
    if not hmac.compare_digest(str(record.get("signature", "")), expected):
        return LicenseState("invalid", "", "", "", 0, "授权文件签名校验失败(被修改或伪造),按试用模式运行", str(path))
    return _evaluate(record, file=str(path))


def ensure_serve_allowed() -> None:
    """serve 启动前的授权门禁:只有「过期且超出宽限」才拒绝,其余状态照常放行。"""
    state = read_license()
    if state.status == "expired":
        raise LicenseExpired(state.message)


def license_state() -> Dict[str, Any]:
    """供 settings.read_state()/doctor/verify 复用的公开状态视图(无敏感字段)。"""
    return read_license().to_public()


if __name__ == "__main__":
    state = read_license()
    print(f"授权状态: {state.message}")
    print(f"文件: {state.file}")
    sys.exit(3 if state.status == "expired" else 0)
