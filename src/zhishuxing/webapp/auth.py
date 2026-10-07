"""管理端会话鉴权:单管理员口令 + HMAC 会话令牌,只保护「管理动作」端点。

与姊妹项目 harness 的整站登录不同(那里是纯内部看板),智枢星有面向乘客的
对话/规划接口和 /mobile PWA——公开域一律放行,本模块只负责管理域的门禁:

- `ADMIN_PASSWORD` 环境变量**设置了才启用**;未设置 = 鉴权关闭,即开发与本机
  演示的默认形态,行为与历史版本完全一致;
- 口令 PBKDF2-SHA256 校验(每次 create_app 时算一次驻内存,不落盘、无 auth.json);
- 会话令牌 = HMAC-SHA256 签名的 `admin.<过期unix秒>`,12 小时有效;登出只清
  Cookie,令牌在有效期内仍可用(无状态令牌的自然取舍,与 harness 同口径);
- 登录失败 5 次锁 10 分钟(进程内计数;waitress 单进程成立,多实例部署需在最外层
  网关限流)。

本模块不 import Flask:guard 的构建/校验是纯逻辑,请求与 Cookie 的处理在 app.py。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
from typing import Any, Dict, Optional

COOKIE_NAME = "zhishuxing_admin"
SESSION_TTL = 12 * 3600
PBKDF2_ITERATIONS = 120_000
LOCKOUT_THRESHOLD = 5
LOCKOUT_SECONDS = 600
_ENV_KEY = "ADMIN_PASSWORD"


class AuthError(RuntimeError):
    """登录失败或被锁定:调用方按 401 返回 error 文案。"""


def enabled() -> bool:
    """鉴权开关:只看环境变量是否提供了非空口令。"""
    return bool((os.environ.get(_ENV_KEY) or "").strip())


def new_guard() -> Optional[Dict[str, Any]]:
    """按当前环境构建 guard(含口令哈希与签名密钥);未设置口令时返回 None=鉴权关闭。"""
    password = (os.environ.get(_ENV_KEY) or "").strip()
    if not password:
        return None
    salt = secrets.token_hex(16).encode("ascii")
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return {
        "salt": salt,
        "derived": derived,
        "secret": secrets.token_hex(32),
        "fails": 0,
        "locked_until": 0.0,
        "lock": threading.Lock(),
    }


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def make_token(guard: Dict[str, Any]) -> str:
    payload = f"admin.{int(time.time()) + SESSION_TTL}"
    raw = payload.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii") + "." + _sign(guard["secret"], payload)


def token_expires_at(token: str) -> int:
    """令牌失效时刻(unix 秒)。仅用于告知客户端,鉴权判定一律走 parse_token。"""
    try:
        raw = base64.urlsafe_b64decode(token.rsplit(".", 1)[0].encode("ascii")).decode("utf-8")
        return int(raw.rsplit(".", 1)[1])
    except Exception:
        return 0


def parse_token(guard: Dict[str, Any], token: str) -> bool:
    """校验会话令牌:签名 + 过期时间,全程 compare_digest 防时序侧信道。"""
    if not token or "." not in token:
        return False
    try:
        b64, sig = token.rsplit(".", 1)
        raw = base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8")
        expected = _sign(guard["secret"], raw)
        if not hmac.compare_digest(sig, expected):
            return False
        username, expiry = raw.rsplit(".", 1)
        return username == "admin" and int(expiry) >= int(time.time())
    except Exception:
        return False


def authenticate(guard: Dict[str, Any], password: str) -> str:
    """校验口令:成功返回新会话令牌;口令不对或处于锁定期抛 AuthError。"""
    now = time.time()
    with guard["lock"]:
        if now < guard["locked_until"]:
            wait = int(guard["locked_until"] - now) + 1
            raise AuthError(f"失败次数过多,已锁定约 {wait} 秒,请稍后再试")
        derived = hashlib.pbkdf2_hmac(
            "sha256", (password or "").encode("utf-8"), guard["salt"], PBKDF2_ITERATIONS
        )
        if not hmac.compare_digest(derived, guard["derived"]):
            guard["fails"] += 1
            if guard["fails"] >= LOCKOUT_THRESHOLD:
                guard["locked_until"] = now + LOCKOUT_SECONDS
                guard["fails"] = 0
                raise AuthError(f"口令不正确,失败 {LOCKOUT_THRESHOLD} 次已锁定 10 分钟")
            raise AuthError(f"口令不正确(再错 {LOCKOUT_THRESHOLD - guard['fails']} 次将锁定 10 分钟)")
        guard["fails"] = 0
    return make_token(guard)
