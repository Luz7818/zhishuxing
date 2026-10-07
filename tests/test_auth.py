"""tests/test_auth.py —— 把「管理端鉴权」钉死在测试里。

覆盖 auth.py 的令牌签发/校验/过期、口令 PBKDF2 校验与失败锁定，以及 app.py 的
分域门禁：ADMIN_PASSWORD 未设置时全部端点保持历史行为（默认关），设置后只有
ADMIN_POST_PATHS 里的管理动作要登录，乘客端点（/api/chat、/api/plan）与
/mobile、/health 保持开放。

隔离：ADMIN_PASSWORD 用 monkeypatch 环境变量（guard 在每次 create_app 时构建，
无跨用例状态）；锁定计数随 guard 新建自动复位。
"""

from __future__ import annotations

import base64
import time

import pytest

from zhishuxing.webapp import auth as admin_auth
from zhishuxing.webapp.app import create_app

PASSWORD = "test-admin-pass-9527"


@pytest.fixture()
def guarded_client(monkeypatch):
    """启用鉴权(设置了 ADMIN_PASSWORD)的应用客户端。"""
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture()
def open_client():
    """未启用鉴权(默认形态)的应用客户端:行为必须与历史版本一致。"""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _login(client, password=PASSWORD):
    return client.post("/api/admin/login", json={"password": password})


# ---------------------------------------------------------------- 默认关闭


def test_disabled_by_default_admin_post_open(open_client):
    resp = open_client.post("/api/rl/load_policy", json={})
    assert resp.status_code == 200  # 未启用鉴权:管理端点保持历史行为


def test_disabled_login_reports_not_enabled(open_client):
    resp = open_client.post("/api/admin/login", json={"password": "anything"})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["enabled"] is False


# ---------------------------------------------------------------- 启用后的门禁


def test_enabled_blocks_unauthenticated(guarded_client):
    resp = guarded_client.post("/api/rl/load_policy", json={})
    assert resp.status_code == 401
    assert "管理" in resp.get_json()["detail"]


def test_login_success_sets_cookie_and_token(guarded_client):
    resp = _login(guarded_client)
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["token"] and data["expires_at"] > int(time.time())
    set_cookie = "; ".join(resp.headers.getlist("Set-Cookie"))
    assert "zhishuxing_admin=" in set_cookie
    assert "HttpOnly" in set_cookie


def test_login_wrong_password_counts_down(guarded_client):
    resp = _login(guarded_client, "wrong-pass")
    assert resp.status_code == 401
    assert "再错" in resp.get_json()["error"]


def test_lockout_after_five_failures(guarded_client):
    for _ in range(5):
        assert _login(guarded_client, "wrong-pass").status_code == 401
    sixth = _login(guarded_client, PASSWORD)  # 连正确口令也在锁定期内被拒
    assert sixth.status_code == 401
    assert "锁定" in sixth.get_json()["error"]


def test_cookie_session_passes_admin_gate(guarded_client):
    assert _login(guarded_client).status_code == 200
    resp = guarded_client.post("/api/rl/load_policy", json={})
    assert resp.status_code == 200  # test_client 自动带会话 Cookie


def test_bearer_token_passes_admin_gate(guarded_client):
    token = _login(guarded_client).get_json()["data"]["token"]
    resp = guarded_client.post(
        "/api/rl/load_policy", json={}, headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200


def test_logout_clears_cookie(guarded_client):
    assert _login(guarded_client).status_code == 200
    assert guarded_client.post("/api/admin/logout", json={}).status_code == 200
    resp = guarded_client.post("/api/rl/load_policy", json={})
    assert resp.status_code == 401  # Cookie 已清,令牌不再随请求发送


def test_bad_token_rejected(guarded_client):
    resp = guarded_client.post(
        "/api/rl/load_policy", json={}, headers={"Authorization": "Bearer forged.token"}
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------- 公开域不受影响


def test_public_endpoints_open_when_enabled(guarded_client):
    assert guarded_client.get("/health").status_code == 200
    assert guarded_client.get("/mobile").status_code == 200
    assert guarded_client.get("/api/settings").status_code == 200  # 移动端依赖的掩码状态
    assert guarded_client.get("/admin/login").status_code == 200
    assert guarded_client.post("/api/navigation/plan", json={"start": [1, 2], "goal": [28, 12]}).status_code == 200
    assert guarded_client.post("/api/chat", json={"message": "赶时间怎么走"}).status_code == 200


def test_login_page_served(guarded_client):
    resp = guarded_client.get("/admin/login")
    assert resp.status_code == 200
    assert "管理登录".encode("utf-8") in resp.data


# ---------------------------------------------------------------- 令牌单元行为


def test_token_parse_and_expiry(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    guard = admin_auth.new_guard()
    assert guard is not None
    token = admin_auth.make_token(guard)
    assert admin_auth.parse_token(guard, token)

    raw = f"admin.{int(time.time()) - 10}".encode("utf-8")  # 过期令牌
    expired = base64.urlsafe_b64encode(raw).decode("ascii") + "." + admin_auth._sign(guard["secret"], raw.decode("utf-8"))
    assert not admin_auth.parse_token(guard, expired)

    tampered = token[:-4] + "beef"  # 改签名
    assert not admin_auth.parse_token(guard, tampered)
    assert admin_auth.token_expires_at(token) > int(time.time())
