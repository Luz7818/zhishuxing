"""tests/test_verify.py —— 把「部署验收」钉死在测试里。

覆盖 verify.py 的本地模式（全链路体检 + 报告落盘 + 授权/鉴权 WARN 通道）与
远程模式的探针逻辑（用 test_client 适配 fetch,并对泄漏/鉴权注入异常响应）。

隔离：workspace 重定向与 test_licensing.py 同款；验收报告写到 tmp_path,
绝不污染真实 data/outputs。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Tuple

import pytest

from zhishuxing import config as cfg
from zhishuxing import licensing
from zhishuxing.verify import _Checker, _remote_checks, _MASKED_SHAPE, run_verify


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.delenv(licensing.ENV_OVERRIDE, raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    return tmp_path


def _expired_license(workspace) -> None:
    record = {
        "product": licensing.PRODUCT,
        "customer": "客户A",
        "issued_at": (date.today() - timedelta(days=60)).isoformat(),
        "expires_at": (date.today() - timedelta(days=20)).isoformat(),
        "signature": "",
    }
    record["signature"] = licensing._signature("客户A", record["issued_at"], record["expires_at"])
    licensing.license_file_path().write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_masked_shape_detector():
    assert _MASKED_SHAPE.match("***") is not None
    assert _MASKED_SHAPE.match("ab***(长度 32)") is not None
    assert _MASKED_SHAPE.match("sk-real-leaked-key-1234567890") is None  # 长明文必须被抓


def test_local_verify_passes_with_warnings(workspace, tmp_path):
    report_dir = tmp_path / "reports"
    code = run_verify(report_dir=report_dir)
    assert code == 0  # 试用模式与未启用鉴权都是 WARN,不阻塞验收
    reports = list(report_dir.glob("verify-report-*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "VERIFY PASS" in text
    assert "授权状态" in text and "管理端鉴权" in text


def test_local_verify_fails_on_expired_license(workspace, tmp_path):
    _expired_license(workspace)
    report_dir = tmp_path / "reports"
    code = run_verify(report_dir=report_dir)
    assert code == 1
    text = next(report_dir.glob("verify-report-*.md")).read_text(encoding="utf-8")
    assert "FAIL  授权状态" in text
    assert "未通过" in text


def test_client_adapter_for_remote_checks(workspace):
    """远程检查逻辑与传输解耦:test_client 适配 fetch 即可复用全部断言。"""
    from zhishuxing.webapp.app import create_app

    app = create_app()
    app.config["TESTING"] = True

    def fetch(method: str, path: str) -> Tuple[int, str]:
        with app.test_client() as client:
            resp = client.open(path, method=method)
            return resp.status_code, resp.data.decode("utf-8")

    checker = _Checker()
    _remote_checks(checker, fetch, "http://test")
    ok, conclusion = checker.summary()
    assert ok, conclusion  # 试用模式/未启用鉴权 → 只有 WARN
    assert any("远程探针" in line for line in checker.lines)


def test_remote_detects_leaked_secret(workspace):
    def fetch(method: str, path: str) -> Tuple[int, str]:
        if path == "/api/settings":
            payload = {
                "ok": True,
                "data": {
                    "items": [{"key": "AMAP_REST_KEY", "value_masked": "sk-real-leaked-key-12345678"}],
                    "license": {"status": "valid", "message": "授权版"},
                    "admin_auth_enabled": True,
                },
            }
            return 200, json.dumps(payload)
        return 200, '{"status": "ok"}' if path == "/health" else ""

    checker = _Checker()
    _remote_checks(checker, fetch, "http://test")
    assert not ok_summary(checker)  # 泄漏必须 FAIL
    assert any("密钥不泄漏" in line and "FAIL" in line for line in checker.lines)


def test_remote_auth_gate_assertion(workspace):
    def fetch(method: str, path: str) -> Tuple[int, str]:
        if method == "POST":
            return 401, '{"detail": "需要登录"}'
        if path == "/api/settings":
            payload = {"ok": True, "data": {"items": [], "license": {"status": "valid", "message": "授权版"}, "admin_auth_enabled": True}}
            return 200, json.dumps(payload)
        return 200, '{"status": "ok"}'

    checker = _Checker()
    _remote_checks(checker, fetch, "http://test")
    assert not checker.failures
    assert any("未登录访问管理端点被拒" in line and "PASS" in line for line in checker.lines)


def ok_summary(checker: _Checker) -> bool:
    ok, _ = checker.summary()
    return ok
