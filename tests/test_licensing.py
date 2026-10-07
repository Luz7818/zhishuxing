"""tests/test_licensing.py —— 把「离线授权机制」钉死在测试里。

覆盖 licensing.py 的状态机（missing/valid/expiring/grace/expired/invalid）、
签发-激活往返、ZHISHUXING_LICENSE_FILE 覆盖、CLI license 子命令、
serve 启动门禁（过期超宽限退出码 3）与 settings.read_state 的授权展示。

隔离：monkeypatch `config.WORKSPACE_ROOT` 到 tmp_path（license_file_path 每次
调用都读模块属性，与 test_settings.py 的重定向手法一致）；ADMIN_PASSWORD /
ZHISHUXING_LICENSE_FILE 用 monkeypatch 环境变量自动还原。全程不碰真实 workspace。
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from zhishuxing import config as cfg
from zhishuxing import licensing
from zhishuxing.cli import main as cli_main
from zhishuxing.settings import read_state


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """把授权文件所在 workspace 重定向到 tmp_path,并清掉可能残留的环境覆盖。"""
    monkeypatch.setattr(cfg, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.delenv(licensing.ENV_OVERRIDE, raising=False)
    return tmp_path


def _write_file(workspace, customer: str, issued: str, expires: str) -> None:
    record = {
        "product": licensing.PRODUCT,
        "customer": customer,
        "issued_at": issued,
        "expires_at": expires,
        "signature": licensing._signature(customer, issued, expires),
    }
    licensing.license_file_path().write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def test_missing_license_is_trial(workspace):
    state = licensing.read_license()
    assert state.status == "missing"
    assert "试用" in state.message


def test_write_and_read_valid_license(workspace):
    licensing.write_license("客户A", days=365, out=licensing.license_file_path())
    state = licensing.read_license()
    assert state.status == "valid"
    assert state.customer == "客户A"
    assert 360 <= state.days_left <= 365
    assert "授权版" in state.message
    assert state.file == str(licensing.license_file_path())  # 有效态也必须回带文件路径


def test_permanent_license(workspace):
    licensing.write_license("客户A", days=0, out=licensing.license_file_path())
    state = licensing.read_license()
    assert state.status == "valid"
    assert state.expires_at == "永久"
    assert state.days_left == 9999


def test_expiring_soon_state(workspace):
    _write_file(workspace, "客户A", date.today().isoformat(), (date.today() + timedelta(days=10)).isoformat())
    state = licensing.read_license()
    assert state.status == "expiring"
    assert "续期" in state.message


def test_grace_period_allows_serve(workspace):
    _write_file(workspace, "客户A", (date.today() - timedelta(days=40)).isoformat(), (date.today() - timedelta(days=5)).isoformat())
    state = licensing.read_license()
    assert state.status == "grace"
    licensing.ensure_serve_allowed()  # 宽限期内不拒绝


def test_expired_beyond_grace_refuses_serve(workspace):
    _write_file(workspace, "客户A", (date.today() - timedelta(days=60)).isoformat(), (date.today() - timedelta(days=20)).isoformat())
    assert licensing.read_license().status == "expired"
    with pytest.raises(licensing.LicenseExpired):
        licensing.ensure_serve_allowed()


def test_tampered_signature_is_invalid(workspace):
    licensing.write_license("客户A", days=365, out=licensing.license_file_path())
    record = json.loads(licensing.license_file_path().read_text(encoding="utf-8"))
    record["customer"] = "客户B"  # 改内容不改签名
    licensing.license_file_path().write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert licensing.read_license().status == "invalid"


def test_broken_file_is_invalid(workspace):
    licensing.license_file_path().write_text("not json", encoding="utf-8")
    assert licensing.read_license().status == "invalid"


def test_wrong_product_is_invalid(workspace):
    _write_file(workspace, "客户A", date.today().isoformat(), (date.today() + timedelta(days=30)).isoformat())
    record = json.loads(licensing.license_file_path().read_text(encoding="utf-8"))
    record["product"] = "other-product"
    licensing.license_file_path().write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert licensing.read_license().status == "invalid"


def test_env_override_path(workspace, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere.lic"
    licensing.write_license("客户B", days=30, out=elsewhere)
    monkeypatch.setenv(licensing.ENV_OVERRIDE, str(elsewhere))
    assert licensing.read_license().customer == "客户B"
    assert licensing.license_file_path() == elsewhere


def test_cli_license_activates_file(workspace, tmp_path):
    source = tmp_path / "issued.lic"
    licensing.write_license("客户C", days=100, out=source)
    assert cli_main(["license", "--file", str(source)]) == 0
    assert licensing.read_license().customer == "客户C"


def test_serve_refused_with_expired_license_exit_code_3(workspace):
    _write_file(workspace, "客户A", (date.today() - timedelta(days=60)).isoformat(), (date.today() - timedelta(days=20)).isoformat())
    assert cli_main(["serve", "--port", "7899"]) == 3


def test_read_state_exposes_license_and_auth_flag(workspace, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    state = read_state()
    assert state["license"]["status"] == "missing"
    assert state["admin_auth_enabled"] is False
    monkeypatch.setenv("ADMIN_PASSWORD", "secret-pass")
    assert read_state()["admin_auth_enabled"] is True


def test_state_machine_boundaries(workspace):
    """30 天阈值与 14 天宽限的边界日期各落一次,防止算术手滑。"""
    licensing.write_license("客户A", days=licensing.EXPIRING_SOON_DAYS, out=licensing.license_file_path())
    assert licensing.read_license().status == "expiring"  # 剩余恰 30 天
    _write_file(workspace, "客户A", date.today().isoformat(), (date.today() + timedelta(days=licensing.EXPIRING_SOON_DAYS + 1)).isoformat())
    assert licensing.read_license().status == "valid"  # 剩余 31 天回到 valid
    _write_file(workspace, "客户A", date.today().isoformat(), (date.today() - timedelta(days=licensing.GRACE_DAYS)).isoformat())
    assert licensing.read_license().status == "grace"  # 过期恰 14 天仍在宽限
    _write_file(workspace, "客户A", date.today().isoformat(), (date.today() - timedelta(days=licensing.GRACE_DAYS + 1)).isoformat())
    assert licensing.read_license().status == "expired"  # 过期 15 天拒启
