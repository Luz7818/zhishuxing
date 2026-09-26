"""tests/test_settings.py —— 把「密钥配置托管」这次改造钉死在测试里。

覆盖 settings.py（注册表 / 掩码 / 清洗 / 校验 / .env 合并写入与备份 / 热重载 / loopback 判定）、
config.reload_env、GET+POST /api/settings、cli doctor，以及最值钱的那条承诺：
**workspace 里没有任何 .env、进程里没有任何密钥时，全链路仍能离线完整演示。**

隔离怎么做的（三条一起用，缺一会污染同一次 pytest 会话）：
1. workspace 重定向：monkeypatch 掉 `config.WORKSPACE_ROOT`。
   `env_file_path()` 是函数、每次调用都读模块属性，于是 .env 的读写全部落到 tmp_path。
   子进程场景改用 `ZHISHUXING_WORKSPACE`（`_find_workspace_root` 的第一优先级）重定向。
   注意 `cfg.paths` 是 import 期算好的，仍指向真实仓库的 configs/data —— 起服务需要它们。
2. 环境变量整体快照 + 差异还原：热重载（`_load_dotenv(override=True)`）会写 os.environ，
   而且 .env 里**未托管的行**（如 OTHER_THING=…）同样会被写进去，逐项 delenv 罩不住，
   所以用 autouse 夹具在单个测试前后做全量快照还原。
3. 模块级保险丝：`real_env_untouched` 比对真实 .env / .env.bak 的 sha256；
   `environment_snapshot_at_import` 记录进本模块前的密钥环境。任何一处绕过重定向
   写到真实文件、或把改动漏给同会话的下一个测试，都会直接失败。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import pytest
from flask.testing import FlaskClient

from zhishuxing import config as cfg
from zhishuxing import settings as settings_store
from zhishuxing.cli import main as cli_main
from zhishuxing.llm.adapters import MockLLMAdapter
from zhishuxing.webapp.app import create_app
from zhishuxing.webapp.service import ZhiShuXingWebService

# 本文件 import 时 cfg.WORKSPACE_ROOT 还没被任何夹具改过 —— 这就是真实仓库根目录
ORIGINAL_WORKSPACE_ROOT = cfg.WORKSPACE_ROOT
# 受保护的真实文件：本模块执行期间不得被改动（只哈希，绝不把内容读进断言消息）
PROTECTED_FILES: Tuple[Path, ...] = (
    ORIGINAL_WORKSPACE_ROOT / ".env",
    ORIGINAL_WORKSPACE_ROOT / settings_store.BACKUP_NAME,
)
# 进本模块前的托管键环境，用于最后一个用例自检「没有污染同会话的其他测试」
ENVIRONMENT_SNAPSHOT_AT_IMPORT = {key: os.environ.get(key) for key in settings_store.MANAGED_KEYS}

# 假密钥：尾段足够独特，明文出现在任何响应/输出里都能被抓到
FAKE_SECRET = "fakeamapkey-9f3d7c1b5e2a8d40c6b1"
REQUIRED_KEYS: Tuple[str, ...] = tuple(item.key for item in settings_store.SETTINGS if item.required)
OPTIONAL_KEYS: Tuple[str, ...] = tuple(item.key for item in settings_store.SETTINGS if not item.required)
# 托管键集合：写 .env 会把这些键刷进进程环境，清理时按它整批处理
MANAGED_KEYS_SET = frozenset(settings_store.MANAGED_KEYS)

# 一份「手工编辑过」的 .env：有注释、有空行、有未托管行、有整行注释掉的托管键、键名两侧有空格
ENV_TEXT_WITH_COMMENTS = (
    "# 智枢星本地配置（手写的，别丢）\n"
    "AMAP_REST_KEY=old-rest-value\n"
    "\n"
    "# 下面这行注释掉的键不属于托管范围\n"
    "# AMAP_JS_KEY=must-stay-as-comment\n"
    "OTHER_THING=keep me exactly\n"
    "  AMAP_TIMEOUT = 8  \n"
)


# ---------------------------------------------------------------- 工具函数


def sha256_of(path: Path) -> str:
    """文件哈希；文件不存在时返回固定串。"""
    if not path.exists():
        return "<missing>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item_of(key: str) -> settings_store.SettingItem:
    return settings_store.SETTINGS_BY_KEY[key]


def set_env(values: Mapping[str, str]) -> None:
    """先清掉全部托管键再按 values 设置（还原由 restore_environment 兜底）。"""
    for key in settings_store.MANAGED_KEYS:
        os.environ.pop(key, None)
    for key, value in values.items():
        os.environ[key] = value


def entry_of(state: Mapping[str, Any], key: str) -> Dict[str, Any]:
    for entry in state["items"]:
        if entry["key"] == key:
            return entry
    raise AssertionError(f"状态视图缺少配置项 {key}")


def assert_no_plaintext(text: str, secret: str) -> None:
    """掩码契约：明文整体、以及第 3 个字符起的尾段，都不许出现在 text 里。"""
    assert secret not in text, "响应/输出里出现了密钥明文"
    assert secret[2:] not in text, "响应/输出泄露了掩码前缀之外的密钥内容"


def assignments_of(lines: Sequence[str], key: str) -> List[str]:
    """真正给该键赋值的行（整行注释与无关行都不算）。"""
    return [
        line
        for line in lines
        if line.strip()
        and not line.strip().startswith("#")
        and line.partition("=")[0].strip() == key
    ]


# ---------------------------------------------------------------- 隔离夹具


@pytest.fixture(scope="module", autouse=True)
def real_env_untouched() -> Iterator[Dict[Path, str]]:
    """保险丝：真实 workspace 的 .env / .env.bak 在本模块执行期间必须一字不改。"""
    before = {path: sha256_of(path) for path in PROTECTED_FILES}
    yield before
    changed = [str(path) for path, digest in before.items() if sha256_of(path) != digest]
    assert not changed, f"真实密钥文件被改动了，说明 workspace 重定向失效：{changed}"


@pytest.fixture(autouse=True)
def restore_environment() -> Iterator[None]:
    """单个测试结束按快照逐项还原 os.environ（新增的删掉、被改的写回）。

    热重载会把 .env 里的键（含未托管行）刷进进程环境，不还原就会让同会话里其他测试
    （如 test_api 的「无 Key 走降级」用例）凭空看到密钥而翻车。
    """
    saved = dict(os.environ)
    yield
    for key in [key for key in os.environ if key not in saved]:
        del os.environ[key]
    for key, value in saved.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 .env 的落点重定向到 tmp_path（读写都够不到真实文件）。"""
    monkeypatch.setattr(cfg, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setenv("ZHISHUXING_WORKSPACE", str(tmp_path))
    assert cfg.env_file_path() == tmp_path / ".env"
    assert settings_store.cfg.WORKSPACE_ROOT == tmp_path  # settings 用的是同一个 config 模块对象
    return tmp_path


@pytest.fixture()
def bare_workspace(workspace: Path) -> Path:
    """干净起点：workspace 里没有 .env，进程里也没有任何托管键。"""
    set_env({})
    assert not cfg.env_file_path().exists()
    return workspace


# ---------------------------------------------------------------- HTTP 客户端夹具


@pytest.fixture()
def api_client(bare_workspace: Path) -> Iterator[FlaskClient]:
    """默认（可写）控制台客户端；bare_workspace 仅用于确立夹具先后顺序。"""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture()
def readonly_client(bare_workspace: Path) -> Iterator[FlaskClient]:
    """settings_writable=False：生产模式监听非 loopback 时写入接口关闭。"""
    app = create_app(settings_writable=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------- 1. 注册表契约


def test_registry_shape_and_lookup_are_consistent():
    keys = [item.key for item in settings_store.SETTINGS]
    assert len(keys) == len(set(keys)), "注册表键重复"
    assert settings_store.MANAGED_KEYS == tuple(keys)
    assert settings_store.SETTINGS_BY_KEY == {item.key: item for item in settings_store.SETTINGS}
    for item in settings_store.SETTINGS:
        # 这四段文案要能直接喂给「设置」视图 / doctor / 移动端引导，不能空着
        assert item.label and item.purpose and item.apply_entry and item.degrades_to
        assert isinstance(item.required, bool) and isinstance(item.numeric, bool)
        assert item.numeric is (item.key == "AMAP_TIMEOUT")  # 数值型校验只该挂在超时项上
    # 离线演示的兜底就是 Mock，把 LLM 密钥算成必需项会让「零密钥可演示」自相矛盾
    assert "SILICONFLOW_API_KEY" in OPTIONAL_KEYS
    assert set(REQUIRED_KEYS) == {"AMAP_REST_KEY", "AMAP_JS_KEY", "AMAP_SECURITY_CODE"}
    assert item_of("AMAP_JS_KEY").derived_from == "AMAP_REST_KEY"


def test_describe_item_exposes_registry_fields_plus_masked_state(bare_workspace: Path):
    set_env({"AMAP_REST_KEY": FAKE_SECRET})
    item = item_of("AMAP_JS_KEY")
    described = settings_store.describe_item(item)
    assert described["key"] == item.key
    assert described["label"] == item.label
    assert described["purpose"] == item.purpose
    assert described["apply_entry"] == item.apply_entry
    assert described["degrades_to"] == item.degrades_to
    assert described["required"] is item.required
    assert described["default"] == item.default
    # JS Key 留空 → 复用 REST Key，并把来源标出来
    assert described["configured"] is True
    assert described["reused_from"] == "AMAP_REST_KEY"
    assert described["value_masked"] == settings_store.mask_value(FAKE_SECRET)
    assert described["length"] == len(FAKE_SECRET)
    assert_no_plaintext(json.dumps(described, ensure_ascii=False), FAKE_SECRET)


def test_effective_value_distinguishes_own_value_from_reuse(bare_workspace: Path):
    set_env({})
    assert settings_store.effective_value(item_of("AMAP_REST_KEY")) == ("", False)
    set_env({"AMAP_JS_KEY": "js-only-value"})
    assert settings_store.effective_value(item_of("AMAP_JS_KEY")) == ("js-only-value", False)
    assert settings_store.effective_value(item_of("AMAP_REST_KEY")) == ("", False)  # 兄弟项不回溯
    set_env({"AMAP_JS_KEY": "   ", "AMAP_REST_KEY": "rest-value"})
    assert settings_store.effective_value(item_of("AMAP_JS_KEY")) == ("rest-value", True)


# ---------------------------------------------------------------- 2. mask_value


@pytest.mark.parametrize("value", ["", None], ids=["empty", "none"])
def test_mask_value_of_unconfigured_value_is_empty_string(value: Optional[str]):
    # 未配置项给空串而不是 "***"，前端据此区分「没填」和「填了但很短」；
    # 未配置时 effective_value 就是 ""（该口径在 test_effective_value_... 里单独钉）
    assert settings_store.mask_value(value or "") == ""


@pytest.mark.parametrize("value", ["a", "ab", "abc", "abcd"])
def test_mask_value_hides_even_the_prefix_for_short_values(value: str):
    masked = settings_store.mask_value(value)
    assert masked == f"***(长度 {len(value)})"
    assert value[0] not in masked  # 连首字符都不给
    assert value not in masked


def test_mask_value_of_long_value_exposes_only_first_two_chars():
    value = "abcdefghij"
    masked = settings_store.mask_value(value)
    assert masked == "ab***(长度 10)"
    assert masked.startswith(value[:2])
    assert value[2:] not in masked
    assert str(len(value)) in masked  # 保留长度：运维靠它判断是不是粘贴漏了字符


@pytest.mark.parametrize("value", [FAKE_SECRET, "x" * 400, "中文密钥值-abcdefg"])
def test_mask_value_never_returns_the_original(value: str):
    masked = settings_store.mask_value(value)
    assert masked and masked != value
    assert_no_plaintext(masked, value)


# ---------------------------------------------------------------- 3. clean_value


@pytest.mark.parametrize(
    "raw",
    ["abc\nEVIL=1", "abc\rEVIL=1", "abc\r\nEVIL=1", "abc\x00def", "abc=def"],
    ids=["newline", "carriage_return", "crlf", "nul", "equals"],
)
def test_clean_value_rejects_injection_characters(raw: str):
    with pytest.raises(settings_store.SettingError) as excinfo:
        settings_store.clean_value(item_of("AMAP_REST_KEY"), raw)
    message = str(excinfo.value)
    assert "AMAP_REST_KEY" in message  # 报错点名是哪一项
    assert "换行" in message and "等号" in message


def test_clean_value_enforces_max_value_length():
    limit = settings_store.MAX_VALUE_LENGTH
    item = item_of("AMAP_REST_KEY")
    assert settings_store.clean_value(item, "x" * limit) == "x" * limit
    with pytest.raises(settings_store.SettingError) as excinfo:
        settings_store.clean_value(item, "x" * (limit + 1))
    assert str(limit) in str(excinfo.value)


def test_clean_value_strips_surrounding_whitespace_and_outer_quotes():
    item = item_of("AMAP_REST_KEY")
    for raw in ["  abc123key  ", '  "abc123key"  ', "'abc123key'", '"abc123key"']:
        assert settings_store.clean_value(item, raw) == "abc123key"
    # 空串 / 纯空白 / None 都表示「清除该项」，是合法值
    assert settings_store.clean_value(item, "") == ""
    assert settings_store.clean_value(item, "   ") == ""
    assert settings_store.clean_value(item, None) == ""


def test_clean_value_numeric_rejects_text_but_allows_blank():
    item = item_of("AMAP_TIMEOUT")
    assert settings_store.clean_value(item, " 12 ") == "12"
    assert settings_store.clean_value(item, "7.5") == "7.5"
    assert settings_store.clean_value(item, "") == ""  # 留空回落默认值
    for bad in ["abc", "10s", "ten"]:
        with pytest.raises(settings_store.SettingError) as excinfo:
            settings_store.clean_value(item, bad)
        assert "数字" in str(excinfo.value)
        assert item.default in str(excinfo.value)  # 报错要告诉用户默认值是多少
    # float() 会接受 nan/inf/1e690，但它们作为超时是「永久挂着」或「立刻炸」，
    # 0 与负数同样无意义 —— 数值项只认有限正数。
    for bad in ["nan", "inf", "-inf", "1e690", "NaN", "0", "-3"]:
        with pytest.raises(settings_store.SettingError) as excinfo:
            settings_store.clean_value(item, bad)
        assert "有限的正数" in str(excinfo.value)
    # 非数值项不受数字校验约束
    assert settings_store.clean_value(item_of("SILICONFLOW_MODEL"), "Qwen/Qwen2.5-7B") == "Qwen/Qwen2.5-7B"


def test_cleaned_value_stays_identical_in_file_and_memory(bare_workspace: Path):
    """钉住「文件里的值 == 内存里的值」：清洗时剥掉的空白与外层引号不许再回来。"""
    result = settings_store.save_settings({"AMAP_REST_KEY": '  "spaced-key-value"  '})
    assert result["written"] == ["AMAP_REST_KEY"]
    assert assignments_of(cfg.env_file_path().read_text(encoding="utf-8").splitlines(), "AMAP_REST_KEY") == [
        "AMAP_REST_KEY=spaced-key-value"
    ]
    assert os.environ["AMAP_REST_KEY"] == "spaced-key-value"
    assert entry_of(result["state"], "AMAP_REST_KEY")["length"] == len("spaced-key-value")


# ---------------------------------------------------------------- 4. validate_updates


def test_validate_updates_rejects_unregistered_keys_and_lists_saveable_ones():
    with pytest.raises(settings_store.SettingError) as excinfo:
        settings_store.validate_updates({"PATH": "c:\\windows", "SILICONFLOW_Evil_KEY": "x"})
    message = str(excinfo.value)
    assert "PATH" in message and "SILICONFLOW_Evil_KEY" in message
    for key in settings_store.MANAGED_KEYS:  # 错误信息要能直接当界面提示用
        assert key in message
    assert "可保存项" in message


def test_validate_updates_rejects_empty_payload():
    with pytest.raises(settings_store.SettingError) as excinfo:
        settings_store.validate_updates({})
    assert "没有需要保存" in str(excinfo.value)


@pytest.mark.parametrize(
    "payload", [[("AMAP_REST_KEY", "v")], "AMAP_REST_KEY=v", 42, None], ids=["list", "str", "int", "none"]
)
def test_validate_updates_rejects_non_mapping_payload(payload: Any):
    with pytest.raises(settings_store.SettingError) as excinfo:
        settings_store.validate_updates(payload)
    assert "JSON 对象" in str(excinfo.value)


def test_validate_updates_orders_by_registry_and_cleans_each_value():
    updates = settings_store.validate_updates(
        {"AMAP_TIMEOUT": " 9 ", "SILICONFLOW_API_KEY": '"sk-abc"', "AMAP_REST_KEY": "rest"}
    )
    assert list(updates) == ["AMAP_REST_KEY", "SILICONFLOW_API_KEY", "AMAP_TIMEOUT"]  # 注册表顺序，写入稳定
    assert updates == {"AMAP_REST_KEY": "rest", "SILICONFLOW_API_KEY": "sk-abc", "AMAP_TIMEOUT": "9"}


def test_validate_updates_rejects_the_whole_batch_when_one_value_is_invalid():
    with pytest.raises(settings_store.SettingError):
        settings_store.validate_updates({"AMAP_REST_KEY": "ok", "AMAP_TIMEOUT": "abc"})


# ---------------------------------------------------------------- 5. _merge_env_text


def test_merge_updates_in_place_and_keeps_comments_and_unmanaged_lines():
    merged = settings_store._merge_env_text(ENV_TEXT_WITH_COMMENTS, {"AMAP_REST_KEY": "brand-new"})
    lines = merged.splitlines()
    assert assignments_of(lines, "AMAP_REST_KEY") == ["AMAP_REST_KEY=brand-new"]  # 就地替换，不追加第二份
    assert lines[0] == "# 智枢星本地配置（手写的，别丢）"  # 原注释逐行保留
    assert "# 下面这行注释掉的键不属于托管范围" in lines
    assert "# AMAP_JS_KEY=must-stay-as-comment" in lines  # 整行注释不许被当成托管键改写
    assert "OTHER_THING=keep me exactly" in lines  # 未托管行原样保留
    assert "" in lines  # 原空行也在


def test_merge_touches_only_the_requested_key():
    merged = settings_store._merge_env_text(ENV_TEXT_WITH_COMMENTS, {"AMAP_TIMEOUT": "3"})
    lines = merged.splitlines()
    assert assignments_of(lines, "AMAP_TIMEOUT") == ["AMAP_TIMEOUT=3"]  # 顺手规范了键名两侧空格
    assert "  AMAP_TIMEOUT = 8  " not in lines
    assert assignments_of(lines, "AMAP_REST_KEY") == ["AMAP_REST_KEY=old-rest-value"]  # 没动的键保持原值


def test_merge_appends_missing_keys_under_marker_once_and_is_idempotent():
    first = settings_store._merge_env_text(ENV_TEXT_WITH_COMMENTS, {"AMAP_JS_KEY": "js-1"})
    assert first.count(settings_store.APPEND_MARKER) == 1
    assert assignments_of(first.splitlines(), "AMAP_JS_KEY") == ["AMAP_JS_KEY=js-1"]

    second = settings_store._merge_env_text(first, {"AMAP_JS_KEY": "js-2", "AMAP_SECURITY_CODE": "sec-2"})
    # 幂等：追加块标记只有一份，托管键不重复出现
    assert second.count(settings_store.APPEND_MARKER) == 1
    for key, value in (("AMAP_JS_KEY", "js-2"), ("AMAP_SECURITY_CODE", "sec-2")):
        assert assignments_of(second.splitlines(), key) == [f"{key}={value}"]
    assert "# 智枢星本地配置（手写的，别丢）" in second.splitlines()
    assert "OTHER_THING=keep me exactly" in second.splitlines()

    third = settings_store._merge_env_text(second, {"AMAP_TIMEOUT": "42"})
    assert third.count(settings_store.APPEND_MARKER) == 1
    assert assignments_of(third.splitlines(), "AMAP_TIMEOUT") == ["AMAP_TIMEOUT=42"]


def test_merge_on_empty_text_still_produces_a_valid_parseable_env():
    merged = settings_store._merge_env_text("", {"AMAP_REST_KEY": "x"})
    assert merged.splitlines() == [settings_store.APPEND_MARKER, "AMAP_REST_KEY=x"]
    assert cfg.parse_env_text(merged) == {"AMAP_REST_KEY": "x"}


def test_merge_writes_empty_value_to_clear_a_key():
    merged = settings_store._merge_env_text(ENV_TEXT_WITH_COMMENTS, {"AMAP_REST_KEY": ""})
    assert assignments_of(merged.splitlines(), "AMAP_REST_KEY") == ["AMAP_REST_KEY="]
    assert cfg.parse_env_text(merged)["AMAP_REST_KEY"] == ""


# ---------------------------------------------------------------- 6. write_settings / save_settings


def test_write_settings_backs_up_before_overwriting_then_hot_reloads(bare_workspace: Path):
    env_file = bare_workspace / ".env"
    env_file.write_text(ENV_TEXT_WITH_COMMENTS, encoding="utf-8")
    set_env({})

    result = settings_store.write_settings({"AMAP_REST_KEY": "new-rest-key-value"})

    backup = bare_workspace / settings_store.BACKUP_NAME
    assert backup.exists(), "写入前必须先留一份 .env.bak"
    assert backup.read_text(encoding="utf-8") == ENV_TEXT_WITH_COMMENTS  # 备份的是旧内容，不是合并后的
    assert result["backup_file"] == str(backup)
    assert result["written"] == ["AMAP_REST_KEY"]
    assert result["reloaded"] is True
    assert result["env_file"] == str(env_file)
    # 热重载：不重启进程也能读到新值；未托管行按 .env 语义一并进入环境
    assert os.environ["AMAP_REST_KEY"] == "new-rest-key-value"
    assert os.environ["OTHER_THING"] == "keep me exactly"


def test_first_write_without_existing_env_file_creates_it_without_backup(bare_workspace: Path):
    result = settings_store.write_settings({"SILICONFLOW_API_KEY": "sk-fake-1234567890"})
    assert result["backup_file"] == ""
    assert not (bare_workspace / settings_store.BACKUP_NAME).exists()
    assert cfg.parse_env_text((bare_workspace / ".env").read_text(encoding="utf-8"))["SILICONFLOW_API_KEY"]
    assert os.environ["SILICONFLOW_API_KEY"] == "sk-fake-1234567890"


def test_repeated_writes_leave_no_temporary_files(bare_workspace: Path):
    settings_store.write_settings({"AMAP_REST_KEY": "x" * 30})
    settings_store.write_settings({"AMAP_JS_KEY": "y" * 30})
    leftovers = sorted(path.name for path in bare_workspace.iterdir() if ".tmp-" in path.name)
    assert leftovers == [], f"原子写入的临时文件没清干净：{leftovers}"
    assert sorted(path.name for path in bare_workspace.iterdir()) == [".env", ".env.bak"]


def test_save_settings_reports_state_and_client_config_staleness(bare_workspace: Path):
    result = settings_store.save_settings({"AMAP_REST_KEY": FAKE_SECRET})
    assert result["client_config_stale"] is False  # 纯后端项：浏览器不必重新拉配置
    assert result["state"]["missing"] == ["AMAP_SECURITY_CODE"]
    assert entry_of(result["state"], "AMAP_JS_KEY")["reused_from"] == "AMAP_REST_KEY"

    second = settings_store.save_settings({"AMAP_SECURITY_CODE": "sec-code-123"})
    assert second["client_config_stale"] is True  # 前端注入项变了 → 页面要重新取 ZSX_CONFIG
    assert second["state"]["missing"] == []
    assert second["state"]["ready"] is True


def test_cleared_key_returns_to_unconfigured_and_degradation_applies(bare_workspace: Path):
    settings_store.save_settings(
        {"AMAP_REST_KEY": "rest-abcdef", "AMAP_JS_KEY": "js-key-abcdef", "AMAP_SECURITY_CODE": "sec-code-abcdef"}
    )
    state = settings_store.read_state()
    assert state["capabilities"]["amap_map"] is True
    assert state["capabilities"]["amap_js_key_reused"] is False

    cleared = settings_store.save_settings({"AMAP_JS_KEY": "", "AMAP_SECURITY_CODE": ""})
    assert os.environ["AMAP_JS_KEY"] == "" and os.environ["AMAP_SECURITY_CODE"] == ""
    assert assignments_of((bare_workspace / ".env").read_text(encoding="utf-8").splitlines(), "AMAP_JS_KEY") == [
        "AMAP_JS_KEY="
    ]
    caps = cleared["state"]["capabilities"]
    assert caps["amap_plan"] is True  # REST Key 还在
    assert caps["amap_map"] is False
    # 降级判断：JS Key 留空回退复用 REST Key，但仍缺安全密钥 → 底图照样起不来
    assert caps["amap_js_key_reused"] is True
    assert caps["amap_map_blocked_by_security_code"] is True
    assert entry_of(cleared["state"], "AMAP_SECURITY_CODE")["configured"] is False
    assert cleared["state"]["missing"] == ["AMAP_SECURITY_CODE"]

    everything = settings_store.save_settings({"AMAP_REST_KEY": ""})
    assert everything["state"]["capabilities"]["amap_plan"] is False
    assert everything["state"]["ready"] is False
    assert everything["state"]["offline_demo_ready"] is True  # 清光密钥也不影响离线演示


def test_read_state_reports_env_file_location_and_existence(bare_workspace: Path):
    state = settings_store.read_state()
    assert state["env_file"] == str(bare_workspace / ".env")
    assert state["env_file_exists"] is False
    assert state["template_file"] == ".env.example"
    (bare_workspace / ".env").write_text("AMAP_REST_KEY=x\n", encoding="utf-8")
    assert settings_store.read_state()["env_file_exists"] is True


# ---------------------------------------------------------------- 7. is_loopback


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "::1",
        "127.0.0.53",  # 127/8 整段都算本机
        "localhost",
        "Localhost",
        "::ffff:127.0.0.1",  # IPv4-mapped IPv6
        "127.1.2.3",
        "[::1]",
        "::1%eth0",  # 带 scope id 时按地址本体判定
    ],
)
def test_is_loopback_accepts_local_addresses(address: str):
    assert settings_store.is_loopback(address) is True


@pytest.mark.parametrize(
    "address", ["8.8.8.8", "10.0.0.1", "::ffff:8.8.8.8", "192.168.1.20", "0.0.0.0", "example.com", "not-an-ip"]
)
def test_is_loopback_rejects_everything_else(address: str):
    assert settings_store.is_loopback(address) is False


@pytest.mark.parametrize("address", [None, "", "   "], ids=["none", "empty", "blank"])
def test_is_loopback_rejects_missing_address(address: Optional[str]):
    # 拿不到来源地址时按「非本机」处理：写接口的默认拒绝方向必须是安全的
    assert settings_store.is_loopback(address) is False


# ---------------------------------------------------------------- 8. GET /api/settings


def test_get_settings_returns_masked_state(api_client: FlaskClient):
    set_env({"AMAP_REST_KEY": FAKE_SECRET, "SILICONFLOW_API_KEY": "sk-" + FAKE_SECRET})
    resp = api_client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    state = body["data"]
    assert [entry["key"] for entry in state["items"]] == list(settings_store.MANAGED_KEYS)
    assert entry_of(state, "AMAP_REST_KEY")["value_masked"] == settings_store.mask_value(FAKE_SECRET)
    assert entry_of(state, "SILICONFLOW_API_KEY")["value_masked"] == settings_store.mask_value("sk-" + FAKE_SECRET)
    assert_no_plaintext(resp.get_data(as_text=True), FAKE_SECRET)
    assert_no_plaintext(resp.get_data(as_text=True), "sk-" + FAKE_SECRET)


def test_get_settings_offline_demo_ready_is_always_true(api_client: FlaskClient):
    scenarios = [
        {},
        {"AMAP_REST_KEY": FAKE_SECRET},
        {item.key: f"value-{index}-abcdef" for index, item in enumerate(settings_store.SETTINGS)},
    ]
    for values in scenarios:
        set_env(values)
        state = api_client.get("/api/settings").get_json()["data"]
        assert state["offline_demo_ready"] is True, "零密钥可离线演示是不可退化的承诺"


@pytest.mark.parametrize(
    "values, expected_missing, expected_ready, expected_caps",
    [
        ({}, list(REQUIRED_KEYS), False, {"amap_plan": False, "amap_map": False, "llm_real": False}),
        (
            {"AMAP_REST_KEY": FAKE_SECRET},
            ["AMAP_SECURITY_CODE"],
            False,
            {"amap_plan": True, "amap_map": False, "llm_real": False},
        ),
        (
            {"AMAP_REST_KEY": FAKE_SECRET, "AMAP_SECURITY_CODE": "sec-code-1"},
            [],
            True,
            # JS Key 留空 → 按 derived_from 复用 REST Key，安全密钥也就有了配对对象，
            # 所以 amap_map 判定为 True（与 config.amap_config 的回退口径一致）
            {"amap_plan": True, "amap_map": True, "llm_real": False},
        ),
        (
            {"SILICONFLOW_API_KEY": "sk-abcdef"},
            list(REQUIRED_KEYS),
            False,
            {"amap_plan": False, "amap_map": False, "llm_real": True},
        ),
        (
            {"AMAP_REST_KEY": FAKE_SECRET, "AMAP_JS_KEY": "js-key-1", "AMAP_SECURITY_CODE": "sec-1"},
            [],
            True,
            {"amap_plan": True, "amap_map": True, "llm_real": False},
        ),
    ],
    ids=["nothing", "rest-only", "rest-and-security", "llm-only", "all-amap"],
)
def test_get_settings_ready_missing_required_are_self_consistent(
    api_client: FlaskClient,
    values: Mapping[str, str],
    expected_missing: Sequence[str],
    expected_ready: bool,
    expected_caps: Mapping[str, bool],
):
    set_env(values)
    state = api_client.get("/api/settings").get_json()["data"]
    items = state["items"]

    assert state["missing"] == list(expected_missing)
    assert state["ready"] is expected_ready
    # 三条自洽关系：missing 恰是「required 且未配置」；ready 就是「missing 为空」；可选项永不进 missing
    assert state["missing"] == [entry["key"] for entry in items if entry["required"] and not entry["configured"]]
    assert state["ready"] is (not state["missing"])
    assert set(state["missing"]) <= set(REQUIRED_KEYS)
    assert set(state["missing"]) & set(OPTIONAL_KEYS) == set()
    for entry in items:
        assert entry["configured"] is bool(entry["length"])
        assert entry["required"] is item_of(entry["key"]).required

    js_ready = entry_of(state, "AMAP_JS_KEY")["configured"]
    security_ready = entry_of(state, "AMAP_SECURITY_CODE")["configured"]
    caps = state["capabilities"]
    assert {key: caps[key] for key in expected_caps} == dict(expected_caps)
    assert caps["amap_plan"] is entry_of(state, "AMAP_REST_KEY")["configured"]
    assert caps["amap_map"] is (js_ready and security_ready)
    assert caps["amap_js_key_reused"] is bool(entry_of(state, "AMAP_JS_KEY")["reused_from"])
    assert caps["amap_map_blocked_by_security_code"] is (js_ready and not security_ready)
    assert caps["llm_real"] is entry_of(state, "SILICONFLOW_API_KEY")["configured"]


# ---------------------------------------------------------------- 9. POST /api/settings


def test_post_settings_from_non_loopback_is_denied_and_writes_nothing(api_client: FlaskClient, bare_workspace: Path):
    (bare_workspace / ".env").write_text(ENV_TEXT_WITH_COMMENTS, encoding="utf-8")
    resp = api_client.post(
        "/api/settings",
        json={"AMAP_REST_KEY": FAKE_SECRET},
        environ_overrides={"REMOTE_ADDR": "203.0.113.9"},
    )
    assert resp.status_code == 403
    body = resp.get_json()
    assert body["ok"] is False
    assert "本机" in body["error"]
    # detail 必须可操作：点名来源 + 给出两条替代路径
    detail = body["detail"]
    assert "203.0.113.9" in detail
    assert "loopback" in detail
    assert ".env" in detail and "重启" in detail
    assert (bare_workspace / ".env").read_text(encoding="utf-8") == ENV_TEXT_WITH_COMMENTS
    assert not (bare_workspace / settings_store.BACKUP_NAME).exists()
    assert "AMAP_REST_KEY" not in os.environ  # 拒绝发生在落盘之前


def test_post_settings_ignores_spoofed_forwarded_header(api_client: FlaskClient):
    resp = api_client.post(
        "/api/settings",
        json={"AMAP_REST_KEY": FAKE_SECRET},
        headers={"X-Forwarded-For": "127.0.0.1"},
        environ_overrides={"REMOTE_ADDR": "198.51.100.7"},
    )
    assert resp.status_code == 403
    assert "198.51.100.7" in resp.get_json()["detail"]  # 只认 socket 直连地址


def test_post_settings_without_source_address_is_denied(api_client: FlaskClient):
    resp = api_client.post(
        "/api/settings", json={"AMAP_REST_KEY": FAKE_SECRET}, environ_overrides={"REMOTE_ADDR": ""}
    )
    assert resp.status_code == 403
    assert "未知" in resp.get_json()["detail"]


def test_post_settings_is_disabled_when_readonly(readonly_client: FlaskClient, bare_workspace: Path):
    resp = readonly_client.post("/api/settings", json={"AMAP_REST_KEY": FAKE_SECRET})
    assert resp.status_code == 403
    body = resp.get_json()
    assert "生产模式" in body["error"]
    assert "waitress" in body["detail"] and ".env" in body["detail"]
    assert not (bare_workspace / ".env").exists()


def test_loopback_guard_is_checked_before_the_readonly_switch(readonly_client: FlaskClient):
    """两个 403 的理由不同：来源判定在前，写入开关在后（别针错方向）。"""
    resp = readonly_client.post(
        "/api/settings", json={"AMAP_REST_KEY": FAKE_SECRET}, environ_overrides={"REMOTE_ADDR": "10.10.10.10"}
    )
    assert resp.status_code == 403
    assert "本机" in resp.get_json()["error"]


def test_readonly_client_can_still_read_settings(readonly_client: FlaskClient):
    assert readonly_client.get("/api/settings").status_code == 200


def test_get_settings_stays_readable_for_lan_clients_and_still_masks(api_client: FlaskClient):
    """读接口对局域网 PWA 开放是设计决定（值一律掩码）；写接口才限本机。"""
    set_env({"AMAP_REST_KEY": FAKE_SECRET})
    resp = api_client.get("/api/settings", environ_overrides={"REMOTE_ADDR": "192.168.1.20"})
    assert resp.status_code == 200
    assert entry_of(resp.get_json()["data"], "AMAP_REST_KEY")["value_masked"] == settings_store.mask_value(FAKE_SECRET)
    assert_no_plaintext(resp.get_data(as_text=True), FAKE_SECRET)


@pytest.mark.parametrize(
    "value",
    [True, 12, ["a", "b"], {"nested": "obj"}],
    ids=["bool", "int", "array", "object"],
)
def test_post_settings_with_non_string_values_never_500s(api_client: FlaskClient, value: Any):
    """非字符串值不许把服务打崩：实现目前按 str() 兜住（要么存要么 400）。"""
    resp = api_client.post("/api/settings", json={"AMAP_SECURITY_CODE": value})
    assert resp.status_code in (200, 400), resp.get_data(as_text=True)
    assert resp.get_json()["ok"] is (resp.status_code == 200)


@pytest.mark.parametrize(
    "payload",
    [
        {"AMAP_REST_KEY": "ok\nAMAP_SECURITY_CODE=evil"},
        {"AMAP_JS_KEY": "ok\rAMAP_SECURITY_CODE=evil"},
        {"AMAP_JS_KEY": "ok;OTHER=1"},
        {"AMAP_TIMEOUT": "10\nSILICONFLOW_API_KEY=sk-leak"},
        {"AMAP_REST_KEY": "x" * 500},
    ],
    ids=["newline", "carriage_return", "equals", "newline-in-numeric", "overlong"],
)
def test_post_settings_rejects_injection_payloads_without_touching_the_file(
    api_client: FlaskClient, bare_workspace: Path, payload: Mapping[str, str]
):
    (bare_workspace / ".env").write_text(ENV_TEXT_WITH_COMMENTS, encoding="utf-8")
    resp = api_client.post("/api/settings", json=payload)
    assert resp.status_code == 400
    assert api_client.get("/api/settings").get_json()["data"]["env_file_exists"] is True
    # 整批拒绝：坏值不许拖上好值一起落盘
    assert (bare_workspace / ".env").read_text(encoding="utf-8") == ENV_TEXT_WITH_COMMENTS
    assert not (bare_workspace / settings_store.BACKUP_NAME).exists()


def test_post_settings_rejects_unknown_key(api_client: FlaskClient):
    resp = api_client.post("/api/settings", json={"NOT_A_SETTING": "x"})
    assert resp.status_code == 400
    message = resp.get_json()["error"]
    assert "NOT_A_SETTING" in message
    for key in settings_store.MANAGED_KEYS:
        assert key in message


@pytest.mark.parametrize(
    "body, expected",
    [([{"AMAP_REST_KEY": "v"}], "JSON 对象"), ("text", "JSON 对象"), ({}, "没有需要保存")],
    ids=["array", "plain-text", "empty-object"],
)
def test_post_settings_rejects_non_object_bodies(api_client: FlaskClient, body: Any, expected: str):
    if isinstance(body, str):
        resp = api_client.post("/api/settings", data=body, content_type="text/plain")
    else:
        resp = api_client.post("/api/settings", json=body)
    assert resp.status_code == 400
    assert expected in resp.get_json()["error"]


def test_post_settings_applies_hot_and_flags_frontend_refresh(api_client: FlaskClient, bare_workspace: Path):
    (bare_workspace / ".env").write_text(ENV_TEXT_WITH_COMMENTS, encoding="utf-8")

    resp = api_client.post("/api/settings", json={"AMAP_REST_KEY": FAKE_SECRET, "AMAP_TIMEOUT": "15"})
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    # 响应契约（按实现断言）：服务端一律热生效 reloaded=True，「需重启」只体现在前端要不要重取配置
    assert set(data) == {"written", "env_file", "backup_file", "reloaded", "client_config_stale", "state"}
    assert data["written"] == ["AMAP_REST_KEY", "AMAP_TIMEOUT"]
    assert data["reloaded"] is True
    assert data["client_config_stale"] is False  # 后端项：不需要浏览器刷新
    assert os.environ["AMAP_REST_KEY"] == FAKE_SECRET and os.environ["AMAP_TIMEOUT"] == "15"
    assert data["backup_file"] == str(bare_workspace / settings_store.BACKUP_NAME)
    lines = (bare_workspace / ".env").read_text(encoding="utf-8").splitlines()
    assert assignments_of(lines, "AMAP_REST_KEY") == [f"AMAP_REST_KEY={FAKE_SECRET}"]
    assert "# 智枢星本地配置（手写的，别丢）" in lines
    assert data["state"]["missing"] == ["AMAP_SECURITY_CODE"]
    assert_no_plaintext(resp.get_data(as_text=True), FAKE_SECRET)

    follow_up = api_client.post("/api/settings", json={"AMAP_JS_KEY": "js-" + FAKE_SECRET})
    follow_data = follow_up.get_json()["data"]
    assert follow_up.status_code == 200
    assert follow_data["reloaded"] is True and follow_data["client_config_stale"] is True
    assert_no_plaintext(follow_up.get_data(as_text=True), FAKE_SECRET)

    # 保存结果立刻被规划链路读到（同一进程、无重启）
    plan = api_client.post("/api/plan", json={"question": "从A口到地铁闸机", "engine": "hub"})
    assert plan.status_code == 200


def test_saved_settings_are_read_back_by_reload_env(bare_workspace: Path):
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as api_client:
        api_client.post("/api/settings", json={"AMAP_REST_KEY": "reloaded-value"})
    assert cfg.reload_env()["AMAP_REST_KEY"] == "reloaded-value"
    assert os.environ["AMAP_REST_KEY"] == "reloaded-value"
    assert settings_store.effective_value(item_of("AMAP_REST_KEY")) == ("reloaded-value", False)


# ---------------------------------------------------------------- 10. cli doctor


def run_doctor(capsys: pytest.CaptureFixture[str], argv: Sequence[str] = ("doctor",)) -> Tuple[int, str]:
    code = cli_main(list(argv))
    return code, capsys.readouterr().out


def conclusion_of(output: str) -> str:
    lines = [line for line in output.splitlines() if "结论" in line]
    assert len(lines) == 1, f"结论行应当唯一：{lines}"
    return lines[0]


def test_doctor_exits_1_and_names_missing_required_items(bare_workspace: Path, capsys):
    code, output = run_doctor(capsys)
    assert code == 1, "必需项缺失时退出码必须是 1"
    conclusion = conclusion_of(output)
    assert "必需项缺失" in conclusion
    for key in REQUIRED_KEYS:
        assert key in conclusion  # 结论要直接点名缺哪几项
    assert str(len(REQUIRED_KEYS)) in conclusion
    for key in REQUIRED_KEYS:  # 未配置的必需项都要给申请入口与当前降级说明
        assert item_of(key).apply_entry in output
        assert item_of(key).degrades_to in output
    assert "离线演示" in output and "始终可用" in output


def test_doctor_exits_0_once_required_items_present(bare_workspace: Path, capsys):
    set_env({"AMAP_REST_KEY": "rest-value-123456", "AMAP_SECURITY_CODE": "sec-value-123456"})
    code, output = run_doctor(capsys)
    assert code == 0
    assert "必需项已全部配置" in conclusion_of(output)
    assert "复用 AMAP_REST_KEY" in output  # JS Key 留空复用 REST，要说清楚
    assert "降级为 Mock 确定性模板" in output


def test_doctor_exits_0_and_reports_everything_filled(bare_workspace: Path, capsys):
    set_env({item.key: f"value-{index}-abcdef" for index, item in enumerate(settings_store.SETTINGS)})
    code, output = run_doctor(capsys)
    assert code == 0
    assert "全部配置项均已填写" in conclusion_of(output)


def test_doctor_strict_counts_optional_blanks(bare_workspace: Path, capsys):
    set_env({"AMAP_REST_KEY": "rest-value-123456", "AMAP_SECURITY_CODE": "sec-value-123456"})
    code, output = run_doctor(capsys, ("doctor", "--strict"))
    assert code == 1
    conclusion = conclusion_of(output)
    assert "可选项缺失" in conclusion
    for key in OPTIONAL_KEYS:
        assert key in conclusion
    # 非 strict 的默认口径不许把可选项当阻断项
    assert run_doctor(capsys)[0] == 0


def test_doctor_output_never_prints_secret_plaintext(bare_workspace: Path, capsys):
    set_env({"AMAP_REST_KEY": FAKE_SECRET, "SILICONFLOW_API_KEY": "sk-" + FAKE_SECRET})
    code, output = run_doctor(capsys)
    assert code == 1
    assert_no_plaintext(output, FAKE_SECRET)
    assert_no_plaintext(output, "sk-" + FAKE_SECRET)
    # 掩码形式必须出现，否则就是整行漏印而不是掩码生效
    assert settings_store.mask_value(FAKE_SECRET) in output
    assert "[已配置]" in output


def test_doctor_uses_redirected_workspace_and_flags_missing_env_file(bare_workspace: Path, capsys):
    code, output = run_doctor(capsys)
    assert code == 1
    assert str(bare_workspace) in output
    assert "不存在" in output and ".env.example" in output


# ---------------------------------------------------------------- 11. 承诺级回归：零密钥全链路离线可演示

CHILD_PROMISE_SCRIPT = '''"""全新进程 + 没有任何 .env 的 workspace：证明零密钥仍能起服务并出结果。"""
import json
import sys
from pathlib import Path

from zhishuxing import config as cfg
from zhishuxing.webapp.app import create_app
from zhishuxing.webapp.service import ZhiShuXingWebService

expected_name = sys.argv[1]
assert cfg.WORKSPACE_ROOT.name == expected_name, (cfg.WORKSPACE_ROOT, expected_name)
assert not (cfg.WORKSPACE_ROOT / ".env").exists(), "测试用 workspace 不该有 .env"

service = ZhiShuXingWebService()
app = create_app(service=service)
app.config["TESTING"] = True
client = app.test_client()

state = client.get("/api/settings").get_json()["data"]
first = client.post("/api/chat", json={"message": "从A口到地铁闸机"}).get_json()["data"]
second = client.post("/api/chat", json={"message": "从A口到地铁闸机"}).get_json()["data"]
intercity = client.post("/api/chat", json={"message": "从深圳北站到宝安机场,赶时间"}).get_json()["data"]
hub_plan = client.post("/api/plan", json={"question": "从A口经主安检到地铁闸机", "engine": "hub"})
amap_plan = client.post("/api/plan", json={"question": "从深圳北站到宝安机场", "engine": "amap"})
index_html = client.get("/").get_data(as_text=True)

print("ZSX_RESULT:" + json.dumps({
    "cwd": str(Path.cwd()),
    "workspace_root": str(cfg.WORKSPACE_ROOT),
    "env_file": state["env_file"],
    "env_file_exists": state["env_file_exists"],
    "configured": [entry["key"] for entry in state["items"] if entry["configured"]],
    "missing": state["missing"],
    "ready": state["ready"],
    "offline_demo_ready": state["offline_demo_ready"],
    "capabilities": state["capabilities"],
    "health": client.get("/health").status_code,
    "llm_adapter": type(service.system.llm).__name__,
    "llm_mock": bool(getattr(service.system.llm, "mock", False)),
    "chat_engine": first["engine"],
    "chat_reply": first["reply"],
    "chat_route_first": first["route"]["route"][0],
    "chat_kb_refs": len(first["kb_refs"]),
    "chat_deterministic": first["reply"] == second["reply"],
    "intercity_status_ok": bool(intercity["reply"]),
    "intercity_route_error": bool(intercity.get("route_error")),
    "hub_plan_status": hub_plan.status_code,
    "amap_plan_status": amap_plan.status_code,
    "amap_plan_error": amap_plan.get_json().get("error", ""),
    "index_injects_empty_js_key": 'amapJsKey: ""' in index_html,
    "index_has_amap_container": "amapContainer" in index_html,
}, ensure_ascii=False))
'''


def scrubbed_child_env(workspace: Path) -> Dict[str, str]:
    """子进程环境：抹掉全部托管键 + 指向空 workspace = 全新 clone、一个密钥都没配。"""
    env = {key: value for key, value in os.environ.items() if key not in MANAGED_KEYS_SET}
    env.update(
        {
            "ZHISHUXING_WORKSPACE": str(workspace),
            "PYTHONPATH": str(ORIGINAL_WORKSPACE_ROOT / "src"),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    env.pop("PYTHONSTARTUP", None)
    return env


@pytest.fixture()
def fresh_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """只带 configs 与知识库语料、**没有 .env** 的 workspace（子进程冷启动用）。"""
    root = tmp_path_factory.mktemp("zsx-bare-workspace")
    shutil.copytree(ORIGINAL_WORKSPACE_ROOT / "configs", root / "configs")
    kb_dir = ORIGINAL_WORKSPACE_ROOT / "data" / "transfer_kb"
    if kb_dir.exists():
        shutil.copytree(kb_dir, root / "data" / "transfer_kb")
    assert not (root / ".env").exists()
    return root


def test_offline_demo_promise_holds_in_fresh_process_without_any_env_file(fresh_workspace: Path):
    """承诺级回归（子进程冷启动）：没有 .env、没有任何密钥，服务照起、对话照出确定性结果。"""
    script = fresh_workspace / "promise_check.py"
    script.write_text(CHILD_PROMISE_SCRIPT, encoding="utf-8")
    # cwd 也放在这个 workspace 里：顺便钉住「路径锚定 workspace、不依赖进程 CWD」
    completed = subprocess.run(
        [sys.executable, str(script), fresh_workspace.name],
        capture_output=True,
        cwd=str(fresh_workspace),
        env=scrubbed_child_env(fresh_workspace),
        timeout=300,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode == 0, f"零密钥冷启动失败 rc={completed.returncode}\n{stdout}\n{stderr}"
    lines = [line for line in stdout.splitlines() if line.startswith("ZSX_RESULT:")]
    assert len(lines) == 1, f"没拿到结果行\n{stdout}\n{stderr}"
    result = json.loads(lines[0][len("ZSX_RESULT:") :])

    # 冷启动读的是被重定向的空 workspace，绝不可能是真实 .env
    assert result["workspace_root"] == str(fresh_workspace)
    assert result["env_file"] == str(fresh_workspace / ".env")
    assert str(ORIGINAL_WORKSPACE_ROOT) not in result["env_file"]
    assert result["cwd"] == str(fresh_workspace)
    assert result["env_file_exists"] is False
    assert result["configured"] == []
    assert result["missing"] == list(REQUIRED_KEYS)
    assert result["ready"] is False
    assert result["offline_demo_ready"] is True
    # 三项在线能力全部降级，但没有一项把服务或接口拖垮
    assert result["capabilities"] == {
        "amap_plan": False,
        "amap_map": False,
        "amap_js_key_reused": False,
        "amap_map_blocked_by_security_code": False,
        "llm_real": False,
    }
    assert result["health"] == 200
    assert result["llm_adapter"] == "MockLLMAdapter" and result["llm_mock"] is True
    assert result["chat_engine"] == "hub"  # 内置枢纽引擎兜底
    assert result["chat_route_first"] == [1, 2]
    assert result["chat_kb_refs"] > 0  # 站内经验引用照样有
    assert "当前为本地模板模式" in result["chat_reply"]  # 回答来自 Mock 确定性模板
    assert result["chat_deterministic"] is True
    assert result["intercity_route_error"] is True and result["intercity_status_ok"] is True
    assert result["hub_plan_status"] == 200
    assert result["amap_plan_status"] == 400
    assert "AMAP_REST_KEY" in result["amap_plan_error"]
    assert result["index_injects_empty_js_key"] is True  # 底图回退 Canvas 折线
    assert result["index_has_amap_container"] is True


def test_offline_demo_promise_holds_in_process_with_scrubbed_environment(bare_workspace: Path):
    """同一承诺的进程内快版本：夹具态（无 .env、无密钥）下 create_app 与对话链路可用。"""
    assert not cfg.env_file_path().exists()
    service = ZhiShuXingWebService()
    assert isinstance(service.system.llm, MockLLMAdapter)
    assert service.system.llm.mock is True  # LLM 适配器状态：降级

    app = create_app(service=service)
    app.config["TESTING"] = True
    with app.test_client() as client:
        state = client.get("/api/settings").get_json()["data"]
        assert state["offline_demo_ready"] is True
        assert state["capabilities"]["llm_real"] is False

        replies = [
            client.post(
                "/api/chat", json={"message": "带老人行李多,优先直梯,去地铁前先上趟卫生间,从A口出发"}
            ).get_json()["data"]
            for _ in range(2)
        ]
        first = replies[0]
        assert first["engine"] == "hub"
        assert "need_restroom" in first["profile"]["hard"]  # 规则解析在 Mock 下依然生效
        assert "prefer_elevator" in first["profile"]["soft"]
        assert "elderly" in first["profile"]["persona"]
        assert first["route"]["route"][0] == [1, 2]
        assert first["kb_refs"]
        assert "已理解您的需求" in first["reply"]
        assert "当前为本地模板模式" in first["reply"]
        # 确定性：同一输入必须同一输出，两条回复逐字相同
        assert replies[0]["reply"] == replies[1]["reply"]

        plan = client.post("/api/plan", json={"question": "从A口到地铁闸机", "engine": "hub"})
        assert plan.status_code == 200
        assert plan.get_json()["data"]["engine"] == "hub"

        failing = client.post("/api/plan", json={"question": "从深圳北站到宝安机场", "engine": "amap"})
        assert failing.status_code == 400
        assert "AMAP_REST_KEY" in failing.get_json()["error"]

        assert settings_store.read_state()["env_file_exists"] is False


# ---------------------------------------------------------------- 12. 隔离自检（务必放在本文件最后）


def test_module_leaves_the_shared_environment_as_it_found_it() -> None:
    """自检：跑完整模块的写入/热重载之后，托管键仍然等于进本模块前的快照。

    这是「不污染同会话其他测试」的可执行证明 —— 任何一处漏还原，本用例就会红。
    """
    current = {key: os.environ.get(key) for key in settings_store.MANAGED_KEYS}
    assert current == ENVIRONMENT_SNAPSHOT_AT_IMPORT
    assert os.environ.get("OTHER_THING") is None, "未托管行被热重载写进了环境且没还原"
    assert cfg.WORKSPACE_ROOT == ORIGINAL_WORKSPACE_ROOT, "workspace 重定向没还原"
