"""部署验收:本地体检(默认)或对运行中实例的远程探针(--url),产出可签字的验收报告。

输出契约(与姊妹项目 harness 的 verify 同口径,便于 CI 与人眼双消费):
- 每项一行 `  PASS|WARN|FAIL  名称(细节)`,分节标题 `[N] 中文描述`;
- 末行 `VERIFY PASS:...` 或 `校验未通过:[...]`,退出码 0/1;
- WARN 通道用于「不阻塞上线但要如实标注」的项(试用模式、可选密钥缺失、
  镜像未随附参考样本、未启用管理鉴权)。

本地模式在进程内构建应用打核心端点(对齐 smoke 的主干,省去最慢的 RL 全仿真);
远程模式用 HTTP 探测部署实例——容器内执行 `zhishuxing verify --url http://127.0.0.1:7860`
即为标准的部署后验收,客户机器无需任何 Python 环境。
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config as cfg
from . import settings as settings_store

# 远程模式的掩码形态:非空掩码必须形如 `ab***(长度 n)`;出现长明文即判泄漏
_MASKED_SHAPE = re.compile(r"^(\*\*\*|\S{0,2}\*\*\*)")


class _Checker:
    """PASS/WARN/FAIL 三态检查器:收集行、计数,末尾给结论。"""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def section(self, title: str) -> None:
        self.lines.append(f"[{title}]")

    def check(self, name: str, cond: bool, detail: str = "", warn_only: bool = False) -> bool:
        mark = "PASS" if cond else ("WARN" if warn_only else "FAIL")
        self.lines.append(f"  {mark}  {name}" + (f"({detail})" if detail else ""))
        if not cond:
            (self.warnings if warn_only else self.failures).append(name)
        return cond

    def summary(self) -> Tuple[bool, str]:
        if self.failures:
            suffix = f"(警告 {len(self.warnings)} 条)" if self.warnings else ""
            return False, f"校验未通过:{self.failures}{suffix}"
        suffix = f"(警告 {len(self.warnings)} 条:{self.warnings})" if self.warnings else ""
        return True, f"VERIFY PASS:部署验收全部通过。{suffix}"


def _leak_check(state: Dict[str, Any], body: str) -> str:
    """密钥泄漏检查(本地模式):逐项取运行时明文,出现在响应体里即泄漏。

    例外:配置值恰好等于公开默认值时不算泄漏(响应里的 default 字段本就含它,
    默认值不是秘密)。"""
    leaked = []
    for item in state["items"]:
        entry = settings_store.SETTINGS_BY_KEY[item["key"]]
        value, _ = settings_store.effective_value(entry)
        if value and value != entry.default and value in body:
            leaked.append(item["key"])
    return ", ".join(leaked)


def _local_checks(checker: _Checker, client: Any) -> None:
    from .core.navigation import NavigationAdapter
    from .core.scenarios import resolve_groups

    state = settings_store.read_state()

    checker.section("1 配置与授权")
    missing = state["missing"]
    checker.check(
        "必需密钥配置",
        not missing,
        "已全部配置" if not missing else f"缺失 {', '.join(missing)}(对应在线能力受限,按合同判定)",
        warn_only=True,
    )
    checker.check("离线演示承诺", bool(state["offline_demo_ready"]))
    lic = state["license"]
    checker.check(
        "授权状态",
        lic["status"] in ("valid", "expiring", "grace"),
        lic["message"],
        warn_only=lic["status"] in ("missing", "invalid"),
    )

    checker.section("2 Web 核心端点(进程内)")
    health = client.get("/health")
    checker.check("GET /health", health.status_code == 200)
    nav = client.post("/api/navigation/load", json={"file_path": str(cfg.paths.navigation_config)})
    checker.check("POST /api/navigation/load", nav.status_code == 200)
    plan = client.post("/api/navigation/plan", json={"start": [1, 2], "goal": [28, 12], "via": ["security"]})
    checker.check("POST /api/navigation/plan", plan.status_code == 200)
    checker.check("GET /api/scenarios", client.get("/api/scenarios").status_code == 200)
    checker.check("GET /api/rl/status", client.get("/api/rl/status").status_code == 200)
    rl_act = client.post(
        "/api/rl/act", json={"observations": [[0.5, 0.25, 1.0, 0.0, 0.2, 0.1, -0.3, 0.4, 0.0, -0.2]]}
    )
    checker.check("POST /api/rl/act", rl_act.status_code == 200)
    settings_resp = client.get("/api/settings")
    if checker.check("GET /api/settings", settings_resp.status_code == 200):
        leaked = _leak_check(state, settings_resp.data.decode("utf-8"))
        checker.check("密钥不泄漏", not leaked, f"响应含明文: {leaked}" if leaked else "全部掩码")
    checker.check("GET /mobile(PWA)", client.get("/mobile").status_code == 200)
    nav_map = NavigationAdapter()
    nav_map.load_navigation(str(cfg.paths.navigation_config))
    payload = json.loads(cfg.paths.scenarios_config.read_text(encoding="utf-8"))["groups"]
    groups = [g.to_payload() for g in resolve_groups(payload, nav_map.map)]
    dash = client.post("/api/dashboard/run", json={"groups": groups, "title": "verify"})
    checker.check("POST /api/dashboard/run(面板渲染)", dash.status_code == 200)

    checker.section("3 数据与运行环境")
    for name, path in (("outputs", cfg.paths.outputs), ("runs", cfg.paths.runs_dir), ("model", cfg.paths.model_dir)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".verify_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checker.check(f"数据目录可写: {name}", True)
        except OSError as exc:
            checker.check(f"数据目录可写: {name}", False, str(exc))
    checker.check(
        "参考样本(data/samples)",
        cfg.paths.samples.exists(),
        "缺失:奖励曲线报告回退合成数据(交付镜像按设计未随附)" if not cfg.paths.samples.exists() else "",
        warn_only=True,
    )

    checker.section("4 管理端鉴权自检")
    if state["admin_auth_enabled"]:
        blocked = client.post("/api/features/run_existing", json={})
        checker.check("未登录访问管理端点被拒", blocked.status_code == 401, f"实际 {blocked.status_code}")
    else:
        checker.check(
            "管理端鉴权", False, "未设置 ADMIN_PASSWORD:管理端点对内网开放,生产部署应设置", warn_only=True
        )


def _remote_checks(checker: _Checker, fetch: Callable[[str, str], Tuple[int, str]], base_url: str) -> None:
    checker.section(f"1 远程探针({base_url})")
    status, body = fetch("GET", "/health")
    checker.check("GET /health", status == 200 and '"status":"ok"' in body.replace(" ", ""))
    checker.check("GET /", fetch("GET", "/")[0] == 200)
    checker.check("GET /mobile(PWA)", fetch("GET", "/mobile")[0] == 200)

    status, body = fetch("GET", "/api/settings")
    state: Dict[str, Any] = {}
    if checker.check("GET /api/settings", status == 200):
        try:
            state = json.loads(body)["data"]
        except (ValueError, KeyError):
            state = {}
        if state:
            # 空掩码 = 未配置(合法);出现非空且不像掩码的长值才算疑似明文
            bad = [
                item["key"]
                for item in state.get("items", [])
                if item.get("value_masked") and not _MASKED_SHAPE.match(item["value_masked"])
            ]
            checker.check("密钥不泄漏(掩码形态)", not bad, f"疑似明文: {', '.join(bad)}" if bad else "全部掩码")
            lic = state.get("license", {})
            checker.check(
                "授权状态",
                lic.get("status") in ("valid", "expiring", "grace"),
                str(lic.get("message", "实例未返回授权状态")),
                warn_only=lic.get("status") in ("missing", "invalid", ""),
            )

    checker.section("2 管理端鉴权自检")
    if state.get("admin_auth_enabled"):
        status, _ = fetch("POST", "/api/features/run_existing")
        checker.check("未登录访问管理端点被拒", status == 401, f"实际 {status}")
    else:
        checker.check(
            "管理端鉴权", False, "实例未启用(未设置 ADMIN_PASSWORD):生产部署应设置", warn_only=True
        )


def _write_report(report_dir: Optional[Path], mode: str, target: str, checker: _Checker, ok: bool, conclusion: str) -> Path:
    directory = Path(report_dir) if report_dir else cfg.paths.outputs
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"verify-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    lines = [
        "# 智枢星 部署验收报告",
        "",
        f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模式: {mode}" + (f" · {target}" if target else ""),
        f"- 结论: {'通过' if ok else '未通过'} — {conclusion}",
        "",
        "```",
    ]
    lines.extend(checker.lines)
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_verify(url: Optional[str] = None, timeout: int = 5, report_dir: Optional[Path] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # 非 tty/旧版本 Python 无需重配

    checker = _Checker()
    if url:
        import requests

        def fetch(method: str, path: str) -> Tuple[int, str]:
            resp = requests.request(method, url.rstrip("/") + path, timeout=timeout)
            return resp.status_code, resp.text

        _remote_checks(checker, fetch, url)
    else:
        from .webapp.app import create_app

        _local_checks(checker, create_app().test_client())

    ok, conclusion = checker.summary()
    for line in checker.lines:
        print(line)
    print()
    print(conclusion)
    try:
        report = _write_report(report_dir, "远程" if url else "本地", url or "", checker, ok, conclusion)
        print(f"验收报告: {report}")
    except OSError as exc:
        print(f"验收报告写入失败({exc});不影响验收结论。")
    return 0 if ok else 1
