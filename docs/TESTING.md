# 智枢星 测试规范

> 用途：给写测试和跑门禁的人与 AI。测试数的事实口径在根目录 `AGENTS.md` 的「当前状态」。

## 测试分层

| 层 | 管什么 | 数量级 |
|---|---|---|
| pytest 单元/集成 | CLI、配置、导航、API、KB | 201 项（`test_settings.py` 一文件占 98） |
| pyflakes | 静态检查（src/ scripts/ tests/） | 0 告警 |
| API 冒烟 | 进程内打全部核心端点 | `zhishuxing smoke` |
| 配置体检 | 在线能力与降级状态 | `zhishuxing doctor` |
| 报告回归 | 7 类分析报告 | `zhishuxing analyze --report all`，7 行 OK |
| CI | ubuntu × Python 3.11/3.12，pyflakes → pytest（`MPLBACKEND=Agg`） | `.github/workflows/ci.yml` |

## 运行命令

```bash
python -m pytest                                # 201 passed（20–50 s，受机器负载影响）
python -m pyflakes src/ scripts/ tests/         # 0 告警
zhishuxing smoke                                # Web smoke test passed.
zhishuxing doctor                               # 本机配好密钥 → 退出码 0；全新 clone → 退出码 1
zhishuxing analyze --report all                 # 7 类报告全 OK
```

`python -m pytest -q` 会变 `-qq`（`pyproject` 的 `addopts` 已含 `-q`）看不到统计行——要计数就
用不带 `-q` 的命令，或 `python -m pytest -o addopts="" --collect-only -q`。

## 用例编写规范

- 测试与 smoke 全程离线确定性：真实 `.env` 密钥不得在测试里发起网络调用（自动挂载放在 serve
  路径而非 `create_app` 就是为了这个）。
- `test_settings.py` 钉着 `.env` 写回的五道闸与 loopback 判定——放宽任何一道都是回归。
- 测试会写 `data/outputs/`（未跟踪，跑完 `git status` 仍干净）；不要把某张图当基线提交进
  `samples/`，除非 README 真要引用它。
- `legacy/` 与 `third_party/` 不参与测试与 pyflakes（有意，见 `docs/ARCHITECTURE.md` 关键约定）。

## 改动后的验证

| 你动了 | 必须跑 |
|---|---|
| `src/zhishuxing/**` | `python -m pytest` + `python -m pyflakes src/ scripts/ tests/` |
| `webapp/app.py` 的端点或返回结构 | `zhishuxing smoke` + `pytest tests/test_api.py`，并同步 `static/app.js` |
| `settings.py` / `config.py` | `pytest tests/test_settings.py` + `zhishuxing doctor` |
| `core/navigation.py` 的规划或代价 | `pytest tests/test_navigation.py tests/test_navigation_prefs.py tests/test_simulation.py` |
| `configs/hub_default.json` | `zhishuxing smoke`（真的会规划一次 `[1,2]→[28,12]`） |
| `data/transfer_kb/shenzhen_north/` | `zhishuxing kb-ingest --query "带老人 优先直梯"` 再 `pytest tests/test_kb.py` |
| `analysis/**` 或报告 | `zhishuxing analyze --report all`（7 行 OK） |
| `core/animation.py` 的力场或放行窗口 | `python code_optimization/benchmark_animation.py` 并更新 `report.md`（会改写入库 JSON） |
| `unity/HubTransferAgent.cs` | 无自动化：按 `unity/README.md` 契约与 `rl/envs.py` 手工核对 |
| 任何一级目录结构 | `python check_docs.py zhishuxing`（在 `Project/文档标准/` 执行） |

改完 `pyproject.toml` 或 `config.py` 后重装一次 `pip install -e .`，否则旧 egg-info 元数据会
掩盖新增入口。
