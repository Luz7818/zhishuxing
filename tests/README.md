# tests/ —— pytest 套件

> 用途：说明 180 个用例分别钉住哪条链路、哪些用例需要 torch、以及加用例时的隔离要求。
> 复核口径：`python -m pytest -o addopts="" --collect-only -q` 计数，`python -m pytest` 跑。

## 当前规模

| 文件 | 用例数 | 钉住什么 |
|---|---|---|
| `test_settings.py` | 98 | 配置注册表、掩码、写入校验、`.env` 合并与原子写、热重载、loopback 判定、`doctor` 退出码，以及"无密钥离线可演示"这条承诺 |
| `test_api.py` | 19 | 全部 HTTP 端点：导航、RL、LLM、面板、报告、`/api/plan` 两个引擎、`/api/chat` 的偏好理解与无 Key 降级、`/mobile` 页面 |
| `test_profile.py` | 12 | 需求档案：规则解析、LLM JSON 解析、多轮合并、`to_cost_spec()` 翻译 |
| `test_amap.py` | 9 | OD 正则提取、新旧两代高德响应结构的折线与详情解析、`resolve_strategy()`、`.env` 解析器 |
| `test_navigation_prefs.py` | 8 | 偏好加权 A*、设施硬约束与降级、软必经点取舍、旧格式导航图兼容 |
| `test_navigation.py` | 7 | A* 基础正确性：可达、绕障、必经地标拼接、非法坐标报错 |
| `test_synthetic.py` | 7 | 四类合成数据的形状、取值范围与同种子可复现 |
| `test_kb.py` | 6 | 入库幂等、BM25 相关性排序、`hub` 过滤 |
| `test_rl_runtime.py` | 6 | 权重扫描与命名解析、加载后 `act()` 输出维度、无权重时的启发式回退（**需要 torch**） |
| `test_simulation.py` | 5 | 引导仿真：动作语义、到达统计、拥堵峰值、分组明细 |
| `test_cli.py` | 3 | `analyze`、`simulate`、`demo` 三个子命令的真实退出码与产物 |

## 依赖边界

- `test_rl_runtime.py` 第 8 行是 `torch = pytest.importorskip("torch", ...)`，
  没装 torch 时这 6 个用例跳过而不是失败。CI 里显式装了 CPU 轮子（`.github/workflows/ci.yml`），
  所以本机跑出 `skipped` 说明你少装了 torch，不是代码坏了。
- 其余用例不需要 torch、不需要 `mlagents_envs`、不需要网络与任何密钥。
- `conftest.py` 提供三个 fixture：`navigation`（session 级）、`service`（session 级）、
  `client`（每个用例新建 app）。

## 加用例时的隔离要求

`test_settings.py` 顶部注释把这套隔离写成了文档，新增涉及 `.env` 或用例内起服务的测试要照做，
三条一起用，缺一条就会污染同一次 pytest 会话：

1. **workspace 重定向**：进程内用 `monkeypatch` 改 `config.WORKSPACE_ROOT`；
   子进程用环境变量 `ZHISHUXING_WORKSPACE`。
   注意 `cfg.paths` 是 import 期算好的，仍指向真实仓库的 `configs/` 与 `data/`，
   这是故意的 —— 起服务需要它们。
2. **`os.environ` 整体快照 + 差异还原**：热重载会写进程环境，不还原会泄漏到后面的用例。
3. **不要往真实 workspace 写 `.env`**：`env_file_path()` 每次调用都读模块属性，
   所以只要第 1 条做对了就写不到仓库里。

## 已知会改动产物的用例

`test_cli.py` 的 `analyze` / `demo` 与 `test_api.py` 的报告用例会把图与 CSV 写进
`data/outputs/`（未跟踪目录）。跑完 `git status` 仍然干净是正常现象，不要为此加断言。

## 改这里之后要跑

```bash
python -m pytest
python -m pyflakes src/ scripts/ tests/
```

`pyproject.toml` 的 `addopts = "-q"` 会让 `python -m pytest -q` 变成 `-qq`，
末行统计会被吞掉；要看到 `180 passed` 就只写 `python -m pytest`。
