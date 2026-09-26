# configs/ —— 静态输入配置

> 用途：说明四个配置文件各自被哪个模块读取、字段改了会波及哪条链路。
> 这里的东西是"数据"而不是代码：改数值不需要改 Python，但改结构会直接抛异常。

| 文件 | 谁读 | 管什么 |
|---|---|---|
| `hub_default.json` | `core/navigation.py` 的 `load_navigation()`；路径由 `config.paths.navigation_config` 给出 | 枢纽网格与地标表，规划与仿真的唯一底图 |
| `scenarios.json` | `core/scenarios.py` 的 `load_scenarios()`；`cli.py` demo/simulate/smoke 与 `GET /api/scenarios` | 3 组演示乘客群体的起终点、必经地标、释放时间与人数 |
| `training.json` | `config.load_training_config()` → `rl/runner.py` 的 `build_train_args()` | MADDPG/MATD3 超参默认值，CLI 参数可逐项覆盖 |
| `sample_instruction_data.jsonl` | `MockLLMAdapter.fine_tune()` / `SiliconFlowLLMAdapter.fine_tune()` 的 `dataset_path` | 3 行指令样例，只用于走通"微调"这一步的接口 |

## hub_default.json 字段

| 字段 | 含义 | 改了会怎样 |
|---|---|---|
| `width` / `height` | 网格尺寸，当前 `30 × 16` | 所有坐标必须仍在界内，否则 `plan_landmark_path()` 抛 `ValueError: 非法起点/终点` |
| `cell_size_m` | 一格折算多少米，当前 `30` | 直接乘进"约 N 米"的显示与 `/api/plan` 的 `meters` |
| `blocked` | 禁行格数组 | A* 绕行路径变化，引导仿真图的底色随之变 |
| `cell_tags` | 设施语义层：`stairs`、`escalator`、`elevator`、`crowd` 四类，每类一组格子 | 需求档案的罚项与禁行都按这里的标签名匹配；写错标签名不会报错，只是偏好不生效 |
| `landmarks` | 地标名到坐标，当前 13 个 | 名字要能被 `LANDMARK_LABELS` 翻成中文，否则界面显示英文原名 |

地标名与中文标签的对应、以及「A口 / 备用安检 / 直梯」这类中文说法到地标名的别名表，
都写在 `src/zhishuxing/core/navigation.py` 顶部的常量里，不在本文件。加新地标时要两处一起改。

## scenarios.json 的两点约定

- 起终点可写 `start_landmark` / `goal_landmark`（地标名）或 `start` / `goal`（`[x, y]` 坐标），
  由 `resolve_groups()` 统一解析；地标名不存在时抛 `KeyError: 未找到地标: <name>`。
- `via_landmarks` 是硬必经点列表，路径按"起点 → 途经 → 终点"分段拼接。

## training.json 的两点约定

- `matd3` 这一段只在 `--algorithm MATD3` 时生效；`build_train_args()` 会把它合并进同一命名空间。
- CLI 覆盖项只覆盖非 `None` 的值（`overrides` 里为 `None` 的参数保留文件默认），
  所以不带参数时这里就是实际生效超参。
- `env_name` 是 `integrated_hub_transfer`，它出现在产物文件名模板
  `{algorithm}_env_{env_name}_number_{number}_seed_{seed}.npy` 与 TensorBoard 目录名里。
  读取侧用的 glob 是 `*_env_*.npy`，不匹配具体环境名，所以改了它并不会让既有 npy 失效，
  只是新旧 run 之间只能靠文件名区分。`zhishuxing analyze --report reward` 取的是
  `data/outputs/` 里**修改时间最新**的那一个，没有再回退 `data/samples/`
  （复核：`analysis/reports.py` 的 `sorted(..., key=lambda p: p.stat().st_mtime, reverse=True)`）。

## 改这里之后要跑

```bash
python -m pytest -q
zhishuxing smoke
```

`zhishuxing smoke` 会真的用 `hub_default.json` 规划一次 `[1,2] → [28,12]` 并跑一遍面板与仿真，
配置写坏时它比测试更早失败。
