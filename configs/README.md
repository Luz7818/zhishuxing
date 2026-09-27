# configs/ —— 静态输入配置

> 用途：说明四个配置文件各自被哪个模块读取、字段改了会波及哪条链路。
> 这里的东西是"数据"而不是代码：改数值不需要改 Python，但改结构会直接抛异常。

## 文件清单

| 文件 | 谁读 | 管什么 |
|---|---|---|
| `hub_default.json` | `core/navigation.py` 的 `load_navigation()`；路径由 `config.paths.navigation_config` 给出 | 枢纽网格与地标表，规划与仿真的唯一底图 |
| `scenarios.json` | `core/scenarios.py` 的 `load_scenarios()`；`cli.py` demo/simulate/smoke 与 `GET /api/scenarios` | 3 组演示乘客群体的起终点、必经地标、释放时间与人数 |
| `training.json` | `config.load_training_config()` → `rl/runner.py` 的 `build_train_args()` | MADDPG/MATD3 超参默认值，CLI 参数可逐项覆盖 |
| `sample_instruction_data.jsonl` | `MockLLMAdapter.fine_tune()` / `SiliconFlowLLMAdapter.fine_tune()` 的 `dataset_path` | 3 行指令样例，只用于走通"微调"这一步的接口 |

路径都由 `config.Paths` 拼出（复核：`grep -n '"configs"' src/zhishuxing/config.py`），
本目录 5 个文件全部入库（复核：`git ls-files configs`）。

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
  只是新旧 run 之间只能靠文件名区分。`zhishuxing analyze --report reward` 先在
  `data/outputs/` 取**修改时间最新**的那个 npy，取不到才回退 `data/samples/` 的三份样本
  （复核：`sed -n '52,68p' src/zhishuxing/analysis/reports.py` 里的
  `search_dirs = [cfg.paths.outputs, cfg.paths.samples]` 与按 `st_mtime` 倒序的 `sorted`）。

## 和谁打交道

- **上游**：全部由人手工编辑，没有任何脚本会重写本目录的文件
  （复核：`grep -rn "configs" src/zhishuxing scripts --include="*.py"`，输出里全是读路径或帮助文本，
  没有一处写）。
- **下游**：`core/navigation.py` 读 `hub_default.json`；`core/scenarios.py` 与 `webapp/service.py`
  的 `GET /api/scenarios` 读 `scenarios.json`；`rl/runner.py` 的 `build_train_args()` 读
  `training.json`；`cli.py` 的 demo 把 `sample_instruction_data.jsonl` 的路径交给微调接口。
- **改这里之后要跑**：

```bash
python -m pytest -q
zhishuxing smoke
```

`zhishuxing smoke` 会真的用 `hub_default.json` 规划一次 `[1,2] → [28,12]` 并跑一遍面板与仿真，
配置写坏时它比测试更早失败。

## 别动

- **目录名 `configs/` 本身**：它是 `_find_workspace_root()` 认 workspace 根的唯一判据，
  删空或改名不报错，但产物路径会退回 `Path.cwd()`、跟着启动目录漂移。
  复核：`sed -n '75,84p' src/zhishuxing/config.py`。
- **`landmarks` 里的 `security` 与 `security_backup` 两个键名**（13 个键中的 2 个）：
  `scenarios.json` 的 `via_landmarks` 与 `zhishuxing smoke` 硬编码引用它们，改名即
  `KeyError: 未找到地标: security`。复核：`sed -n '347,357p' src/zhishuxing/cli.py`。
- **`cell_tags` 的四个标签名 `stairs` / `escalator` / `elevator` / `crowd`**：偏好罚项按标签名匹配，
  写错不抛异常，只是偏好静默不生效。
- **`sample_instruction_data.jsonl`**：两个 `fine_tune()` 只把 `dataset_path` 这个字符串抄进产物
  元数据，并不读文件内容，所以它看着"没人用"却删不得 —— 删了 `mock_llm_finetune_metadata.json`
  里的路径就指向不存在的位置。复核：`sed -n '67,80p' src/zhishuxing/llm/adapters.py`。
- **`training.json` 的 `env_name`**：写进 TensorBoard 目录名与 `*.npy` 文件名模板，改了不会崩，
  但历史 run 只能靠文件名区分，等于切断可追溯性。
