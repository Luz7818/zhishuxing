# data/ —— 输入语料与产物

> 用途：说清哪些文件是入库的源数据、哪些是跑一次就变的产物，以及改动的正确顺序。
> 本目录入库的只有 `real/` 的模板、`transfer_kb/` 与 `samples/`；三个运行时目录首次运行才建出来。

## 文件清单

本目录根下只有 `README.md` 一份文件，其余入库文件全在 `real/`、`transfer_kb/` 与 `samples/`
三个子目录里，逐份说明见下面两张表，这里只登记口径与命名规律。

| 文件 | 干什么 | 备注 |
|---|---|---|
| `README.md` | 本说明 | 唯一的根级文件（复核：`find data -maxdepth 1 -type f`） |
| `real/` | 真实客流数据约定目录(2.2.0):heatmap/transfer/efficiency 自动发现消费 | 格式与命名见 [real/README.md](real/README.md);真实 CSV 不入库,模板入库 |
| `transfer_kb/` 的 23 个文件 | 22 篇语料 + 1 份入库语料 `corpus.jsonl` | 语料命名 `NN_english_slug.md`（2026-10-07 起由中文名英文化），一文件一篇；23 复核 `git ls-files data/transfer_kb \| wc -l` |
| `samples/` 的 18 个文件 | 报告与演示的参考产物 | 带 `_simulated` 后缀的 6 个出自固定种子合成数据；18 复核 `git ls-files data/samples \| wc -l` |

`data/` 下入库文件合计 46 个（复核：`git ls-files data` 计数）。`outputs/`、`model/`、`runs/`
里的东西一个都没入库，所以不在计数内（复核：`git check-ignore -v data/outputs data/model data/runs`）。

## 子目录

| 子目录 | 是否入库 | 负责 |
|---|---|---|
| `transfer_kb/` | ✓ | 换乘经验知识库：源文档 + 入库语料 |
| `samples/` | ✓ | 参考产物，README 展示图与奖励 npy 从这里取；只由脚本重建 |
| `real/` | 部分 ✓（模板入库） | 真实客流数据约定目录：`README.md` 与 3 个 `*.example.csv` 模板入库，真实 CSV 不入库 |
| `outputs/` | ✗（在 `.gitignore` 里） | 运行时输出：报告图、CSV、GIF、JSON、Web 面板图 |
| `model/` | ✗ | 训练权重 `*_actor_*_agent_*.pth` |
| `runs/` | ✗ | TensorBoard 日志 |

`outputs/`、`model/`、`runs/` 由 `config.Paths.ensure_runtime_dirs()` 在起服务或跑仿真时创建，
不要在仓库里手工放文件：奖励曲线报告先找 `outputs/` 再回退 `samples/`。

## transfer_kb/ —— 知识库

| 路径 | 内容 |
|---|---|
| `shenzhen_north/*.md` | 22 篇演示语料（复核：`ls data/transfer_kb/shenzhen_north` 计数），一文件一篇，`NN_english_slug.md` 命名（2026-10-07 起由中文名英文化，文件名只做标识，检索靠语料的 title/tags） |
| `corpus.jsonl` | 入库后的 BM25 检索语料，22 行，字段 `id`、`hub`、`title`、`source`、`tags`、`content`（复核：`wc -l data/transfer_kb/corpus.jsonl`） |

`corpus.jsonl` 是 `zhishuxing kb-ingest` 的产物，不要手改：它的 `id` 规则是
`hub:相对路径:段序号`，手改的行会在下一次入库时被整体覆盖。入库幂等可重跑
（复核：`zhishuxing kb-ingest`，输出 `"changed": 0` 表示语料与源文档已一致）。

内容性质要写清楚：这 22 篇是**按公开出行攻略与站方指引手工整理的演示语料**，不是站内实测数据，
站内布局以现场为准。换枢纽时新建 `transfer_kb/<hub>/` 并用 `--hub` 指过去即可，不需要改代码。

## samples/ —— 参考产物

18 个文件（复核：`ls data/samples` 计数），按来源分三类：

| 来源 | 文件 | 重建方式 |
|---|---|---|
| 7 类分析报告 | `reward_curve.png`、`shenzhen_north_congestion_heatmap.png`、`zone_improvement_ranking.png`、`peak_shaving_by_timeslot.png`、`transfer_time_distribution_simulated.csv`、`security_queue_comparison.png` 与 `.csv`、`transfer_efficiency_scenario_comparison.png` 与 `.csv`、`finetune_metrics_simulated.png` 与 `.csv`、`transfer_env_demo.gif` | `zhishuxing analyze --report all` 后从 `data/outputs/` 拷回 |
| 演示与汇总 | `zhishuxing_dashboard.png`、`zhishuxing_summary.json`、`mock_llm_finetune_metadata.json` | `zhishuxing demo` |
| 训练产物样本 | `MADDPG_env_integrated_hub_transfer_number_1_seed_0_simulated.npy` 及 seed 1、2 两份 | 由 `zhishuxing train` 产出，留在这里供离线渲染奖励曲线 |

`samples/` 里**没有** `transfer_time_distribution.png`：奖励曲线报告只依赖 `*_env_*.npy`
这一种命名（复核：`ls data/samples/*_env_*.npy`）。

## 和谁打交道

- **上游**：`zhishuxing kb-ingest` 由 `data/transfer_kb/<hub>/` 的源文档写 `corpus.jsonl`；
  `zhishuxing analyze` / `demo` / `simulate` / `animate` 与 `analysis/reports.py` 写 `outputs/`；
  `zhishuxing train` 写 `model/` 与 `runs/`。本目录不吃外部输入。
- **下游**：`outputs/` 经 Flask 的 `GET /outputs/<path:filename>` 同源暴露给前端
  （复核：`grep -n 'outputs/<path' src/zhishuxing/webapp/app.py`）；`samples/` 是奖励曲线报告
  的回退目录、`tests/test_cli.py` 的产物名断言来源、`docs/GET-START.md` 的离线样例；
  `transfer_kb/corpus.jsonl` 由 `llm/kb.py` 建索引。
- **改这里之后要跑**：

```bash
python -m pytest -q
zhishuxing smoke
```

改了 `transfer_kb/shenzhen_north/` 的源文档，先重跑入库再跑测试：

```bash
zhishuxing kb-ingest --query "带老人 优先直梯"
```

`tests/test_kb.py` 的检索相关性用例读的是 `corpus.jsonl`，语料与索引不同步时它会失败。

## 别动

- `samples/` 下的 `png`、`csv` 与 `gif`：`docs/GET-START.md` 拿它们当离线样例，`reward`
  报告没有训练产物时回退到这里取 `*_env_*.npy`。删了测试不会变红，但新克隆就出不了图；
  要更新就重跑报告再拷回。复核：`sed -n '52,56p' src/zhishuxing/analysis/reports.py`。
- `.gitignore` 中 `!data/samples/*.gif` 这一行是全局 `*.gif` 忽略的例外。删掉它，
  `transfer_env_demo.gif` 就不会再被跟踪，README 里的动图在 clone 之后是空的。
- `transfer_kb/corpus.jsonl`：它是 `kb-ingest` 的产物却不是可弃的产物 —— `llm/kb.py` 只读 JSONL，
  删掉它检索链路立刻空转，而重跑入库需要源文档仍在（复核：`zhishuxing kb-ingest --query "直梯"`
  输出的 `"changed": 0`）。
- `outputs/`、`model/`、`runs/` 三个运行时目录：里面没有任何入库文件（复核：
  `git ls-files data/outputs data/model data/runs` 无输出），删掉内容后
  `config.Paths.ensure_runtime_dirs()` 会在起服务时把目录重建。真正要当心的是往 `model/`
  手工放权重 —— `rl/runtime.py` 按文件名里的 step 取最大那份来推理，假权重会被直接用掉。
