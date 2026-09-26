# data/ —— 输入语料与产物

> 用途：说清哪些文件是入库的源数据、哪些是跑一次就变的产物，以及改动的正确顺序。
> 本目录只有 `transfer_kb/` 与 `samples/` 入库；三个运行时目录首次运行才建出来。

## 子目录

| 子目录 | 是否入库 | 负责 |
|---|---|---|
| `transfer_kb/` | ✓ | 换乘经验知识库：源文档 + 入库语料 |
| `samples/` | ✓ | 参考产物，README 展示图与奖励 npy 从这里取；只由脚本重建 |
| `outputs/` | ✗（在 `.gitignore` 里） | 运行时输出：报告图、CSV、GIF、JSON、Web 面板图 |
| `model/` | ✗ | 训练权重 `*_actor_*_agent_*.pth` |
| `runs/` | ✗ | TensorBoard 日志 |

`outputs/`、`model/`、`runs/` 由 `config.Paths.ensure_runtime_dirs()` 在起服务或跑仿真时创建，
不要在仓库里手工放文件：奖励曲线报告先找 `outputs/` 再回退 `samples/`。

## transfer_kb/ —— 知识库

| 路径 | 内容 |
|---|---|
| `shenzhen_north/*.md` | 22 篇演示语料（复核：`ls data/transfer_kb/shenzhen_north` 计数），一文件一篇，`NN_主题.md` 命名 |
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

## 改这里之后要跑

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

- `samples/` 下的 `png` 与 `gif`：仓库 README 的展示图直接引用它们，删掉就是破图；
  要更新就重跑报告再拷贝回来。
- `.gitignore` 中 `!data/samples/*.gif` 这一行是全局 `*.gif` 忽略的例外。删掉它，
  `transfer_env_demo.gif` 就不会再被跟踪，README 里的动图在 clone 之后是空的。
