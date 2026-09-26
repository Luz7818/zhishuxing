# code_optimization/ —— 行人仿真性能基准

> 用途：说明 `core/animation.py` 那 19x 加速是从哪来、怎么复测、以及复测会改写哪个入库文件。

这个目录是一次**已完成**的优化工作的存档：把 `HubTransferAnimator.step()` 从逐 agent 的
Python 循环改成 NumPy 向量化，并留下一份"冻结的旧实现"作为对照基线。
优化结论已经移植进 `src/zhishuxing/core/animation.py`，本目录不参与运行时。

## 文件清单

| 文件 | 干什么 | 备注 |
|---|---|---|
| `report.md` | 优化过程报告：基线热点、v1 与 v2 各自改了什么、统计等价性验收、未实施的后续建议 | 独立主题文档，本目录以它为主 |
| `baseline_animation.py` | 与历史 `animate_transfer_env.py` 逐行一致的原始实现（244 行），只保留 `step()` / `congestion_index()` 及依赖 | 基线，不要"顺手优化"它，它存在的意义就是慢 |
| `benchmark_animation.py` | 跑 baseline / v1 / v2 / 移植后 core 四种实现 × 两种规模的计时与等价性校验（286 行） | **执行它会重写 `benchmark_results.json`** |
| `benchmark_results.json` | 上一次基准的机器可读结果 | 入库的生成物，随 `report.md` 一起看 |

## 结果

| 规模 | 基线耗时 | 移植后耗时 | 加速 |
|---|---|---|---|
| 44 人 × 1000 帧（默认） | 0.682 s | 0.035 s | 19.25x |
| 500 人 × 300 帧 | 7.162 s | 0.825 s | 8.69x |

复核：`code_optimization/benchmark_results.json` 里两个 case 的 `speedup_core` 字段。

加速比在大规模下反而下降，原因写在 `report.md`：向量化把 O(n²) 的成对距离矩阵本身变成了地板。
再往上的规模要换空间哈希网格邻居检索（`report.md` 末尾列为"未实施"）。

## 验收口径：统计等价，不是逐位一致

社会力从"就地更新的位置"改成"帧首快照位置"计算，这是群体仿真的常规做法，
代价是轨迹不再与旧实现逐位相同。等价性按三项验收（数据同上，`equivalence` 字段）：

| 项 | 默认规模 | 大规模 |
|---|---|---|
| 到达人数 | 与基线一致 | 与基线一致 |
| 引导介入次数偏差 | 4 | 24 |
| 拥堵指数偏差 | 0.0455 | 0.0 |

所以**不要**给动画加"逐帧像素比对"式的测试，那会把一个有意取舍当成 bug 来修。
需要逐位复现旧行为时用 `baseline_animation.py`。

## 复测

```bash
python code_optimization/benchmark_animation.py
```

在仓库根执行。它把控制台摘要与 `benchmark_results.json` 一起更新。
因为 `benchmark_results.json` 是入库文件，跑之前先 `git status` 确认干净，
跑完 `git diff code_optimization/benchmark_results.json` 判断改动是否只是计时抖动。

数值会随机器波动（`report.md` 里 v2 独立实现是 19.53x、移植版 19.25x，同一台机器上的两次计时），
所以别把加速比当回归门禁 —— 它没有阈值断言，只是基准记录。

## 和谁打交道

- **上游**：`src/zhishuxing/core/animation.py` 是被测对象。
- **下游**：`analysis/reports.py` 的 `run_animation_report()` 与 `zhishuxing animate` 用移植后的实现。
- 改了 `core/animation.py` 的力场、放行窗口或卡滞阈值（>18 帧）之后，这里的结果就过期了，
  要么重测要么在 `report.md` 里注明基线对应的实现版本。
