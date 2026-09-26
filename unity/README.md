# Unity 接入模板（综合交通枢纽快速换乘）

> 用途：说明 `unity/` 这一侧负责什么、Python 侧要配合到哪一步，以及在 Unity Editor 里怎么挂。
> 本目录不是可构建的 Unity 工程，只有智能体脚本与这份接入说明。

## 本目录负责什么

Python 侧的训练循环、算法与运行时都在 `src/zhishuxing/rl/`；Unity 侧要提供的只是
一个"能被 `mlagents_envs` 驱动的场景 + 智能体脚本"。本目录放的就是后者的模板：
它定义观测向量怎么拼、动作两个维度怎么用、奖励由哪几项组成，Python 侧按同一套契约读写。

| 文件 | 干什么 | 与 Python 侧的对应关系 |
|---|---|---|
| `HubTransferAgent.cs` | 挂在智能体 GameObject 上的 `Agent` 子类（133 行）：`CollectObservations()` 拼「目标相对位置 (dx, dz) + 自身速度 (vx, vz) + 邻近行人相对位置」；`OnActionReceived()` 把 `a[0]` 当前进/后退、`a[1]` 当转向；`Heuristic()` 供键盘演示 | 观测与动作语义对齐 `src/zhishuxing/rl/envs.py`；内置网格仿真的同一套契约见 `src/zhishuxing/core/simulation.py`（那里 `OBS_DIM = 4 + 2×3`） |
| `README.md` | 本文档：挂载步骤、参数对齐、奖励建议、训练启动与产物位置 | — |

`[Header("Reward Weights")]` 下的 5 个奖励权重（`stepPenalty` -0.001、`progressReward` 0.01、
`congestionPenalty` -0.02、`collisionPenalty` -0.2、`successReward` 2.0）都是 Inspector 可调字段，
改它们不需要改代码。

**本目录当前不可构建**：仓库里没有 `Assets/`、`ProjectSettings/` 等 Unity 工程文件，
也没有 `.meta`，所以它不参与 CI，也没有任何 Python 测试引用它。要在真机上训练，
需要把这些脚本拷进你自己的 ML-Agents 工程并按下面的参数配好。

该模板用于对接当前 Python 侧的 RL 运行时（`src/zhishuxing/rl/runner.py`，训练入口为 `zhishuxing train`）。

## 1. 场景挂载

1. 在每个智能体对象上挂载 `HubTransferAgent`。
2. 同对象挂载：
   - `Behavior Parameters`
   - `Decision Requester`（建议 `Decision Period = 5`）
3. `Behavior Parameters` 关键设置：
   - `Behavior Name`: `HubAgent`
   - `Space Size`(Vector Observation): 与 `CollectObservations()` 输出维度一致
   - `Actions`: `Continuous Actions`, `Branch Size = 2`
4. 给场景中障碍物打 `Obstacle` Tag，其他智能体可打 `Agent` Tag。

## 2. 观测/动作对齐

- 当前脚本动作维度为 2：
  - `a[0]`: 前进/后退
  - `a[1]`: 左右转向
- 观测由以下部分拼接：
  - 目标相对位置 `(dx, dz)`
  - 自身速度 `(vx, vz)`
  - 邻近人流相对位置（每个 `(dx, dz)`）

## 3. 奖励建议

模板里已实现：
- 每步轻微惩罚（鼓励更快完成换乘）
- 接近目标奖励
- 拥堵惩罚
- 碰撞惩罚
- 到达目标奖励

你可以按业务目标再加：
- 列车发车窗口惩罚
- 平台超载惩罚
- 团体换乘成功率奖励

## 4. Python 侧启动

先装训练依赖（`mlagents_envs` 在 PyPI 上没有 1.x 版本，必须从源码装，本机镜像见 `third_party/`）：

```bash
pip install -e .[train]
pip install ./third_party/ml-agents-develop/ml-agents-develop/ml-agents-envs
```

在**仓库根目录**执行（PowerShell 与 bash 同一条命令）：

```bash
zhishuxing train --mlagents_file "D:\Builds\HubTransfer\HubTransfer.exe" --behavior_name HubAgent --episode_limit 200 --max_train_steps 500000 --evaluate_freq 5000
```

连接 Unity Editor（Play 模式）时去掉 `--mlagents_file`：

```bash
zhishuxing train --behavior_name HubAgent --episode_limit 200 --max_train_steps 200000
```

> 注意：训练脚本会同时创建训练环境与评估环境两个实例，Editor 模式可能连接不稳定，
> 建议优先用 Build 出来的 `.exe`。
> 没装 `mlagents_envs` 时 `zhishuxing train` 直接失败：
> `ModuleNotFoundError: No module named 'mlagents_envs'`，退出码 1。

## 5. 结果查看

产物路径由 `src/zhishuxing/config.py` 统一推导（复核：`python -c "from zhishuxing import config; print(config.paths)"`）：

| 内容 | 位置 | 命名 |
|---|---|---|
| TensorBoard 日志 | `data/runs/` | `{algorithm}_env_{env_name}_number_{n}_seed_{s}` |
| 评估奖励数组 | `data/outputs/` | `{algorithm}_env_{env_name}_number_{n}_seed_{s}.npy` |
| actor 权重 | `data/model/<env_name>/` | `{algorithm}_actor_number_{n}_step_{k}k_agent_{id}.pth` |

`env_name` 取自 `configs/training.json`，当前是 `integrated_hub_transfer`。三个目录都是运行时
产物且都在 `.gitignore` 里，不要提交。

奖励曲线可视化（读上面那批 npy，取最新一个）：

```bash
zhishuxing analyze --report reward
```

看训练进度用 TensorBoard：`tensorboard --logdir data/runs`（`[train]` 依赖组里有 `tensorboard`）。
训练完成后在 Web 控制台「RL 智能体」视图点「加载最新策略权重」，即可用真实策略跑引导仿真。
