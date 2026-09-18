# Unity 接入模板（综合交通枢纽快速换乘）

该模板用于对接当前 Python 侧 `MADDPG/MADDPG_main.py`。

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

## 4. Python 侧启动（Windows PowerShell）

在仓库根目录运行：

```powershell
python .\MADDPG\MADDPG_main.py --mlagents_file "D:\Builds\HubTransfer\HubTransfer.exe" --behavior_name "HubAgent" --base_port 5005 --episode_limit 200 --max_train_steps 500000 --evaluate_freq 5000
```

如果在 Unity Editor Play 模式下连接：

```powershell
python .\MADDPG\MADDPG_main.py --behavior_name "HubAgent" --base_port 5005 --episode_limit 200 --max_train_steps 200000
```

> 注意：当前 Python 训练脚本会创建训练环境+评估环境两个实例，Editor 模式可能连接不稳定，建议优先使用 Build(.exe) 训练。

## 5. 结果查看

- TensorBoard 日志：`runs/`
- 奖励数组：`data_train/`
- 模型权重：`model/integrated_hub_transfer/`

奖励曲线可视化：

```powershell
python .\MADDPG\plot_results.py --window 10
```
