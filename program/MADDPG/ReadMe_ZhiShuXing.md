# 智枢星：动态客流下综合性交通枢纽智慧换乘引导系统

## 1. 目标
在现有 `MADDPG` 工程上增加可视化与系统编排能力，形成可演示的“智枢星”原型：
- 动态客流仿真与热力可视化
- 智慧换乘路径引导（可加载导航图）
- 可插拔大模型接口（加载 / 微调 / 推理）
- 可兼容调用现有可视化脚本（奖励曲线、拥堵热力图、P50/P90/Max、动图）

## 2. 文件结构
- `MADDPG/zhishuxing/adapters.py`：大模型适配接口与默认实现 `MockLLMAdapter`
- `MADDPG/zhishuxing/navigation.py`：导航图加载 + A* 路径规划
- `MADDPG/zhishuxing/visualization.py`：动态客流热力图 + 引导路径可视化
- `MADDPG/zhishuxing/rl_bridge.py`：MADDPG 运行时桥接（策略权重加载 / 观测→动作推理 / 内置多智能体引导仿真 / 训练奖励曲线）
- `MADDPG/zhishuxing/system.py`：系统编排（仿真、策略建议、既有能力桥接）
- `MADDPG/zhishuxing/sample_navigation.json`：示例导航图
- `MADDPG/zhishuxing/sample_instruction_data.jsonl`：示例微调数据
- `MADDPG/run_zhishuxing_demo.py`：一键演示入口

## 3. 快速运行
在工作区根目录执行：

```powershell
(E:\Software\Anaconda\shell\condabin\conda-hook.ps1)
conda activate my
python .\MADDPG\run_zhishuxing_demo.py --output_dir .\data_train
```

生成结果：
- `data_train/zhishuxing_dashboard.png`
- `data_train/mock_llm_finetune_metadata.json`
- `data_train/zhishuxing_summary.json`

## 4. 关键接口
### 大模型接口（可替换成真实微调流程）
`LLMAdapter` 协议：
- `load_model(model_id, model_path=None)`
- `fine_tune(dataset_path, output_dir, config=None)`
- `infer(prompt, context=None)`

你可以把 `MockLLMAdapter` 替换为基于 `transformers + peft` 的真实实现，并保持同样接口。

### 导航接口
`NavigationAdapter`：
- `load_navigation(path)`：加载 JSON 路网
- `plan_path(start, goal)`：A* 规划
- `plan_landmark_path(start, via, goal)`：支持“必经安检”等业务路径

### MADDPG 强化学习桥接接口
`MADDPGRuntime`（`zhishuxing/rl_bridge.py`，Web 前端经 `webapp` 的 `/api/rl/*` 端点调用）：
- `get_status()` / `refresh()`：扫描 `model/integrated_hub_transfer/` 下的 actor 权重与 `data_train/*_env_*.npy` 奖励数据
- `load_policy(checkpoint_dir=None)`：按 agent 加载各自最新 step 的 actor 权重（从权重张量形状自动推断网络维度）；无权重或无 torch 时自动回退启发式策略并如实标注 `policy_source`
- `act(observations)`：批量推理，观测语义与 `UnityTemplate/HubTransferAgent.cs` 对齐（目标相对位置 + 自身速度 + 邻近行人相对位置）
- `reward_series()` / `render_reward_curve()`：读取训练评估奖励序列并出图
- `run_guided_simulation(navigation, groups, ...)`：内置网格枢纽仿真，多智能体在已加载策略（或启发式）引导下沿 A* 路径通行，输出换乘步数、拥堵指数与轨迹图

## 5. 与现有功能联动
如需同时触发已有可视化脚本：

```powershell
python .\MADDPG\run_zhishuxing_demo.py --output_dir .\data_train --run_existing
```

该参数会尝试调用：
- `plot_results.py`
- `plot_congestion_heatmap.py`
- `plot_transfer_time_distribution.py`
- `animate_transfer_env.py`

> 注：这些脚本是否成功输出，取决于你当前环境和脚本参数设置。
