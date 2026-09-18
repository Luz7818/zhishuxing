# 智枢星 Web 部署说明

本模块将现有“智枢星”能力封装为可部署网页，支持：
- 导航图加载接口
- 路径规划接口
- **MADDPG 强化学习接口**：训练产物扫描、actor 策略权重加载、观测→动作推理、多智能体引导仿真、训练奖励曲线
- 大模型加载与微调接口
- 动态客流可视化面板生成
- 运行既有功能脚本（奖励曲线、热力图、动图）

## 1. 目录
- `MADDPG/webapp/app.py`：Flask 后端与 API
- `MADDPG/webapp/templates/index.html`：网页前端
- `MADDPG/webapp/static/*`：静态资源
- `MADDPG/zhishuxing/rl_bridge.py`：MADDPG 运行时桥接（供 Web 端点调用）
- `MADDPG/run_zhishuxing_web.py`：服务启动脚本
- `MADDPG/webapp/smoke_test.py`：冒烟测试

## 2. 启动（开发）
在工作区根目录执行：

```powershell
(E:\Software\Anaconda\shell\condabin\conda-hook.ps1)
conda activate my
python .\MADDPG\run_zhishuxing_web.py --host 0.0.0.0 --port 7860
```

浏览器打开：`http://127.0.0.1:7860`

## 3. 启动（生产部署）
使用 `waitress` 生产托管：

```powershell
(E:\Software\Anaconda\shell\condabin\conda-hook.ps1)
conda activate my
python .\MADDPG\run_zhishuxing_web.py --host 0.0.0.0 --port 7860 --production
```

## 4. API 清单
- `GET /health`
- `POST /api/navigation/load`
- `POST /api/navigation/plan`
- `GET /api/rl/status`
- `POST /api/rl/load_policy`
- `POST /api/rl/act`
- `GET /api/rl/rewards`
- `POST /api/rl/simulate`
- `POST /api/llm/load`
- `POST /api/llm/fine_tune`
- `POST /api/dashboard/run`
- `POST /api/features/run_existing`
- `GET /outputs/<filename>`

### MADDPG 接口说明
- `GET /api/rl/status`：扫描 `model/` 下的 `*_actor_*_agent_*.pth` 权重与 `data_train/*_env_*.npy` 奖励文件，返回策略加载状态（`policy_source` 为 `MADDPG` / `MADDPG(部分)+启发式` / `启发式回退`）。
- `POST /api/rl/load_policy`：`{"checkpoint_dir": "可选自定义权重目录"}`，加载每个 agent 的最新 step 权重。
- `POST /api/rl/act`：`{"observations": [[...], ...]}`，观测语义与 `UnityTemplate/HubTransferAgent.cs` 对齐（目标相对位置、自身速度、邻近行人相对位置），返回每个 agent 的连续动作。
- `GET /api/rl/rewards`：读取 `data_train/*_env_*.npy` 评估奖励序列并渲染奖励曲线图。
- `POST /api/rl/simulate`：`{"groups": [...同 dashboard 群组格式...], "config": {"max_steps": 240, "agents_per_group": 6, "seed": 42}}`，在内置网格枢纽仿真中以已加载的 MADDPG 策略（无权重时自动回退启发式并明确标注）执行多智能体引导，输出换乘步数、拥堵指数、轨迹图。

> 说明：训练产生的 actor 权重保存在 `MADDPG/model/integrated_hub_transfer/`，仓库默认未提交该目录；无权重时 Web 端点仍可用（启发式回退），但会如实标注策略来源。

## 5. 冒烟测试

```powershell
(E:\Software\Anaconda\shell\condabin\conda-hook.ps1)
conda activate my
python .\MADDPG\webapp\smoke_test.py
```

通过后会输出：`Web smoke test passed.`
