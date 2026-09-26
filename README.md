# 智枢星 ZhiShuXing

<p align="center">
  <img src="src/zhishuxing/webapp/static/assets/logo-full.svg" alt="智枢星" width="420" />
</p>

<p align="center">
  <a href="https://github.com/Luz7818/zhishuxing/actions/workflows/ci.yml"><img src="https://github.com/Luz7818/zhishuxing/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/tests-180%20passed-brightgreen" alt="tests" />
  <img src="https://img.shields.io/badge/license-Proprietary-red" alt="license" />
</p>

> 动态客流下的综合性交通枢纽智慧换乘引导系统
> **MADDPG 多智能体强化学习 + 枢纽仿真 + LLM 对话式引导 + Web 控制台/移动端 PWA**

「智枢星」面向大型综合交通枢纽（以深圳北站为演示场景），解决一个具体问题：**在高峰动态客流下，乘客如何带着各自的个性化需求快速、舒适地完成站内换乘**。

核心主张：做一个真正懂乘客的换乘引导助手。乘客用自然语言表达需求（赶时间、优先直梯、途经卫生间、带老人行李、轮椅无障碍……），系统把口语解析成结构化需求档案 → 偏好感知规划 → 引用站内换乘经验 → 解释「为什么这样走符合你的需求」。

---

## 目录

- [快速开始](#快速开始)
- [项目整体功能](#项目整体功能)
- [系统架构](#系统架构)
- [各板块功能详解](#各板块功能详解)
- [仓库位置安排](#仓库位置安排)
- [对话式助手工作原理](#对话式助手工作原理)
- [Web API 一览](#web-api-一览)
- [统一 CLI](#统一-cli)
- [MADDPG 训练（连接 Unity）](#maddpg-训练连接-unity)
- [测试与质量保障](#测试与质量保障)
- [环境变量](#环境变量)
- [成果预览](#成果预览)
- [注意事项](#注意事项)

---

## 快速开始

```bash
# 1) 安装（Python ≥ 3.10）
pip install -e .            # 基础依赖；训练链路: pip install -e .[train]

# 2) 可选：配置密钥（不配置也能完整离线演示）
cp .env.example .env        # 填入高德 / SiliconFlow Key，见「环境变量」

# 3) 换乘经验知识库入库（首次运行）
zhishuxing kb-ingest        # data/transfer_kb/shenzhen_north/ → corpus.jsonl（--query 自检检索）

# 4) 启动 Web 控制台 + 移动端 PWA（同一后端）
zhishuxing serve --port 7860
#    控制台   http://127.0.0.1:7860
#    移动端   http://127.0.0.1:7860/mobile

# 5) 一键演示 / 分析报告 / 引导仿真
zhishuxing demo                   # → data/outputs/zhishuxing_dashboard.png + summary
zhishuxing analyze --report all   # 全部 7 类分析报告
zhishuxing simulate               # MADDPG（无权重时启发式回退）引导仿真

# 6) 运行测试
pip install -e .[dev] && pytest   # 180 项
```

> 无任何密钥时系统**全链路可用**：LLM 走 Mock 确定性降级、地图回退 Canvas 折线示意、规划走内置枢纽引擎——离线即可完整演示。

## 项目整体功能

| 功能 | 说明 | 入口 |
|---|---|---|
| **对话式换乘助手（核心）** | `POST /api/chat` 多轮对话闭环：需求档案解析 → 偏好感知规划 → 站内换乘经验检索（BM25）→ 个性化回答合成 | 控制台「智能换乘助手」/ PWA 对话页 |
| **需求档案 PassengerProfile** | 时间/距离/舒适/拥挤四维优先级 + 硬约束（途经卫生间、不走楼梯）+ 软偏好（优先直梯、少步行、避开拥挤）+ 乘客画像（行李/老人/儿童/轮椅/赶时间，自动派生隐含偏好），多轮会话增量合并 | `llm/profile.py` |
| **偏好感知规划** | 导航图 schema v2（直梯/扶梯/楼梯/拥挤区 cell_tags + 设施地标），加权 A*：楼梯/扶梯按需求加价或禁行，「途经卫生间」软必经点（顺路即纳入、绕行过多则如实说明），硬约束不可行时自动降级并注明 | `core/navigation.py` |
| **换乘经验知识库** | 深圳北站演示语料（22 篇，标注来源），`kb-ingest` 持续收录任意 txt/md/html，纯 Python BM25 检索零第三方依赖 | `llm/kb.py` + `data/transfer_kb/` |
| **多智能体强化学习** | MADDPG / MATD3 双算法（Unity ML-Agents 环境），训练产物可被 Web/CLI 直接加载推理 | `rl/` + `unity/` |
| **智慧换乘引导仿真** | A* 路径规划（支持「必经安检」）、动态客流热力、MADDPG（或启发式回退）驱动的多智能体仿真与动图 | `core/` + `rl/runtime.py` |
| **真实路线规划** | LLM 提取出行诉求 → 高德地理编码/公交换乘（偏好参数真实生效）；内置枢纽引擎可完全离线 | `planning/amap.py` |
| **地图展示双通道** | 优先渲染高德 JS API 真实底图（折线沿真实道路、深色风格），未配置 Key 或加载失败自动回退 Canvas 折线并注明原因 | `webapp/static/app.js` |
| **分析报告** | 奖励曲线、拥堵热力图、P50/P90/Max 换乘时间、安检排队对比、场景效率对比、微调指标、行人动图——固定种子可复现 | `analysis/reports.py` |
| **Web 控制台 + 移动端 PWA** | 同一 Flask 后端；控制台多视图（概览/枢纽导航/RL 智能体/客流面板/路线规划/LLM/报告/助手），PWA 可安装、离线缓存、后端不可用自动回退演示数据 | `webapp/` + `web/mobile/` |

## 系统架构

```text
┌─────────────────────────────── 展示层 ───────────────────────────────┐
│  Web 控制台 (webapp/static, 多视图 SPA)      移动端 PWA (web/mobile)   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ fetch / JSON
┌──────────────────────────────▼──────────────────────────────────────┐
│                     Flask 应用 (webapp/app.py)                       │
│         路由 / 校验 / 运行时配置 … 服务层编排 (webapp/service.py)    │
└───┬──────────────┬───────────────┬──────────────┬───────────────────┘
    │              │               │              │
┌───▼───┐   ┌──────▼─────┐   ┌─────▼─────┐  ┌─────▼──────────┐
│对话助手 │   │ 系统编排    │   │ RL 运行时  │  │ 真实路线规划     │
│llm/    │   │ core/      │   │ rl/runtime│  │ planning/amap   │
└───┬───┘   └──────┬─────┘   └─────┬─────┘  └─────┬──────────┘
    │              │               │              │
    │   ┌──────────▼──────────┐    │       高德 REST / JS API
    │   │ 导航加权A*+设施层     │    │       (未配 Key 自动降级)
    │   │ 客流生成·引导仿真    │    │
    │   └──────────┬──────────┘    │
    │              │      MADDPG/MATD3 策略加载/推理
    │              │      (无权重时启发式回退, core↔rl 经
    │              │       PolicyProtocol 解耦)
┌───▼──────────────▼───────────────▼──────────────────────────────────┐
│  数据与资产: configs/(导航图v2·场景·超参)  data/transfer_kb/(BM25语料) │
│  Unity 训练产物(npy/pth)  data/outputs/(运行时)  data/samples/(参考)  │
└─────────────────────────────────────────────────────────────────────┘
```

设计要点（详见 [docs/architecture.md](docs/architecture.md)）：

- **core 与 rl 解耦**：仿真通过 `PolicyProtocol` 注入策略，MADDPGRuntime / HeuristicPolicy 可互换，无 torch 也能跑全链路
- **可选依赖隔离**：`torch` / `mlagents_envs` / `openai` 均为可选，Web/演示链路无硬依赖（`rl` 包惰性导出）
- **路径零 CWD 依赖**：所有产物锚定 workspace（`config.Paths`），密钥全部环境变量，零硬编码
- **离线可演示**：LLM Mock、Canvas 底图、启发式策略三重降级，断网也能完整走通演示闭环

## 各板块功能详解

### `src/zhishuxing/llm/` — 对话式智能引导
| 文件 | 职责 |
|---|---|
| `assistant.py` | `TransferAssistant` 编排器：解析→合并档案→分流枢纽/高德引擎→检索经验→LLM 合成回答；Mock 模式确定性模板，离线闭环 |
| `profile.py` | `PassengerProfile` 需求档案：规则关键词 + LLM JSON 双通道解析，画像派生隐含偏好，多轮增量合并 |
| `kb.py` | `TransferKB` 站内换乘经验库：中文 2-gram + ASCII 分词的 BM25 内存索引，`ingest_directory` 从 txt/md/html 清洗入库（幂等可重跑） |
| `adapters.py` | LLM 适配器：`MockLLMAdapter`（确定性）/ `SiliconFlowLLMAdapter`（OpenAI 兼容，JSON 模式解析 + chat 合成） |

### `src/zhishuxing/core/` — 枢纽领域内核
| 文件 | 职责 |
|---|---|
| `navigation.py` | 导航图（schema v2，含设施语义层与地标）：加权 A*、软必经点取舍、硬约束降级、偏好代价规格 `RouteCostSpec` |
| `flow.py` / `scenarios.py` | 动态客流矩阵生成、乘客分组与场景解析 |
| `simulation.py` | 多智能体引导仿真（行人级），`PolicyProtocol` 注入策略，步进向量化（较基线 19x 加速） |
| `system.py` | `ZhiShuXingSystem` 总编排：导航 + 客流 + 引导文案 + 可视化面板 |
| `animation.py` | 换乘环境动图（GIF）渲染 |

### `src/zhishuxing/rl/` — 多智能体强化学习
| 文件 | 职责 |
|---|---|
| `agents.py` | MADDPG 与 MATD3（TD3 三技巧独立类路径），数值行为与历史实现一致 |
| `networks.py` / `buffer.py` | Actor/Critic 网络、经验回放 |
| `envs.py` | Unity ML-Agents gym 风格封装 |
| `runner.py` | 训练主循环：噪声衰减、周期评估、npy/权重落盘 |
| `runtime.py` | 运行时：扫描训练产物、加载最新 actor 权重、观测→动作推理；无权重回退启发式 |

### `src/zhishuxing/planning/` — 真实路线规划
`amap.py`：LLM/正则 OD 提取 → 高德地理编码 → 公交换乘（策略映射 + 偏好参数）→ 分段详情；新版 v3 响应嵌套结构（`walking.steps`/`bus.buslines`）兼容。

### `src/zhishuxing/analysis/` — 分析与可视化
`reports.py`（7 类可 import 报告函数，CLI 薄封装）、`synthetic.py`（固定种子合成数据单一来源）、`plotting.py`（中文字体/热力/轨迹渲染）、`io_utils.py`（CSV IO）。

### `src/zhishuxing/webapp/` — Web 控制台
`app.py`（Flask 路由）、`service.py`（服务层编排：系统 + RL 运行时 + 高德 + 对话助手）、`static/`（多视图 SPA：Canvas 网格选点、奖励/微调图表、KPI 化结果、对话视图）、`templates/`、品牌资产 `static/assets/`。

### `web/mobile/` — 移动端 PWA
对话式换乘引导 + 需求档案可视化；可安装、ServiceWorker 离线缓存、后端不可用自动回退演示数据；品牌图标由 `scripts/render_brand_assets.py` 生成。

### 其余目录
- `unity/` — Unity 侧智能体模板 `HubTransferAgent.cs`（观测/动作契约与 `rl/envs.py` 对齐）
- `configs/` — 导航图 schema v2（`hub_default.json`）、演示场景、训练超参、微调样例数据
- `data/transfer_kb/` — 知识库源文档（`shenzhen_north/` 22 篇）+ 入库语料 `corpus.jsonl`
- `data/samples/` — 参考产物（README 展示图、奖励 npy）；`data/outputs|model|runs/` 为运行时输出（gitignore）
- `code_optimization/` — 行人仿真向量化基准与报告（19.25x 加速）
- `docs/` — 架构决策与迁移映射；`legacy/ui/` — 历史 Streamlit 原型归档（密钥已移除）

## 仓库位置安排

```text
zhishuxing/
├─ README.md                    # 本文档：项目总览入口
├─ docs/architecture.md         # 架构决策、数据流、历史迁移映射
├─ pyproject.toml               # 包定义 / 依赖 / CLI 入口 / pytest 配置
├─ requirements.txt             # pip 直装依赖（含 mlagents-envs 源码安装说明）
├─ .env.example                 # 密钥模板（.env 本地填写，不入库）
├─ .gitignore
│
├─ configs/                     # 静态配置：导航图 / 场景 / 训练超参
├─ data/
│  ├─ transfer_kb/              # 换乘经验知识库（源文档 + corpus.jsonl）
│  ├─ samples/                  # 参考产物（入库，README 展示用）
│  └─ outputs|model|runs/       # 运行时产物（gitignore，首次运行自动生成）
│
├─ src/zhishuxing/              # 核心 Python 包
│  ├─ config.py                 #   路径推导 + 环境变量密钥（.env 自动加载）
│  ├─ settings.py               #   运行时配置：掩码回读 + 原子写回 .env
│  ├─ cli.py                    #   统一 CLI：demo/train/analyze/simulate/…
│  ├─ core/                     #   领域内核：导航/客流/仿真/编排/动图
│  ├─ rl/                       #   强化学习：算法/环境/训练/运行时
│  ├─ llm/                      #   对话助手：档案/知识库/编排/适配器
│  ├─ planning/                 #   高德真实路线规划
│  ├─ analysis/                 #   分析报告/合成数据/绘图/IO
│  └─ webapp/                   #   Flask 应用 + 前端静态资源 + 品牌资产
│
├─ web/mobile/                  # 移动端 PWA（HTML/清单/SW/图标）
├─ unity/                       # Unity 智能体模板
├─ scripts/                     # 工具脚本（品牌资产生成）
├─ tests/                       # pytest 套件（180 项，覆盖 API/导航/档案/KB/RL/CLI）
├─ code_optimization/           # 性能基准与优化报告
└─ legacy/ui/                   # 历史 Streamlit 原型归档
```

**找东西的速查**：改对话逻辑 → `llm/assistant.py`；改规划代价/设施 → `core/navigation.py` + `configs/hub_default.json`；加 API → `webapp/app.py` + `service.py`；改界面 → `webapp/static/app.js`（控制台）/ `web/mobile/mobile_app.html`（PWA）；换枢纽数据 → `configs/hub_default.json` + `data/transfer_kb/`。

## 对话式助手工作原理

```text
乘客消息 ──► ① 需求解析（规则关键词优先，口语表达走 LLM JSON 提取）
        ──► ② 与会话需求档案增量合并（多轮累积，「我现在有点赶时间」也能懂）
        ──► ③ OD 提取：起终点命中枢纽地标 → 枢纽内加权 A* 偏好规划
                      （优先直梯→楼梯/扶梯重罚、轮椅→楼梯禁行、途经卫生间→软必经点）
                否则 → 高德引擎（strategy 映射 + 偏好提示）
        ──► ④ 换乘经验知识库 BM25 检索 top-3（注明来源标题）
        ──► ⑤ LLM 合成个性化回答（为什么这样走符合需求）；Mock 模式走确定性模板
```

效果示例：输入「带老人行李多，优先直梯，去地铁前先上趟卫生间，从A口出发」→ 需求档案 `途经卫生间、优先直梯、行李较多、携老人` → 路线绕行**无障碍直梯**并**顺路途经卫生间A** → 引用《卫生间分布指南》《老人出行建议》《行李多怎么办》。

## Web API 一览

| 端点 | 说明 |
|---|---|
| `GET /health` | 健康检查 |
| `POST /api/chat` / `POST /api/chat/reset` | 对话式换乘助手（多轮会话、需求档案、偏好规划、经验引用）/ 会话重置 |
| `POST /api/navigation/load` / `POST /api/navigation/plan` | 导航图加载 / A* 规划（支持必经地标） |
| `GET /api/navigation/grid` | 导航网格（含设施语义层 cell_tags 与地标中文标签） |
| `GET /api/scenarios` | 演示场景列表 |
| `GET /api/rl/status` | MADDPG 训练产物扫描与策略状态 |
| `POST /api/rl/load_policy` | 加载每个 agent 最新 step 的 actor 权重 |
| `POST /api/rl/act` | 观测→动作推理（观测语义与 Unity 模板对齐） |
| `GET /api/rl/rewards` | 训练评估奖励曲线 |
| `POST /api/rl/simulate` | 多智能体引导仿真（换乘步数/拥堵/轨迹图） |
| `POST /api/llm/load` / `fine_tune` / `simulate_metrics` | LLM 加载（Mock/真实）/ 微调 / 指标报告 |
| `POST /api/dashboard/run` | 客流热力 + 引导路径面板 |
| `POST /api/plan` | 真实路线规划（engine=amap 高德 / engine=hub 枢纽内偏好规划；prefs 支持 note 自由文本需求） |
| `POST /api/features/run_existing` | 运行全部 7 类分析报告（结构化结果） |
| `GET /api/settings` / `POST /api/settings` | 运行时配置查看（密钥只回掩码）/ 保存写回 `.env`（写接口仅限本机，生产非 loopback 自动禁用） |
| `GET /mobile` | 移动端 PWA |

## 统一 CLI

```bash
zhishuxing demo        # 一键演示：导航→客流面板→模拟微调→汇总
zhishuxing train       # 连接 Unity 运行 MADDPG/MATD3 训练
zhishuxing analyze     # 分析报告（--report all / reward / congestion / …）
zhishuxing simulate    # MADDPG（或启发式回退）引导仿真
zhishuxing animate     # 生成枢纽换乘环境动图
zhishuxing serve       # 启动 Web 控制台（--production 走 waitress）
zhishuxing smoke       # Web API 冒烟检查
zhishuxing doctor      # 环境与密钥自检（--strict 把可选项缺失也计入非零退出）
zhishuxing kb-ingest   # 换乘经验文档(txt/md/html)入库为 JSONL 语料
```

## MADDPG 训练（连接 Unity）

```bash
zhishuxing train \
  --mlagents_file "D:\\Builds\\HubTransfer\\HubTransfer.exe" \
  --behavior_name HubAgent \
  --episode_limit 200 --max_train_steps 500000 --evaluate_freq 5000
# 留空 --mlagents_file 连接 Unity Editor；--algorithm MATD3 启用 TD3 变体
```

训练产物：`data/outputs/MADDPG_env_*.npy`（评估奖励）、`data/model/integrated_hub_transfer/*_actor_*_agent_*.pth`（权重）。训练完成后在 Web 控制台点击「加载最新策略权重」，即可用真实策略执行引导仿真。Unity 侧模板见 `unity/HubTransferAgent.cs`（观测/动作语义与 `rl/envs.py` 一致）。

## 测试与质量保障

- **pytest 180 项**：覆盖 Web API（含新旧两代高德响应结构）、导航偏好规划、需求档案、知识库检索、RL 运行时、合成数据、CLI
- **静态检查**：`python -m pyflakes src/ scripts/ tests/` 零告警（CI 同步执行）
- **CI**：GitHub Actions（[.github/workflows/ci.yml](.github/workflows/ci.yml)）——pyflakes + pytest（torch 走 CPU 轮子）
- **失败分支实测**：无 Key、断网、加载失败等降级路径均有对应用例或演示说明

```bash
pip install -e .[dev]
pytest
```

## 环境变量

| 变量 | 用途 |
|---|---|
| `SILICONFLOW_API_KEY` | 对话合成 / 需求解析 / OD 提取（OpenAI 兼容；未配置时走规则解析+模板回答，全链路离线可用） |
| `AMAP_REST_KEY` / `AMAP_JS_KEY` | 高德真实路线规划 / 地图 JS 底图 |
| `AMAP_SECURITY_CODE` | 高德 JS API 安全密钥（2021-12 后申请的 Key 必填） |
| `ZHISHUXING_WORKSPACE` | 工作区根目录覆盖（默认自动推导） |

支持根目录 `.env` 自动加载（零第三方依赖）；见 `.env.example`。**密钥绝无默认值，不入库。**

## 成果预览

![换乘环境动图](data/samples/transfer_env_demo.gif)
![拥堵热力图](data/samples/shenzhen_north_congestion_heatmap.png)
![安检排队对比](data/samples/security_queue_comparison.png)

## 注意事项

- `data/transfer_kb/shenzhen_north/` 换乘经验为**演示语料**（基于公开出行攻略与站方指引整理，已标注来源），站内布局请以现场为准；生产使用应替换为经授权的官方数据
- 训练需 Unity 场景（Behavior 名 `HubAgent`、连续动作），建议先用小步数验证链路；`mlagents-envs` 需从源码安装（见 `requirements.txt` 注释与 `docs/architecture.md`）
- `data/outputs` 中的分析图基于固定种子合成数据，保证可复现；接入真实训练数据后同一脚本直接出图
- 移动端「AR 实景导航」的演示素材 `AR.gif`（约 10MB）不随仓库分发，缺失时页面展示占位说明；`web/mobile/AR.gif` 放入即可启用
- 本仓库未声明统一开源许可证（pyproject 标注 Proprietary）；ML-Agents 镜像（`third_party/`，不入库）遵循其原许可
- 历史版本曾将密钥硬编码在 `UI/config_direct.py` 并已进入 git 历史，**请务必到平台作废重发**
