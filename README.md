# 智枢星 · 动态客流下的综合交通枢纽智慧换乘引导

> 用途：第一次打开这个仓库的人。看完知道它是什么、能不能解决你的问题、怎么立刻跑起来。
> 要真的用起来或改它，看 [上手手册](docs/getting-started.md)；被本文引用的事实以 [AGENTS.md](AGENTS.md) 为准。

<p align="center">
  <img src="src/zhishuxing/webapp/static/assets/logo-full.svg" alt="智枢星" width="360" />
</p>

面向大型综合交通枢纽（演示场景为深圳北站）的换乘引导系统：**高峰动态客流下，乘客带着各自的
个性化需求怎么快速、舒适地完成站内换乘**。乘客用口语说需求（赶时间、优先直梯、途经卫生间、
带老人行李、轮椅无障碍），系统解析成结构化需求档案 → 偏好感知规划 → 引用站内换乘经验 →
解释「为什么这样走符合你的需求」。引导策略由 MADDPG 多智能体强化学习训练，
没有训练权重时自动回退启发式并如实标注。

**180 个测试通过 · 9 个 CLI 子命令 · 24 个路由注册（23 条不同路径）· 7 类分析报告（复核：`AGENTS.md` 的「当前真实状态」）**

## 30 秒跑通

不需要密钥、不需要网络、不需要 Unity。在仓库根目录：

```bash
pip install -e .
zhishuxing doctor
zhishuxing smoke
```

`doctor` 逐条报告哪些在线能力已配、当前降级成什么（密钥只回掩码）。全新 clone 的末三行：

```
  在线能力 : 真实路线规划 降级为内置枢纽引擎 / 高德底图 未配置 JS Key，回退 Canvas 折线 / 真实 LLM 降级为 Mock 确定性模板
  离线演示 : 始终可用（Mock 对话 + Canvas 折线 + 内置枢纽引擎），与是否填写密钥无关
  结论     : 必需项缺失 3 项 → AMAP_REST_KEY, AMAP_JS_KEY, AMAP_SECURITY_CODE（离线演示不受影响，仅对应在线能力受限）
```

这一条退出码是 1，含义是"在线能力没配齐"，不是装坏了。`smoke` 在当前进程内打一遍全部核心端点：

```
Web smoke test passed.
```

再往下任选：

```bash
zhishuxing demo                   # 导航 → 客流面板 → 模拟微调 → 汇总 JSON
zhishuxing analyze --report all   # 7 类报告，全 OK 时退出码 0
zhishuxing serve --host 127.0.0.1 --port 7860   # 控制台 http://127.0.0.1:7860，PWA /mobile
```

缺依赖时的降级是设计好的：没装 `torch` 时 RL 的 6 个用例跳过而非失败、策略回退启发式；
没装 `mlagents_envs` 只有 `train` 用不了；没装 `openai` 或没配 LLM 密钥时对话走 Mock 模板，
响应里如实标注当前是本地模板模式。

## 9 个子命令

| 命令 | 一句职责 |
|---|---|
| `demo` | 一键演示：加载导航图 → 渲染客流引导面板 → 跑一遍微调接口 → 写汇总 JSON（`--reports` 顺带出 7 类报告） |
| `train` | 连 Unity（`.exe` 或 Editor）跑 MADDPG / MATD3 训练，产出 npy 奖励、pth 权重、TensorBoard 日志 |
| `analyze` | 出分析报告，`--report` 取 `reward` `heatmap` `transfer` `queue` `efficiency` `finetune` `animation` 或 `all` |
| `simulate` | 在内置网格枢纽上跑多智能体引导仿真，输出到达统计与轨迹图 |
| `animate` | 生成枢纽换乘环境的行人动图（GIF） |
| `serve` | 起 Web 控制台与 PWA（`--production` 用 waitress 托管；监听非本机时默认关闭密钥写入） |
| `smoke` | 不占端口的 API 冒烟检查，改完后端最先跑它 |
| `doctor` | 配置体检：7 项托管配置的状态、当前降级行为；退出码 0 就绪 / 1 有缺失 |
| `kb-ingest` | 把 txt/md/html 换乘经验清洗入库为 JSONL 语料，`--query` 可当场自检检索 |

## 双端形态

同一套 Flask 后端（`src/zhishuxing/webapp/`），两端都是请求/响应，**没有流式推送**（无 SSE、无 WebSocket）。

- **Web 控制台** `http://127.0.0.1:7860/`：概览、路线规划、枢纽导航、客流面板、RL 智能体、
  分析报告、设置 7 个视图，加右侧可展开的智能换乘助手对话面板。地图优先走高德 JS API 真实底图，
  未配 Key 或加载失败自动回退 Canvas 折线并注明原因。
- **移动端 PWA** `http://127.0.0.1:7860/mobile`：同源托管，可安装、ServiceWorker 离线缓存；
  只调 `/api/chat` 与 `/api/settings`，后端不可用时回退本地演示数据并写明「演示数据」。

## HTTP 路由（逐条对 `webapp/app.py`）

| 方法与路径 | 干什么 |
|---|---|
| `GET /` | 控制台页面，Jinja 注入前端配置 |
| `GET /health` | 健康检查与已加载导航图 |
| `GET /outputs/<file>` | 取报告图、CSV、GIF |
| `GET /mobile`、`GET /mobile/<file>` | 同源提供 PWA 页面与资源（保证 SW scope 覆盖） |
| `POST /api/navigation/load` | 按路径加载导航图 |
| `POST /api/navigation/plan` | A\* 规划，支持必经地标 |
| `GET /api/navigation/grid` | 完整网格：禁行格、地标、设施语义层与中文标签 |
| `GET /api/scenarios` | 3 组演示场景（坐标已解析） |
| `POST /api/llm/load` | 加载 LLM 适配器（Mock / 真实；真实失败时注明 `real_adapter_error`） |
| `POST /api/llm/fine_tune` | 微调接口，Mock 实现写元数据 JSON |
| `POST /api/llm/simulate_metrics` | 微调指标曲线（PNG + CSV + 序列） |
| `GET /api/rl/status` | 扫描训练产物、当前策略来源与加载错误 |
| `POST /api/rl/load_policy` | 按 agent 加载最新 step 的 actor 权重 |
| `POST /api/rl/act` | 观测 → 动作推理，无权重回退启发式 |
| `GET /api/rl/rewards` | 评估奖励曲线（读 `*_env_*.npy`） |
| `POST /api/rl/simulate` | 多智能体引导仿真 + 轨迹图 |
| `POST /api/dashboard/run` | 客流热力 + 引导路径面板 |
| `POST /api/features/run_existing` | 进程内跑完 7 类报告，逐报告返回状态 |
| `POST /api/plan` | 路线规划：`engine=amap` 走高德，`engine=hub` 走枢纽内偏好 A\* + RL 仿真 |
| `POST /api/chat` | 对话式换乘助手（需求档案、偏好规划、经验引用） |
| `POST /api/chat/reset` | 重置会话 |
| `GET /api/settings` | 配置状态（值一律掩码，故对局域网只读开放） |
| `POST /api/settings` | 保存并写回 `.env` + 热重载（仅限本机 loopback 请求） |

## 目录怎么分

| 目录 | 负责 |
|---|---|
| `src/` | Python 包本体：`core` `rl` `llm` `planning` `analysis` `webapp` 六个子包 |
| `configs/` | 导航网格、演示场景、训练超参、微调样例指令 |
| `data/` | 知识库语料与入库参考产物；`outputs` `model` `runs` 是运行时目录 |
| `web/` | 移动端 PWA 静态文件 |
| `unity/` | Unity 侧智能体脚本与接入说明（不是可构建工程） |
| `tests/` | pytest 套件，按链路分 11 个文件 |
| `scripts/` | 品牌图标渲染工具 |
| `code_optimization/` | 行人仿真向量化基准与报告 |
| `legacy/` | 迁移前的 Streamlit 原型归档 |

逐目录说明见各目录的 `README.md`；`third_party/`（上游镜像）与产物目录见 `.docsignore`。

## 已知做不到什么

1. **RL 训练需要你自己提供 Unity 场景**：仓库里的 `unity/` 只有智能体脚本，没有工程文件，
   也没构建过。`mlagents_envs` 在 PyPI 上没有 1.x 版本，要从源码安装，否则 `train` 直接报
   `ModuleNotFoundError`。不训练时引导仿真是启发式回退。
2. **分析图除奖励曲线外都是固定种子的合成数据**，用来说明"接上真实数据后同一脚本直接出图"，
   不能当实测结论引用。换乘米数按 `cell_size_m = 30` 换算，是示意值。
3. **不配高德密钥就没有真实路线与真实底图**：`engine=amap` 返回 400 说明原因，
   规划自动改用内置枢纽引擎，地图回退 Canvas 折线。
4. **知识库的 22 篇经验是按公开攻略手工整理的演示语料**，不是站内实测数据，站内布局以现场为准。
5. 对话会话与需求档案只存进程内存，重启服务即清空；没有账号体系。
6. 移动端「AR 实景导航」素材（约 10 MB）不随仓库分发，缺失时显示占位说明。

## 环境要求

Python ≥ 3.10（CI 跑 3.11 与 3.12）。必需依赖 5 个：`numpy`、`matplotlib`、`flask`、
`waitress`、`requests`。可选：`.[train]` → torch + tensorboard，`.[llm]` → openai，
`.[dev]` → pytest。密钥走环境变量或根目录 `.env`（零第三方 dotenv 依赖，自己解析），
配置项与作用见 [.env.example](.env.example) 与手册第 5 节。

## 许可

`pyproject.toml` 标注 `Proprietary`，仓库内没有 LICENSE 文件，未授予开源许可。
`third_party/`（不入库）下的上游镜像遵循各自原许可。
git 历史未清理：首次提交里的旧配置模块含真实密钥形状的字面量，**相关密钥须在平台侧作废重发**，
删掉当前文件不等于历史安全。

---

准备改这个仓库的 AI 助手请先读 [AGENTS.md](AGENTS.md)。
