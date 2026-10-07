# src/ —— Python 包本体

> 用途：说明 `src/zhishuxing/` 各子包管哪一段、每个关键文件干什么、被谁调用。
> 全仓库的运行时逻辑都在这里；`configs/` 与 `data/` 是它的输入，`web/` 与 `unity/` 是它的两端。

包以 src-layout 打包（`pip install -e .`），入口 `zhishuxing = zhishuxing.cli:main`（复核：
`pyproject.toml` 的 `[project.scripts]`）。`zhishuxing/__init__.py` 里的 `__version__` 与
`pyproject.toml` 的 `version` 各自写了一份，当前都是 `2.4.0`（复核：
`python -c "import zhishuxing;print(zhishuxing.__version__)"`）。

## 文件清单

`src/` 根下只有 `README.md`、包目录 `zhishuxing/`，以及生成物 `zhishuxing.egg-info/`。
下表是包顶层那 4 个 `.py`。包内共 35 个 `.py` 且全部入库（复核：
`find src/zhishuxing -name "*.py" -not -path "*__pycache__*" | wc -l` 与
`git ls-files "src/**/*.py" | wc -l`，两条都输出 `35`）。

| 文件 | 干什么 | 被谁调用 |
|---|---|---|
| `__init__.py` | 只定义 `__version__`（桶导出已删：全仓符号一律从具体子模块导入） | `AGENTS.md` 的版本复核命令 |
| `config.py` | 推导 workspace 根与全部产物路径；解析 `.env`；暴露 `amap_config()` / `siliconflow_config()` | 几乎每个模块（`import .. as cfg`） |
| `settings.py` | 7 项托管配置的注册表、掩码回读、写入校验、`.env` 原子写与热重载、loopback 判定 | `webapp/app.py` 的 `/api/settings`、`cli.py` 的 `doctor` |
| `cli.py` | 9 个子命令的 argparse 定义与派发，每个 `cmd_*` 内部才 import 重依赖 | 控制台脚本 `zhishuxing`；`tests/test_cli.py` |

## 子目录

`src/` 的二级目录只有 `zhishuxing/` 一个；再往下是它的 6 个子包，下钻说明写在本节。
`__pycache__/` 是字节码缓存、`zhishuxing.egg-info/` 是 `pip install -e .` 的生成物，
两者都被 `.gitignore` 挡着，不算内容（复核：`git check-ignore -v src/zhishuxing.egg-info`）。

| 子目录 | 负责 |
|---|---|
| `zhishuxing/` | 整个 Python 包：4 个顶层模块 + 下面 6 个子包，入口是 `cli.py` |
| `zhishuxing/core/` | 7 个 `.py`：A* 导航、场景解析、客流撒点、行人引导仿真、面板编排、GIF 动图。不依赖 torch，不依赖网络 |
| `zhishuxing/rl/` | 7 个 `.py`：MADDPG/MATD3 算法、网络与回放池、Unity 环境封装、训练循环、推理运行时 |
| `zhishuxing/llm/` | 7 个 `.py`：需求档案、BM25 经验检索、回答编排、适配器（Mock 与 SiliconFlow）、动作协议、会话持久化 |
| `zhishuxing/analysis/` | 5 个 `.py`：7 类报告、合成数据、绘图与字体、CSV IO。全仓唯一一份实现 |
| `zhishuxing/planning/` | 2 个 `.py`：`amap.py` 真实路线规划（OD 提取 → 地理编码 → 公交换乘） |
| `zhishuxing/webapp/` | 3 个 `.py` + `templates/index.html` + `static/`：应用工厂、服务层、控制台前端 |

复核每条计数：`find src/zhishuxing/<子包> -name "*.py" -not -path "*__pycache__*" | wc -l`
（数字含各包的 `__init__.py`）。

## `zhishuxing/core/` —— 枢纽领域内核（不依赖 torch，不依赖网络）

| 文件 | 干什么 | 被谁调用 |
|---|---|---|
| `navigation.py` | 导航图加载与 A*：4 邻接 + 曼哈顿启发式；`RouteCostSpec` 承载标签罚项/禁行标签/软必经点；`plan_with_preferences()` 做偏好加权、软必经取舍与硬约束降级；`resolve_hub_landmark()` 把「A口/安检/直梯」等中文说法映射到坐标 | `system.py`、`simulation.py`、`flow.py`、`webapp/service.py`、`llm/assistant.py`、`llm/profile.py` |
| `scenarios.py` | `PassengerGroup` 数据类与场景解析：`start_landmark`/坐标两种写法都能读 | `cli.py`（demo/simulate/smoke）、`service.py` |
| `flow.py` | 按释放时间沿路径撒乘客 + 随机扩散，产出归一化的动态客流矩阵 | `system.py` |
| `simulation.py` | 行人级多智能体引导仿真：观测 `[dx,dy,vx,vy,3×邻近相对位置]`（`OBS_DIM=10`）、动作 `[前进,转向]` 且转向按整档 90° 量化；策略经 `PolicyProtocol` 注入，`HeuristicPolicy` 是内置回退 | `rl/runtime.py`、`cli.py simulate`、`service.py` |
| `system.py` | `ZhiShuXingSystem` 编排：导航 + 客流 + 引导文案 + 面板渲染 + 一次跑完 7 类报告 | `service.py`、`cli.py demo` |
| `animation.py` | 社会力微观行人模型与 GIF 动图；`step()` 已向量化（基准见 `../legacy/code_optimization/README.md`） | `analysis/reports.py`、`cli.py animate` |

## `zhishuxing/rl/` —— 多智能体强化学习

各子包 `__init__.py` 是空壳：全仓调用方都从具体子模块导入，`import zhishuxing` 也不会连带导入 `torch` / `mlagents_envs`（可选依赖隔离见 `docs/ARCHITECTURE.md`）。

| 文件 | 干什么 | 被谁调用 |
|---|---|---|
| `runtime.py` | **不需要 torch 也能 import**：扫描 `data/model/**/*.pth`、按 agent 取最新 step 加载 actor、`act()` 推理、读 `*_env_*.npy` 画奖励曲线；无权重或无 torch 时回退启发式并把 `policy_source` 写成「启发式回退」 | `webapp/service.py`、`cli.py simulate`、`tests/test_rl_runtime.py` |
| `agents.py` | MADDPG 与 MATD3 两套更新逻辑（TD3 三技巧为独立类路径），`--algorithm` 真正生效 | `runner.py` |
| `networks.py` | `Actor` / `Critic_MADDPG` / `Critic_MATD3`；`runtime.py` 按 `fc1.weight` / `fc3.weight` 形状反推网络尺寸，所以权重目录不必带训练脚本 | `agents.py`、`runtime.py` |
| `buffer.py` | 按 agent 分槽的环形经验回放 | `runner.py` |
| `envs.py` | Unity ML-Agents 的 gym 风格封装；`mlagents_envs` 走 `_mlagents_imports()` 懒加载，缺包时调用即报错并给源码安装指引 | `runner.py` |
| `runner.py` | 训练/评估双环境主循环、噪声线性衰减、按 `evaluate_freq` 落 npy 与权重 | `cli.py train` |

## `zhishuxing/llm/` —— 对话式换乘引导

| 文件 | 干什么 | 被谁调用 |
|---|---|---|
| `assistant.py` | `TransferAssistant.handle()` 五步编排：解析需求 → 合并会话档案 → OD 分流（命中地标走枢纽规划，否则走高德）→ BM25 检索 top-3 → 合成回答；会话态在内存，重启即失 | `service.py` 的 `/api/chat` |
| `profile.py` | `PassengerProfile`：4 维优先级 + 硬约束 + 软偏好 + 5 类画像（画像按 `PERSONA_IMPLICATIONS` 派生隐含偏好）；规则关键词优先、真实 LLM JSON 兜底；`to_cost_spec()` 翻译成 A* 参数 | `assistant.py`、`service.py plan_route`、`tests/test_profile.py` |
| `kb.py` | 纯 Python BM25（中文 2-gram + ASCII 词，标题重复一次加权）；`ingest_directory()` 把 txt/md/html 清洗切段写入 JSONL，按 id 覆盖所以可重复跑 | `assistant.py`、`cli.py kb-ingest` |
| `adapters.py` | `MockLLMAdapter`（确定性模板，`mock=True`）与 `SiliconFlowLLMAdapter`（OpenAI 兼容，缺 Key 或未装 `openai` 时抛 `RuntimeError` 由调用方降级） | `system.py`、`service.py`、`assistant.py` |
| `actions.py` | 智能体动作协议：动作注册表白名单 + 两条触发通道（"打开/切换板块"规则短路不经 LLM、泛化请求由 LLM 输出 JSON 动作信封），服务端校验后下发前端执行（如 `switch_tab`） | `assistant.py` 的 `handle()` |
| `session_store.py` | 会话持久化：SQLite 单文件（`data/runs/sessions.db`）写穿读缓存，重启按 `session_id` 惰性回填，30 天不活跃清理 | `assistant.py`（构造 `SessionStore`） |

## `zhishuxing/planning/` 与 `zhishuxing/analysis/`

| 文件 | 干什么 | 被谁调用 |
|---|---|---|
| `planning/amap.py` | 真实路线规划：LLM 或正则提 OD → 高德地理编码 → 公交换乘（`resolve_strategy()` 把偏好真的映射进 `strategy`）→ 分段详情/提醒；兼容新旧两代响应结构 | `service.py plan_route`、`assistant.py` |
| `analysis/reports.py` | 7 个 `run_*_report()`，每个可 import 也可 CLI，返回 `{ok, files, 摘要}`；输出锚定 `data/outputs/` | `cli.py analyze`、`system.py run_reports()`、`service.py` |
| `analysis/synthetic.py` | 全仓唯一一份合成数据生成器（拥堵矩阵、换乘时间分布、安检排队、微调指标），默认种子固定 | `reports.py`、`service.py` |
| `analysis/plotting.py` | 中文字体配置、`moving_average`、`matplotlib.use("Agg")` 与枢纽快照渲染 `HubVisualizer` | `reports.py`、`core/system.py`、`core/simulation.py` |
| `analysis/io_utils.py` | 报告用的 CSV 读写 | `reports.py` |

## `zhishuxing/webapp/` —— Flask 控制台后端

| 文件 | 干什么 | 被谁调用 |
|---|---|---|
| `app.py` | `create_app()` 应用工厂与全部路由；文件末尾 `app = create_app()` 供 `flask run` 直接发现 | `cli.py serve`/`smoke`、`tests/conftest.py` |
| `service.py` | `ZhiShuXingWebService`：把 core / rl.runtime / planning / llm 编排成 API 语义，产物 URL 一律 `/outputs/<file>` | `app.py` |
| `templates/index.html` | 控制台骨架，Jinja 注入 `window.ZSX_CONFIG`（JS Key、安全密钥、`settingsWritable`） | `app.py` 的 `home()` |
| `static/app.js` | 7 个视图的交互与 Canvas 绘制；高德脚本加载失败或未配 Key 时回退 Canvas 折线 | 浏览器 |
| `static/style.css`、`static/assets/` | 样式与品牌矢量/图标 | 同上 |

## 和谁打交道

- **上游**：`configs/`（导航图与超参）、`data/transfer_kb/`（检索语料）、`data/model/`（训练权重）。
- **下游**：`data/outputs/`（图/CSV/GIF/JSON）、Web 控制台、`web/mobile/` PWA。
- **改这里之后要跑**：`python -m pytest` 与 `zhishuxing smoke`（见仓库根 `AGENTS.md`）。

## 别动

- `src/zhishuxing.egg-info/`：`pip install -e .` 的生成物，已被 `.gitignore` 的 `*.egg-info/` 排除，
  手改无意义，重装即覆盖。
- `rl/envs.py` 的顶层 `mlagents_envs` 导入：不要"顺手"改成惰性导入，`runner.py` 依赖它
  在训练启动前就暴露环境问题。
- `analysis/plotting.py` 的 `matplotlib.use("Agg")`：去掉后无头环境会在报告渲染处挂住。
