# 智枢星 架构

> 用途：给要理解或改动本仓结构的人。架构总览、数据流、仓库地图、关键约定、迁移映射与历史
> 问题守护表都在这里。操作步骤在 `docs/GET-START.md`；数字口径在根目录 `AGENTS.md` 的「当前状态」。

## 架构总览

综合交通枢纽智慧换乘引导系统的演示工程：Python 包提供 CLI、Flask 控制台后端与移动端 PWA
三个面，里面串着 MADDPG 强化学习、内置网格枢纽仿真、高德路线规划和一条 LLM 对话链路；
所有在线依赖都配了离线降级，没有密钥没有 Unity 也能跑完整演示。

## 数据流

```
 Unity 场景(exe/Editor)──▶ rl.envs ──▶ rl.runner(MADDPG/MATD3)──▶ data/outputs/*.npy
                                                              data/model/**/*.pth
                                                                        │
 core.navigation(A* + 设施层) ──▶ core.simulation ◀──────────────────────┘
        │                            │        ▲  rl.runtime（策略加载/推理，无权重回退启发式）
        ▼                            ▼        └─ PolicyProtocol：core 与 rl 的唯一接缝
 core.flow ──▶ core.system（编排）──▶ analysis.plotting（热力/轨迹快照）
                    │
 webapp.service ────┴──▶ webapp.app（Flask）──▶ Web 控制台 / 移动端 PWA
        │
        ├── llm.assistant ──▶ llm.profile（需求档案）+ llm.kb（BM25 经验检索）
        │        └─ 起终点命中地标 → 偏好规划；否则 ▼
        ├── planning.amap（高德 REST + LLM/正则 OD 提取）
        └── analysis.reports（7 类报告）──▶ data/outputs/*.{png,csv,gif,json}
```

四条设计原则，改代码时它们是会咬人的约束：

1. **逻辑平移不重设计**：合成数据的数值特征、MADDPG 超参与产物命名、API 契约、前端交互都
   保持历史形态。
2. **单一来源**：字体配置、滑窗平均、CSV IO、合成数据生成器已全部归一到 `analysis/`。
3. **路径零 CWD 依赖**：产物一律锚定 workspace（`config.Paths`），密钥全部走环境变量。
4. **可选依赖隔离**：`torch` / `mlagents_envs` / `openai` 都是可选，Web 与演示链路对它们无硬依赖。

## 仓库地图

| 路径 | 职责 | 关键点 |
|---|---|---|
| `src/zhishuxing/core/` | 枢纽领域内核：A* 导航、动态客流、行人级引导仿真、面板编排、GIF 动图 | 不依赖 torch 与网络；`PolicyProtocol` 是 core 与 rl 的唯一接缝 |
| `src/zhishuxing/rl/` | MADDPG/MATD3 算法、Unity 环境封装、训练循环、推理运行时 | `runtime.py` 无 torch 可用；`envs.py` 顶层 import `mlagents_envs`，缺包即报错 |
| `src/zhishuxing/llm/` | 对话式换乘助手：需求档案、动作协议(`actions.py` 白名单+信封)、BM25 经验检索、回答合成、LLM 适配器 | `assistant.py` 的五步链路每一步都有确定性降级 |
| `src/zhishuxing/planning/` | 真实路线规划：OD 提取 → 高德地理编码 → 公交换乘 | 只调 v3 的 `geocode` 与 `direction/transit/integrated` 两个接口 |
| `src/zhishuxing/analysis/` | 7 类报告、合成数据、绘图与 CSV IO | 全仓唯一一份字体/滑窗平均/合成数据/CSV IO 实现 |
| `src/zhishuxing/webapp/` | Flask 应用工厂、服务层、控制台前端 | 端点清单见 `app.py`，改契约要同步 `static/app.js` |
| `configs/` | 导航网格、演示场景、训练超参、微调样例 | `hub_default.json` 改结构会让 smoke 直接失败 |
| `data/transfer_kb/` | 换乘经验语料（22 篇源文档 + `corpus.jsonl`） | 源文档改了要重跑 `zhishuxing kb-ingest` |
| `data/samples/` | 入库的参考产物，README 展示图与奖励 npy 来源 | 只由脚本重建，别手改 |
| `data/` 下 `outputs/`、`model/`、`runs/` | 运行时产物 | 全部 gitignore，首次运行自动建目录 |
| `web/mobile/` | 移动端 PWA | 由 `/mobile` 同源托管；无构建步骤 |
| `unity/` | Unity 侧智能体脚本与接入说明 | **不是可构建工程**，没有 Assets/ProjectSettings |
| `scripts/` | 工具脚本：品牌 PNG 图标渲染（`render_brand_assets.py`）与 Windows exe 打包（`build_exe.py`） | 渲染需要 Pillow（已在 dev 组）；打包需要 pyinstaller |
| `packaging/` | Windows 单文件 exe 的打包资产：`exe_entry.py` 打包入口 + `app.ico`、`_build_version.txt` 两个生成物 | 后两者由 `build_exe.py` 生成，勿手改 |
| `tests/` | pytest 套件 | `test_settings.py` 一个文件占 98 个用例 |
| `code_optimization/` | 行人仿真向量化基准与报告 | 跑基准会重写入库的 `benchmark_results.json` |
| `legacy/ui/` | 迁移前的 Streamlit 原型 | 不参与测试，也不在 pyflakes 门禁里 |
| `docs/DEPLOY.md` | Docker 优先的部署手册 | — |

`unity/README.md`（Unity 接入）与 `code_optimization/report.md`（性能基准）是各自主题的深度
文档，保留独立文件；其余总览性内容不要在三处复述。

## 关键约定（违反会出问题的）

- **配置加载优先级：shell 环境变量 > `.env` > 代码默认值。** `config._load_dotenv()` 默认
  `override=False`。违反后果：`.env` 写了值但不生效，而 `doctor` 显示的是生效值，看起来像
  "配置丢了"。
- **`reload_env()` 是唯一 `override=True` 入口**，只在 `POST /api/settings` 保存后调用，只刷新
  `.env` 里出现的键。**在设置页保存过的项以 `.env` 为准**，会盖掉同名 shell 变量；不要把只存在
  于 shell 的密钥在设置页清空——清空会写 `KEY=` 并立即生效，等效于抹掉这把密钥。
- **写回 `.env` 的五道闸**：① 只接受 `SETTINGS` 注册表内 7 个键；② 值含换行/回车/NUL/`=`
  拒绝；③ 超 400 字符拒绝、`numeric` 项要有限正数；④ 先备份 `.env.bak` 再 `os.replace` 原子
  替换；⑤ `POST /api/settings` 只认 `remote_addr` 是否 loopback，**不信 `X-Forwarded-For`**。
  全部有用例钉着（`tests/test_settings.py`）。
- **对外一律掩码。** `mask_value()` 只给前 2 字符与长度，≤4 连前缀都不给；任何新接口、CLI
  输出、日志不得回显明文。
- **路径只由 `config.py` 推导。** 产物走 `config.paths`（`ZHISHUXING_WORKSPACE` 可覆盖）；
  相对 CWD 的写法就是 bug。副作用：`web/mobile/` 也从 workspace 取，改 workspace 会让
  `/mobile` 变 404。
- **降级逻辑住在各适配器里，`settings.py` 只报状态不改变行为。** 留空即：LLM → Mock 模板；
  底图 → Canvas 折线；规划 → 内置枢纽引擎；策略 → `HeuristicPolicy`。新增在线能力要同时提供
  "没有它也能跑"的路径，并把 `degrades_to` 写进注册表。
- **`analysis/` 是唯一来源。** 字体配置、`moving_average`、CSV 读写、合成数据生成器归一到
  `analysis/{plotting,io_utils,synthetic}.py`；别处再写一份字体会让无头环境重新出现方框字。
- **`matplotlib.use("Agg")` 在 `analysis/plotting.py` 顶层。** 要绘图的模块必须先 import 它
  再 pyplot，否则无头环境挂起。
- **产物命名不能改。** `*_env_*_number_*_seed_*.npy` 与 `*_actor_number_*_step_*k_agent_*.pth`
  是读写双方契约，改了等于让既有训练产物失效。
- **`legacy/` 不参与测试**（同名函数歧义），在 `.docsignore` 与门禁之外。
  `third_party/` 上游镜像目录已于 2026-10-07 删除（代码零引用，训练依赖改从 ml-agents
  官方仓库源码安装）；`.gitignore` 仍排除该名，防止镜像被误提交。
- **不要在本仓写真实密钥。** 密钥取值无默认值，空串按未配置处理。

## 迁移映射（旧 → 新）

v2 重构是把历史散装脚本平移进包，不是重写。左列**仓库里已不存在**（仅作迁移记录保留），
右列全部可验证：

| 旧位置 | 新位置 |
|---|---|
| `program/MADDPG/MADDPG_main.py` | `rl/runner.py` + `cli.py train` |
| `program/MADDPG/{maddpg,matd3}.py` | `rl/agents.py` |
| `program/MADDPG/{networks,replay_buffer,environment}.py` | `rl/{networks,buffer,envs}.py` |
| `program/MADDPG/zhishuxing/rl_bridge.py` | `rl/runtime.py` + `core/simulation.py` |
| `program/MADDPG/zhishuxing/{navigation,system,visualization}.py` | `core/{navigation,system}.py` + `analysis/plotting.py` |
| `program/MADDPG/zhishuxing/adapters.py` | `llm/adapters.py` |
| `program/MADDPG/webapp/*` | `src/zhishuxing/webapp/*` |
| `program/MADDPG/plot_*.py`、`animate_transfer_env.py` | `analysis/{synthetic,reports}.py` + `core/animation.py` |
| `program/fine/*` | `analysis/{reports,synthetic}.py` |
| `program/UnityTemplate/` | `unity/` |
| `UI/mobile_app.html` 等 | `web/mobile/` |
| `UI/jiaohu*.py` | `legacy/ui/`（密钥已移除，改走环境变量） |
| `program/data_train/` | `data/samples/`（参考产物）/ `data/outputs/`（运行时） |

## 已修过的历史问题（别退回去）

| 症状 | 现在的守护 |
|---|---|
| 安检排队图南区「无引导 / MADDPG」两条线的数据与标签互换 | `analysis/reports.py` 里注明了这处，改图要看 `queue` 报告 |
| 奖励曲线会误选任意 `.npy` | 只匹配 `*_env_*.npy` 这一种命名 |
| `plot_finetune_metrics` 无 Agg 后端且 `plt.show()`，无头环境挂住 | `analysis/plotting.py` 顶层 `matplotlib.use("Agg")` |
| `mlagents-envs>=1.1.0` 装不上的 pin 与没用的 `gym` 依赖 | 已删除，改为源码安装说明 |
| MATD3 `save_model` 不建目录、ReplayBuffer 样本不足报错含糊 | `rl/agents.py`、`rl/buffer.py` |
| 收集了偏好参数却把 `strategy` 写死为 0 | `planning/amap.py` 的 `resolve_strategy()` 真正映射 |
| 高德 v3 嵌套响应折线解析为空 | `_iter_segment_steps()` 兼容新旧两代，`tests/test_amap.py` 覆盖 |
| 地图容器零尺寸时建图得到空白 | 前端先显示容器再建图 |
| TensorBoard 日志目录错位 | `rl/runner.py` 的 `log_dir` 已对齐产物命名 |
| 移动端 `sw.js` 预缓存引用不存在的 `VR1.png` | 缓存清单只有 9 个真实文件 |
| 移动端引用未入库的 10 MB `AR.gif` | `onerror` 回退占位说明 |
| 无 `.gitignore` 时 node_modules/密钥一起入库 | `.gitignore` 已治理（提交 `b62cc2c3` 曾有 44389 个文件 524 MB；历史已重写收口） |
| manifest 主题色与页面 meta 不一致 | 两边都是 `#4f46e5` |
| 全仓 pyflakes 告警 | 已清零并写进 CI |

## 子目录说明索引

各一级目录的板块说明：`code_optimization/`、`configs/`、`data/`、`legacy/`、`scripts/`、
`src/`、`tests/`、`unity/`、`web/` 各自有 `README.md`；`docs/` 是文档目录本身。

## 已知架构问题

- 演示/教学工程定位：会话曾仅内存（2.3 起已 SQLite 持久化）、无账号、单实例、默认启发式
  回退与合成数据；走向「可用」的前置是多实例与真实环境验证（`unity/` 路线 3.0 的录屏判据
  仍待真实环境验证）。
