# 给 AI 的项目说明

> 用途：给 AI 编码助手。这里是事实与约束，不含介绍性文字。改动本仓库前先读这份。
> README.md 与 docs/getting-started.md 里被引用的事实以本文件为准，它们只链接不复述。

## 一句话

综合交通枢纽智慧换乘引导系统的演示工程：Python 包提供 CLI、Flask 控制台后端与移动端 PWA 三个面，
里面串着 MADDPG 强化学习、内置网格枢纽仿真、高德路线规划和一条 LLM 对话链路；
所有在线依赖都配了离线降级，所以没有密钥没有 Unity 也能跑完整演示。

## 当前真实状态

复核环境：Windows + Python 3.12.0（`python --version`），已 `pip install -e .`；
本机 `torch`、`openai`、`Pillow` 在装，`mlagents_envs` 没装（所以下面 `train` 那条会失败）。

| 项 | 值 | 复核命令 |
|---|---|---|
| 测试 | `180 passed`（17–19 s） | `python -m pytest` |
| 用例分布 | `180 tests collected` | `python -m pytest -o addopts="" --collect-only -q` |
| 静态检查 | `0 告警`，退出码 0 | `python -m pyflakes src/ scripts/ tests/` |
| API 冒烟 | `Web smoke test passed.`，退出码 0 | `zhishuxing smoke` |
| 配置体检 | 本机配好 3 项必需密钥 → 退出码 0；全新 clone 无 `.env` → 退出码 1 并列缺 3 项 | `zhishuxing doctor` |
| 7 类报告 | 全部 `OK`，退出码 0 | `zhishuxing analyze --report all` |
| HTTP 路由 | 24 个注册 / 23 条不同路径（`/api/chat` 与 `/api/settings` 各含 GET+POST） | `grep -cE '@app\.(get\|post)\(' src/zhishuxing/webapp/app.py` |
| CLI 子命令 | 9 个 | `grep -cE 'add_parser\("[a-z-]+"' src/zhishuxing/cli.py` |
| CI | 定义在 `.github/workflows/ci.yml`：ubuntu-latest × Python `3.11` / `3.12`，装 `.[dev]` + CPU 版 torch，先 pyflakes 再 pytest（`MPLBACKEND=Agg`）。本机没有 `gh`，运行绿不绿要在 Actions 页面看 | `cat .github/workflows/ci.yml` |
| 版本 | `2.1.0` | `python -c "import zhishuxing;print(zhishuxing.__version__)"`，另一份在 `pyproject.toml` |
| Python 要求 | `>=3.10`（CI 只跑 3.11/3.12） | `pyproject.toml` 的 `requires-python` |
| 许可证 | Proprietary，仓库内没有 LICENSE 文件 | `git ls-files` 里搜不到 license |
| 运行时依赖 | `numpy`、`matplotlib`、`flask`、`waitress`、`requests` | `pyproject.toml` 的 `dependencies` |
| 可选依赖 | `[train]` → torch + tensorboard；`[llm]` → openai；`[dev]` → pytest。`mlagents_envs` 不在任何一格里 | `python -c "import importlib.metadata as m;print(m.requires('zhishuxing'))"` |

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

1. **逻辑平移不重设计**：合成数据的数值特征、MADDPG 超参与产物命名、API 契约、前端交互都保持历史形态。
2. **单一来源**：字体配置、滑窗平均、CSV IO、合成数据生成器历史上散在 6 份以上，已全部归一到 `analysis/`。
3. **路径零 CWD 依赖**：产物一律锚定 workspace（`config.Paths`），密钥全部走环境变量。
4. **可选依赖隔离**：`torch` / `mlagents_envs` / `openai` 都是可选，Web 与演示链路对它们无硬依赖。

## 仓库地图

| 路径 | 职责 | 关键点 |
|---|---|---|
| `src/zhishuxing/core/` | 枢纽领域内核：A* 导航、动态客流、行人级引导仿真、面板编排、GIF 动图 | 不依赖 torch 与网络；`PolicyProtocol` 是 core 与 rl 的唯一接缝 |
| `src/zhishuxing/rl/` | MADDPG/MATD3 算法、Unity 环境封装、训练循环、推理运行时 | `runtime.py` 无 torch 可用；`envs.py` 顶层 import `mlagents_envs`，缺包即报错 |
| `src/zhishuxing/llm/` | 对话式换乘助手：需求档案、BM25 经验检索、回答合成、LLM 适配器 | `assistant.py` 的五步链路每一步都有确定性降级 |
| `src/zhishuxing/planning/` | 真实路线规划：OD 提取 → 高德地理编码 → 公交换乘 | 只调 v3 的 `geocode` 与 `direction/transit/integrated` 两个接口 |
| `src/zhishuxing/analysis/` | 7 类报告、合成数据、绘图与 CSV IO | 全仓唯一一份字体/滑窗平均/合成数据/CSV IO 实现 |
| `src/zhishuxing/webapp/` | Flask 应用工厂、服务层、控制台前端 | 端点清单见 `app.py`，改契约要同步 `static/app.js` |
| `configs/` | 导航网格、演示场景、训练超参、微调样例 | `hub_default.json` 改结构会让 smoke 直接失败 |
| `data/transfer_kb/` | 换乘经验语料（22 篇源文档 + `corpus.jsonl`） | 源文档改了要重跑 `zhishuxing kb-ingest` |
| `data/samples/` | 入库的参考产物，README 展示图与奖励 npy 来源 | 只由脚本重建，别手改 |
| `data/outputs|model|runs/` | 运行时产物 | 全部 gitignore，首次运行自动建目录 |
| `web/mobile/` | 移动端 PWA | 由 `/mobile` 同源托管；无构建步骤 |
| `unity/` | Unity 侧智能体脚本与接入说明 | **不是可构建工程**，没有 Assets/ProjectSettings |
| `scripts/` | 唯一的工具脚本：品牌 PNG 图标渲染 | 需要 Pillow，而 Pillow 未声明 |
| `tests/` | pytest 套件 | `test_settings.py` 一个文件占 98 个用例 |
| `code_optimization/` | 行人仿真向量化基准与报告 | 跑基准会重写入库的 `benchmark_results.json` |
| `legacy/ui/` | 迁移前的 Streamlit 原型 | 不参与测试，也不在 pyflakes 门禁里 |
| `third_party/` | ml-agents 与 MARL 的上游镜像 | gitignore，本机参考用，不属本仓文档范围 |
| `docs/` | 上手手册 `docs/getting-started.md` | 架构与约定已并入本文件 |

`unity/README.md`（Unity 接入）与 `code_optimization/report.md`（性能基准）是各自主题的深度文档，
保留独立文件；其余总览性内容不要在三处复述。

## 关键约定（违反会出问题的才写）

- **配置加载优先级：shell 环境变量 > `.env` > 代码默认值。** `config._load_dotenv()` 默认
  `override=False`，已存在的环境变量不被 `.env` 覆盖。原因：服务常在 systemd/容器里由环境注入密钥。
  违反后果：`.env` 写了值但不生效，而 `doctor` 显示的是生效值，看起来像"配置丢了"。
- **`reload_env()` 是唯一用 `override=True` 的入口**，只被 `settings.write_settings()` 调用，
  即 `POST /api/settings` 保存之后。它只刷新 `.env` 里出现的键，不碰其他环境变量。
  这构成一个真实的例外：**在「设置」视图里保存过的项以 `.env` 为准**，会盖掉同名 shell 变量。
  所以不要把只存在于 shell 的密钥在设置页清空——清空会写 `KEY=` 并立即生效，等效于抹掉这把密钥。
- **写回 `.env` 的五道闸**：① 只接受 `settings.SETTINGS` 注册表内的 7 个键，未注册键整批拒绝；
  ② 值里含换行、回车、NUL 或 `=` 一律拒绝（防注入出第二个配置项）；③ 超过 400 字符拒绝、
  `numeric` 项要有限正数；④ 先备份 `.env.bak`，再写同目录临时文件 `os.replace` 原子替换。
  ⑤ `POST /api/settings` 只认 socket 的 `remote_addr` 是否 loopback，**不信 `X-Forwarded-For`**。
  这些都有用例钉着（`tests/test_settings.py`）。
- **对外一律掩码。** `settings.mask_value()` 只给前 2 字符与长度，长度 ≤4 时连前缀都不给。
  任何新接口、CLI 输出、日志都不得回显明文。`GET /api/settings` 因此对局域网只读开放，
  写接口才有 loopback 限制。
- **路径只由 `config.py` 推导。** 所有产物走 `config.paths`，锚定 workspace 根
  （`ZHISHUXING_WORKSPACE` 可覆盖，否则从包源码位置向上找带 `configs/` 的目录）。
  新代码里出现 `Path("data/...")` 这种相对 CWD 的写法就是 bug：从别的目录启动会写错地方。
  副作用：`web/mobile/` 也从 workspace 取，改 workspace 会让 `/mobile` 变 404。
- **降级逻辑住在各适配器里，`settings.py` 只报状态不改变行为。** 留空即：LLM → `MockLLMAdapter`
  确定性模板；底图 → Canvas 折线；规划 → 内置枢纽引擎；策略 → `HeuristicPolicy`。
  新增在线能力时要同时提供"没有它也能跑"的路径，并把 `degrades_to` 写进注册表。
- **`analysis/` 是唯一来源。** 中文字体配置、`moving_average`、CSV 读写、合成数据生成器历史上
  散落 6 份以上，已归一到 `analysis/{plotting,io_utils,synthetic}.py`。别在 `core/` 或 `webapp/`
  里再写一份字体设置——那会让无头环境重新出现方框字。
- **`matplotlib.use("Agg")` 在 `analysis/plotting.py` 顶层。** 任何要绘图的模块都必须先 import 它
  再 pyplot，否则无头环境会挂起。
- **产物命名不能改。** `{algorithm}_env_{env_name}_number_{n}_seed_{s}.npy` 与
  `{algorithm}_actor_number_{n}_step_{k}k_agent_{id}.pth` 是读方（`runtime.py` 的
  `CHECKPOINT_PATTERN`、`reports.py` 的 `*_env_*.npy` glob）与写方（`runner.py`）之间的契约，
  改了命名等于让既有训练产物失效。
- **`legacy/` 与 `third_party/` 不参与测试。** 前者是迁移前的 Streamlit 原型，依赖（streamlit）
  根本没声明，且与 `planning/amap.py` 有四个同名函数，收进测试会出现"测的是哪一份"的歧义；
  后者是上游镜像，里面自带几百份 README，其相对链接不由本仓维护。两者都在 `.docsignore` /
  门禁命令之外。
- **不要在本仓写真实密钥。** `config.py` 里所有密钥取值都没有默认值，空串按未配置处理
  （`_env_or_default` 用 `or` 而不是 `get(k, default)`，就是为了把设置页写入的 `KEY=` 也当未配置）。

## 改动后的验证

| 你动了 | 必须跑 |
|---|---|
| `src/zhishuxing/**` | `python -m pytest` + `python -m pyflakes src/ scripts/ tests/` |
| `webapp/app.py` 的端点或返回结构 | `zhishuxing smoke` + `python -m pytest tests/test_api.py`，并同步 `static/app.js` |
| `settings.py` / `config.py` | `python -m pytest tests/test_settings.py` + `zhishuxing doctor` |
| `core/navigation.py` 的规划或代价 | `python -m pytest tests/test_navigation.py tests/test_navigation_prefs.py tests/test_simulation.py` |
| `configs/hub_default.json` | `zhishuxing smoke`（它真的会规划一次 `[1,2]→[28,12]`） |
| `data/transfer_kb/shenzhen_north/` | `zhishuxing kb-ingest --query "带老人 优先直梯"` 再 `python -m pytest tests/test_kb.py` |
| `analysis/**` 或报告 | `zhishuxing analyze --report all`（7 行 `OK` 才算过） |
| `core/animation.py` 的力场或放行窗口 | `python code_optimization/benchmark_animation.py`，并更新 `report.md`（注意它会改写入库 JSON） |
| `unity/HubTransferAgent.cs` | 无自动化：按 `unity/README.md` 的观测/动作契约与 `rl/envs.py` 手工核对 |
| 任何一级目录结构 | `python check_docs.py zhishuxing`（在 `Project/文档标准/` 目录执行） |

改完 `pyproject.toml` 或 `config.py` 后要重装一次 `pip install -e .`，否则
`src/zhishuxing.egg-info/` 里的旧元数据会掩盖新增入口。

## 已知坑

- **Windows 控制台中文乱码**：默认 cp936，`zhishuxing doctor` 这类中文输出会花掉，
  用管道捕获时还会变成 `UnicodeDecodeError`。先 `set PYTHONIOENCODING=utf-8`（PowerShell 用
  `$env:PYTHONIOENCODING="utf-8"`）。
- **`python -m pytest -q` 看不到统计行**：`pyproject.toml` 里已有 `addopts = "-q"`，命令行再给 `-q`
  就成了 `-qq`，末行 `180 passed` 被吞。要计数就用 `python -m pytest`，或用
  `python -m pytest -o addopts="" --collect-only -q`。
- **`src/zhishuxing.egg-info/` 是生成物**：`pip install -e .` 写出来的，已被 `*.egg-info/` 忽略。
  它会让"搜 `zhishuxing` 全仓"的结果多出一份带旧 README 的副本，别拿它当事实来源。
- **`.gitignore` 的 `.env.*` 会吃掉 `.env.example`**：第 51 行的 `.env.*` 匹配到它，
  靠第 53 行的 `!.env.example` 例外救回来（例外必须在后，gitignore 是最后匹配的规则赢）。
  当前状态：`.env.example` **已被跟踪**（复核：`git ls-files .env.example` 有输出、
  `git check-ignore -v .env.example` 无输出）。谁把这两行调换顺序或删掉例外，
  `.env.example` 就会静默变成不入库，新 clone 的人不知道要配哪些项。
  同理 `.env.bak`（设置页写的备份）是被忽略的，属正常。
- **`scripts/render_brand_assets.py` 需要未声明的 Pillow**：`pyproject.toml` 与 `requirements.txt`
  都没有 pillow，干净环境跑它要先 `pip install pillow`。本机装了所以看不出来。
- **`serve` 默认监听 `0.0.0.0`**，而开发模式打印的是 `http://127.0.0.1:<port>`。
  密钥写接口在**开发模式下不受 `--production` 那道开关限制**（`settings_writable` 仍为 True），
  挡在外面的只有每个请求的 loopback 校验。只想本机用就显式加 `--host 127.0.0.1`。
- **`zhishuxing train` 不需要 Unity 也会立刻失败**：`rl/envs.py` 顶层 import `mlagents_envs`，
  报 `ModuleNotFoundError: No module named 'mlagents_envs'`（退出码 1）。这不是配置问题，
  PyPI 上没有 1.x 版本，只能从源码装（`third_party/` 有镜像）。
- **测试与 smoke 会写 `data/outputs/`**：跑完 `git status` 仍是干净的是因为该目录未跟踪。
  不要为此加断言，也不要把某张图当"基线"提交进 `samples/`，除非 README 真要引用它。
- **对话会话与需求档案只存内存**（`TransferAssistant._sessions`），重启服务即清空。
  演示时别期待跨重启的多轮上下文。
- **全链路都是请求/响应，没有流式推送**：没有 SSE、没有 WebSocket，`/api/chat` 一次返回完整回答
  （复核：`src/zhishuxing/webapp/app.py` 全文无 `text/event-stream`，前端 `static/app.js` 无 `EventSource`）。

## 迁移映射（旧 → 新）

v2 重构是把历史散装脚本平移进包，不是重写。找不到"以前那段在哪"时查这张表。
左列是重构前的路径，**仓库里已不存在**（本机无从复核，仅作迁移记录保留）；右列全部可点开验证。

| 旧位置 | 新位置 |
|---|---|
| `program/MADDPG/MADDPG_main.py` | `rl/runner.py` + `cli.py train` |
| `program/MADDPG/{maddpg,matd3}.py` | `rl/agents.py`（TD3 三技巧为独立类路径） |
| `program/MADDPG/{networks,replay_buffer,environment}.py` | `rl/{networks,buffer,envs}.py` |
| `program/MADDPG/zhishuxing/rl_bridge.py` | `rl/runtime.py` + `core/simulation.py` |
| `program/MADDPG/zhishuxing/{navigation,system,visualization}.py` | `core/{navigation,system}.py` + `analysis/plotting.py` |
| `program/MADDPG/zhishuxing/adapters.py` | `llm/adapters.py`（新增 SiliconFlow 真实实现） |
| `program/MADDPG/webapp/*` | `src/zhishuxing/webapp/*`（新增 `/api/plan`、`/mobile`、`/api/settings`） |
| `program/MADDPG/plot_*.py`、`animate_transfer_env.py` | `analysis/{synthetic,reports}.py` + `core/animation.py` |
| `program/fine/*` | `analysis/reports.py`（效率对比）+ `analysis/synthetic.py`（微调指标） |
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
| `mlagents-envs>=1.1.0` 这个装不上的 pin、以及没用到的 `gym` 依赖 | 已从依赖里删除，改为源码安装说明 |
| MATD3 `save_model` 不建目录、ReplayBuffer 样本不足时报错含糊 | `rl/agents.py`、`rl/buffer.py` |
| 收集了偏好参数却把 `strategy` 写死为 0 | `planning/amap.py` 的 `resolve_strategy()` 真正映射 |
| 新版高德 v3 换乘响应改成嵌套结构后折线解析为空 | `_iter_segment_steps()` 兼容新旧两代，`tests/test_amap.py` 覆盖 |
| 地图容器零尺寸时创建高德地图得到空白 | 前端先显示容器再建图 |
| TensorBoard 日志目录多传一个 `args.algorithm`，导致与 npy 产物命名错位 | `rl/runner.py` 的 `log_dir` 已对齐产物命名 |
| 移动端 `sw.js` 预缓存引用不存在的 `VR1.png`，导致 SW 从未注册成功 | 缓存清单只有 9 个真实文件，版本号已到 `zhishuxing-mobile-v5` |
| 移动端引用未入库的 10 MB `AR.gif` 导致 clone 后破图 | `onerror` 回退占位说明 |
| 无 `.gitignore` 时 `node_modules`、`.pyc`、大媒体与密钥一起入库 | `.gitignore` 已治理。提交 `b62cc2c3` 里 `node_modules` 有 44389 个文件、未压缩体积 524 MB（复核：`git ls-tree -r --long b62cc2c3` 按路径前缀累加第 4 列）；**git 历史未重写**，见下面「不要做的事」 |
| manifest 主题色与页面 meta 不一致 | 两边都是 `#4f46e5` |
| 全仓 pyflakes 告警（未定义名、未使用导入、死变量） | 已清零并写进 CI |

## 不要做的事

- 不要把 `data/samples/` 当运行时目录去改：它是 README 展示图的来源，改图要重跑报告再拷回。
- 不要为了"看起来有 CI"给 `legacy/` 补测试或把它塞进 pyflakes；`third_party/` 也不要纳入文档检查，
  它已在 `.docsignore` 里，理由见上面的约定。
- 不要放宽 `POST /api/settings` 的 loopback 判定，也不要改成读 `X-Forwarded-For` 来判断来源。
  `--allow-remote-settings` 是给生产模式监听内网时留的手动逃生门，不是默认值。
- 不要在 `unity/` 里造 Unity 工程文件来"让它能构建"，也不要为它写 CI。
- 不要把 `MADDPG` 的效果写进任何数字结论：默认没有权重，`policy_source` 就是「启发式回退」；
  `data/outputs/` 里的分析图来自固定种子的合成数据。
- 不要动 git 历史来清理已泄漏的密钥（`git show 93656291:UI/config_direct.py` 里能 grep 到 3 处
  密钥形状字面量，历史未重写）。这属破坏性操作，且清史不能代替平台侧作废。
- 不要在文档里另写一套测试数或端点清单，指回本文件的表格。
