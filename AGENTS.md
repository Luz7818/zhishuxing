# AGENTS.md —— 项目协作与代码开发规范（唯一权威入口）

> 用途：给 AI 编码助手与所有开发者。这里是规范入口与索引：目标、原则、流程、模块规则、
> 维护矩阵、阅读清单都在这份文件里。细则一律链接到对应文件，冲突时以细则文件为准并回改本文件。
> README.md 与 docs/GET-START.md 里被引用的事实以本文件的「当前状态」为准，它们只链接不复述。

## 项目目标

- 定位：综合交通枢纽智慧换乘引导系统的演示工程：Python 包提供 CLI、Flask 控制台后端与移动端
  PWA 三个面，里面串着 MADDPG 强化学习、内置网格枢纽仿真、高德路线规划和一条 LLM 对话链路；
  所有在线依赖都配了离线降级，没有密钥没有 Unity 也能跑完整演示。
- 核心功能：口语需求 → 结构化档案 → 偏好感知规划 → 经验引用 → 解释性引导；9 个 CLI 子命令、
  7 类分析报告、Web 控制台与移动端 PWA。
- 技术栈：Python ≥3.10（numpy/matplotlib/flask/waitress/requests，可选 torch/openai）+
  Unity 侧 UPM 本地包 + Docker 部署（复核：`python -c "import zhishuxing;print(zhishuxing.__version__)"`）。
- 详情：[README.md](README.md)、[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 开发原则

1. 正确性优先。
2. 可维护性优先。
3. 代码简洁、项目简洁。
4. 小步迭代。
5. 单模块开发。
6. 每个改动必须有明确设计与验收标准。
7. 禁止一次生成整个项目。
8. 禁止跳步开发。

执行口径：先想后写（假设与歧义先挑明）；最简优先（不加没要求的功能与抽象——「逻辑平移不重
设计」，API 契约与产物命名保持历史形态）；外科手术式改动（不动无关代码，每行改动可追溯到
需求）；目标驱动（先有可验证判据再动手，宣称完成前先跑通 [docs/TESTING.md](docs/TESTING.md)
的门禁）。

## 开发流程

**分析 → 设计 → 实现 → 测试 → 文档更新 → Git提交 → 等待确认**。不得跳过任何阶段。

| 阶段 | 产出物 | 放行标准 |
|---|---|---|
| 分析 | 影响面清单（core/rl/llm/webapp/analysis 哪一侧，是否触及契约） | 影响面说全 |
| 设计 | 方案说明（契约与命名影响、回退方式） | 验收标准已定义；与更简方案比较过 |
| 实现 | 代码 | 只含设计内改动，符合 [docs/CODE-STYLE.md](docs/CODE-STYLE.md) |
| 测试 | 门禁结果 | [docs/TESTING.md](docs/TESTING.md) 全过 |
| 文档更新 | 受影响文档 diff | 维护矩阵逐项过完 |
| Git提交 | 提交 | 符合 [docs/GIT.md](docs/GIT.md)，一批一提交 |
| 等待确认 | —— | 等人确认后推送 |

## 模块开发规则

- 一个智能体一次只开发一个模块；模块完成后才能进入下一模块。
- 如需同时开发，使用多个子智能体，每个子智能体同样一次只开发一个模块。

模块完成标准（全部满足才算完成）：

1. 功能完成：达到 [TODO.md](TODO.md) 中该任务的验收标准。
2. 测试通过：符合 [docs/TESTING.md](docs/TESTING.md)。
3. 最简原则：代码和项目架构都保持最简洁，无冗余抽象与重复实现。
4. [TODO.md](TODO.md) 更新：勾选完成项、明确下一项。
5. [HISTORY.md](HISTORY.md) 追加变更记录（破坏契约升主版本，加能力升次版本，修缺陷升补丁）。
6. 受影响的 docs 更新（按需）。
7. [README.md](README.md) 更新（如有面向使用者的变化）。
8. Commit message 符合 [docs/GIT.md](docs/GIT.md)。

## 文档维护规则

| 事件 | 需更新 |
|---|---|
| 模块完成 | `TODO.md`、`HISTORY.md`、受影响 docs |
| 版本发布 | `HISTORY.md` 新条目 + `pyproject.toml` 版本号 + tag |
| 架构决策（契约、降级路径、目录/产物约定变化） | `docs/ARCHITECTURE.md` + `HISTORY.md` 记录缘由 |
| 命令/入口/端点变化 | `README.md` / `docs/GET-START.md` / 对应子目录 README |
| 增删一级或二级目录 | 仓根 `目录说明.md` + 本文件 |
| 测试数/端点数/密钥项变化 | 本文件「当前状态」 |
| 新对话/新任务开始 | 按下方阅读清单阅读 |

## 开发前阅读清单

每个新对话/新任务，按顺序阅读：

1. 本文件（`AGENTS.md`）
2. [TODO.md](TODO.md)
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（数据流、仓库地图、关键约定、迁移映射、
   历史问题守护表在这里）
4. [docs/GET-START.md](docs/GET-START.md)
5. [HISTORY.md](HISTORY.md)
6. 与任务相关的 [docs/CODE-STYLE.md](docs/CODE-STYLE.md)、[docs/TESTING.md](docs/TESTING.md)、[docs/GIT.md](docs/GIT.md)

阅读完成后**不要写代码**：先做架构评审，输出——项目理解 / 核心模块 / 模块依赖关系 / 潜在风险 /
建议优化项 / 推荐开发顺序 / 是否发现架构问题——然后等待确认。

## Git 索引

- Git 规范：[docs/GIT.md](docs/GIT.md)（真实数据与密钥的入库边界、历史重写收口记录、一批一提交）

## 当前状态

复核环境：Windows + Python 3.12.0（`python --version`），已 `pip install -e .`；
本机 `torch`、`openai`、`Pillow` 在装，`mlagents_envs` 没装（所以下面 `train` 那条会失败）。

| 项 | 值 | 复核命令 |
|---|---|---|
| 测试 | `201 passed`（本机实测 20–50 s 之间浮动，受机器负载影响，别当判据） | `python -m pytest` |
| 用例分布 | `201 tests collected` | `python -m pytest -o addopts="" --collect-only -q` |
| 静态检查 | `0 告警`，退出码 0 | `python -m pyflakes src/ scripts/ tests/` |
| API 冒烟 | `Web smoke test passed.`，退出码 0 | `zhishuxing smoke` |
| 配置体检 | 本机配好 3 项必需密钥 → 退出码 0；全新 clone 无 `.env` → 退出码 1 并列缺 3 项 | `zhishuxing doctor` |
| 7 类报告 | 全部 `OK`，退出码 0 | `zhishuxing analyze --report all` |
| HTTP 路由 | 24 个注册 / 23 条不同路径（`/api/chat` 与 `/api/settings` 各含 GET+POST） | `grep -c '@app.get(' src/zhishuxing/webapp/app.py` 与 `grep -c '@app.post(' src/zhishuxing/webapp/app.py`，两条之和应为 24 |
| CLI 子命令 | 9 个 | `grep -cE 'add_parser\("[a-z-]+"' src/zhishuxing/cli.py` |
| CI | 定义在 `.github/workflows/ci.yml`：ubuntu-latest × Python `3.11` / `3.12`，装 `.[dev]` + CPU 版 torch，先 pyflakes 再 pytest（`MPLBACKEND=Agg`）。**这里不写"最近一次是哪个提交"**——分支每推一次它就变，写进文档同一次提交里就作废了；当前分支 HEAD 的徽章为 `passing`（复核见右）。本机没有 `gh`，但徽章与 Actions 接口对**公开仓都免认证**；要提交号与耗时再用 `/actions/runs`（匿名限 60 次/小时/IP，别拿它轮询） | `python -c "import urllib.request as u;b=u.urlopen(u.Request('https://github.com/Luz7818/zhishuxing/workflows/CI/badge.svg',headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read().decode();print('passing' in b)"` 应为 `True`；步骤读 `.github/workflows/ci.yml` |
| 版本 | `2.4.0` | `python -c "import zhishuxing;print(zhishuxing.__version__)"`，另一份在 `pyproject.toml` 的 `project.version` |
| Python 要求 | `>=3.10`（CI 只跑 3.11/3.12） | `pyproject.toml` 的 `requires-python` |
| 许可证 | Proprietary，全文在根目录 LICENSE（教学/科研内部使用，第三方需书面授权） | `git ls-files "*LICENSE*"` 恰好 1 行 |
| 运行时依赖 | `numpy`、`matplotlib`、`flask`、`waitress`、`requests` | `pyproject.toml` 的 `dependencies` |
| 可选依赖 | `[train]` → torch + tensorboard；`[llm]` → openai；`[dev]` → pytest。`mlagents_envs` 不在任何一格里 | `python -c "import importlib.metadata as m;print(m.requires('zhishuxing'))"` |

## 已知坑（省下一次的调查时间）

- **Windows 控制台中文乱码**：默认 cp936，先 `set PYTHONIOENCODING=utf-8`（PowerShell 用
  `$env:PYTHONIOENCODING="utf-8"`）。
- **`python -m pytest -q` 看不到统计行**：`addopts` 已含 `-q`，再给 `-q` 成 `-qq`。要计数用
  `python -m pytest`。
- **`src/zhishuxing.egg-info/` 是生成物**：会让"搜 zhishuxing 全仓"多出一份带旧文档的副本，
  别拿它当事实来源。
- **`.gitignore` 的 `.env.*` 会吃掉 `.env.example`**：靠后面的 `!.env.example` 例外救回
  （gitignore 最后匹配赢；当前已被跟踪）。谁调换顺序或删掉例外，新 clone 的人就不知道要配哪些项。
  同理 `.env.bak` 被忽略，属正常。
- **`serve` 默认监听 `127.0.0.1`**：局域网/公网显式传 `--host 0.0.0.0`；密钥写接口按监听地址
  裁决，须显式 `--allow-remote-settings` 才开放（双层防线见 `docs/ARCHITECTURE.md` 关键约定）。
- **`zhishuxing train` 需要 mlagents_envs**（懒加载，未装时给安装指引）；PyPI 无 1.x 版本，
  只能从 ml-agents 官方仓库 release/18 分支源码装（`rl/envs.py` 报错给完整命令）；
  其余功能不受影响。本机曾有的 `third_party/` 镜像已于 2026-10-07 删除（代码零引用）。
- **测试与 smoke 会写 `data/outputs/`**（未跟踪）；不要把某张图当基线提交进 `samples/`。
- **对话会话已持久化**（2.3）：内存是第一读写层，SQLite 写穿到 `data/runs/sessions.db`，
  重启按 session_id 惰性回填，30 天不活跃自动清理。
- **全链路都是请求/响应，没有流式推送**：没有 SSE/WebSocket。
- 不要把 `data/samples/` 当运行时目录去改；不要给 `legacy/` 补测试或塞进 pyflakes；
  不要放宽 `POST /api/settings` 的 loopback 判定；不要给 `unity/` 造工程文件或写 CI；
  不要把 MADDPG 的效果写进任何数字结论（默认无权重，启发式回退）；不要在文档里另写一套
  测试数或端点清单（指回本文件）；git 历史已在 2.2.0 重写收口，不要为清密钥再重写。
