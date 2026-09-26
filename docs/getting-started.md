# 智枢星 上手手册

> 用途：给要真的把它用起来或改它的人。每一步给命令、给预期输出、给出错时怎么办。
> 按顺序往下做即可，不需要跳读。行业词第一次出现都有一句解释（第 9 节汇总）。
> 测试数、命令与门禁的权威口径在仓库根的 [AGENTS.md](../AGENTS.md)，本文件只链接不复述。

## 1. 你需要准备什么

| 项目 | 要求 | 怎么确认 |
|---|---|---|
| 操作系统 | Windows / Linux / macOS 均可（本手册的输出在 Windows + Python 3.12.0 上采集） | `python --version` |
| Python | 3.10 以上；CI 只跑 3.11 与 3.12 | `python --version` |
| 必需依赖 | `numpy`、`matplotlib`、`flask`、`waitress`、`requests` 这 5 个 | `python -c "import importlib.metadata as m;print(m.requires('zhishuxing'))"` 的前 5 项 |
| 可选依赖 | `torch`（加载训练权重、跑 RL 测试）、`openai`（真实 LLM 适配器）、`tensorboard`（训练日志）、`pytest`（测试） | 见第 2 节的 extras |
| 训练专用 | `mlagents_envs`，PyPI 上没有 1.x 版本，只能从源码装 | 缺失只影响 `zhishuxing train`，见第 6 节 |
| 网络 | 不配密钥、不训练时**完全不需要**：全链路都有离线降级 | `zhishuxing doctor` 看当前降级项 |
| 密钥 | 3 项必需（高德两把 Key + JS 安全密钥）+ 4 项可选，见第 3 节 | `zhishuxing doctor` |
| 浏览器 | 任意现代浏览器；控制台与移动端 PWA 都用同源 fetch | — |

Windows 上如果命令输出是乱码，先执行 `set PYTHONIOENCODING=utf-8`（PowerShell 用
`$env:PYTHONIOENCODING="utf-8"`），原因见第 8 节。

## 2. 装好它

```bash
git clone git@github.com:Luz7818/zhishuxing.git
cd zhishuxing
pip install -e .
```

预期输出的末行（重装时会先出现 `Uninstalling zhishuxing-2.1.0`）：

```
Successfully installed zhishuxing-2.1.0
```

`-e` 是 editable（源码改了立刻生效，不用重装）。控制台脚本入口注册为 `zhishuxing`，
定义在 `pyproject.toml` 的 `[project.scripts]`。

按用途追加可选组：

```bash
pip install -e .[dev]     # pytest
pip install -e .[llm]     # openai，真实 LLM 适配器要用
pip install -e .[train]   # torch + tensorboard，训练链路
```

确认装对了：

```bash
zhishuxing --help
```

看到 9 个子命令即正常（`demo` `train` `analyze` `simulate` `animate` `serve` `smoke` `doctor`
`kb-ingest`，逐个的作用见第 4 节）。

## 3. 五分钟看一遍

### 3.1 先体检配置

```bash
zhishuxing doctor
```

刚 clone 完、没有 `.env` 时输出长这样（末三行是关键）：

```
智枢星 · 配置体检（密钥一律掩码，不输出明文）
  工作区   : C:\...\zhishuxing
  配置文件 : C:\...\zhishuxing\.env（不存在，请复制 .env.example 为 .env）

  [未配置] AMAP_REST_KEY            留空即按默认值/降级运行
             用途    : 后端地理编码 + 公交换乘规划，决定 /api/plan 与 /api/chat 能否出真实路线。
             申请    : https://lbs.amap.com/ → 控制台 → 应用管理 → 添加 Key，服务平台选「Web服务」。
             填写    : C:\...\zhishuxing\.env 中的 AMAP_REST_KEY=（或控制台「设置」视图保存）
             当前降级: 真实路线规划不可用：引擎自动改用内置枢纽引擎，市际 OD 给出话术提示而不是报错。
  ...（其余 6 项同结构：已配置 / 未配置、用途、申请入口、填哪儿、当前降级成什么）

  在线能力 : 真实路线规划 降级为内置枢纽引擎 / 高德底图 未配置 JS Key，回退 Canvas 折线 / 真实 LLM 降级为 Mock 确定性模板
  离线演示 : 始终可用（Mock 对话 + Canvas 折线 + 内置枢纽引擎），与是否填写密钥无关
  结论     : 必需项缺失 3 项 → AMAP_REST_KEY, AMAP_JS_KEY, AMAP_SECURITY_CODE（离线演示不受影响，仅对应在线能力受限）
```

**退出码 1 是正常状态**，它只表示在线能力还没配齐，不表示装坏了。
三项都填好后同一命令退出码变 0；`zhishuxing doctor --strict` 会把 4 项可选项留空也算进非零退出。
已配置项的值只显示掩码（前 2 字符 + 长度），手册里不复现。

### 3.2 跑一次 API 冒烟

```bash
zhishuxing smoke
```

预期输出：

```
Web smoke test passed.
```

它在**当前进程内**用 Flask 的 test client 依次打 `/health`、`POST /api/navigation/load`、
`POST /api/navigation/plan`（`[1,2] → [28,12]` 途经 `security`）、`POST /api/dashboard/run`
并回取生成的图片 URL、`GET /api/rl/status`、`POST /api/rl/act`、`POST /api/rl/simulate`。
任何一步状态码不是 200 就 `AssertionError` 退出。这条命令不占端口、不需要浏览器，
改完后端后最先跑它。

### 3.3 起控制台

```bash
zhishuxing serve --host 127.0.0.1 --port 7860
```

预期输出：

```
开发模式启动: http://127.0.0.1:7860
 * Serving Flask app 'zhishuxing.webapp.app'
 * Debug mode: off
WARNING: This is a development server. Do not use a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:7860
Press CTRL+C to quit
```

浏览器打开 `http://127.0.0.1:7860`。左侧 7 个视图：概览、路线规划、枢纽导航、客流面板、
RL 智能体、分析报告、设置；右侧「智能换乘助手」是可展开的对话面板。
移动端 PWA 在 `http://127.0.0.1:7860/mobile`（同源托管，见第 7 节）。

**没有 `--host` 时默认监听 `0.0.0.0`**（`cli.py` 的 serve 参数默认值），
即使开发模式打印的是 `127.0.0.1`。只想本机用请显式带上 `--host 127.0.0.1`。

### 3.4 直接问一句

在这个服务上（保持终端别关，另开一个终端）：

```bash
python -c "import requests;print(requests.post('http://127.0.0.1:7860/api/chat',json={'message':'带老人行李多，优先直梯，去地铁前先上趟卫生间，从A口到地铁闸机'}).json()['data']['reply'])"
```

离线（无密钥）时看到的就是这份确定性模板输出：

```
已理解您的需求:途经卫生间、优先直梯、避开拥挤、行李较多、携老人。
枢纽内步行约 1110 米(38 格)
已按需求途经卫生间A(顺路)
路径经过设施:直梯×2
卫生间A:已纳入路线
提示:请跟随站内引导标识与电子屏指引前往目标检票口。
参考站内经验:《卫生间分布指南》、《老人出行建议》、《高铁到地铁快速换乘》。
(当前为本地模板模式;配置 SILICONFLOW_API_KEY 后可获得更自然的对话回答。)
```

这一条走完了助手的全部五步：口语 → 需求档案 → 地标解析与偏好加权 A* → BM25 经验检索 → 回答合成。
米数是 `configs/hub_default.json` 里 `cell_size_m = 30` 乘路径格数得来的示意值，不是实测步行距离。

## 4. 跑分析（7 类报告）

```bash
zhishuxing analyze --report all      # 7 类全跑
zhishuxing analyze --report reward   # 只跑一类
```

`--report` 的合法取值就是这 7 类加 `all`：

| 名称 | 出什么 | 产物（在 `data/outputs/`） |
|---|---|---|
| `reward` | MADDPG 评估奖励曲线（原始 + 平滑，y 轴归一化） | `reward_curve.png` |
| `heatmap` | 拥堵热力三联图、区域改善率排名、分时段削峰 | `shenzhen_north_congestion_heatmap.png`、`zone_improvement_ranking.png`、`peak_shaving_by_timeslot.png` |
| `transfer` | 换乘时间 P50/P90/Max 随训练迭代的分布 | `transfer_time_distribution.png`、`transfer_time_distribution_simulated.csv` |
| `queue` | 安检排队：无引导 vs MADDPG 动态分流，北区与南区 | `security_queue_comparison.png`、`security_queue_comparison.csv` |
| `efficiency` | 高峰/平峰 × 普通/大件行李四场景的训练前后对比 | `transfer_efficiency_scenario_comparison.png`、`.csv` |
| `finetune` | LLM 微调指标 loss / BLEU-4 / ROUGE-1 / ROUGE-L | `finetune_metrics_simulated.png`、`.csv` |
| `animation` | 枢纽换乘环境行人动图 | `transfer_env_demo.gif` |

全部成功时的输出（每类一行 `[名字] OK -> 文件列表`）：

```
[reward] OK -> ['...\data\outputs\reward_curve.png']
[heatmap] OK -> ['...shenzhen_north_congestion_heatmap.png', '...zone_improvement_ranking.png', '...peak_shaving_by_timeslot.png']
...
[animation] OK -> ['...transfer_env_demo.gif']
```

退出码：7 类都成功是 0，有失败是 1（失败行会打成 `[名字] FAILED: <原因>`）。

**这些数据是合成数据**，除 `reward` 之外。`reward` 读的是真实训练产物 `*_env_*.npy`：
先看 `data/outputs/`，没有就回退 `data/samples/`（那里留了 3 个种子各一份的样本）。
其余 6 类由 `analysis/synthetic.py` 按固定种子生成，改种子等于改图，别把它当实测结论引用。

`zhishuxing demo` 是另一条链路：导航 → 客流面板 → 模拟微调 → 汇总 JSON，加 `--reports` 时顺带跑上面 7 类。
`zhishuxing animate --frames 60 --fps 20 --n_agents 44` 单独出动图。
`zhishuxing simulate` 跑多智能体引导仿真并出轨迹图，没有权重时输出里 `policy_source` 是「启发式回退」。

## 5. 配密钥

只有想让"真实路线 + 真实底图 + 真实对话"这三件事生效时才需要配。**不配也能完整演示。**

```bash
cp .env.example .env
```

`.env.example` 里 7 个变量各自的作用与不填的后果（值一律留空，本手册不写任何密钥）：

| 变量 | 作用 | 不填会怎样 |
|---|---|---|
| `AMAP_REST_KEY` | 后端调高德地理编码与公交换乘，决定 `POST /api/plan`（`engine=amap`）与 `/api/chat` 的市际 OD 能否出真实路线 | `engine=amap` 返回 400 并说明原因；`/api/chat` 给出话术提示并自动改用内置枢纽引擎，演示不中断 |
| `AMAP_JS_KEY` | 浏览器侧高德 JS API 的真实地图底图（前端注入项） | 回退 Canvas 离线折线示意，说明栏标注「未配置 AMAP_JS_KEY」 |
| `AMAP_SECURITY_CODE` | 与上面 JS Key 配对的安全密钥，**必须一起填** | 2021-12 之后申请的 JS Key 缺它初始化不出来，表现是「高德地图初始化失败」再回退 Canvas——只填 JS Key 等于没填 |
| `SILICONFLOW_API_KEY` | LLM 对话合成 / 需求档案解析 / OD 提取 | 走 `MockLLMAdapter` 确定性模板；规则关键词解析仍生效，全链路离线可演示 |
| `SILICONFLOW_BASE_URL` | OpenAI 兼容端点地址，可指向自建 vLLM / Ollama | 用内置默认端点（SiliconFlow 官方 `/v1`） |
| `SILICONFLOW_MODEL` | 默认模型 ID | 用内置默认模型名；控制台对话面板里手填的模型 ID 优先级更高 |
| `AMAP_TIMEOUT` | 后端调高德的超时秒数 | 用默认 10 秒 |

三件事要知道：

1. **格式限制**：只支持 `KEY=VALUE` 与整行 `#` 注释两种行，值不要加引号、不要写行内 `#` 注释
   （解析器会把它们当值的一部分）。项目不依赖 `python-dotenv`。
2. **优先级**：服务启动时系统环境变量优先，`.env` 不覆盖已存在的同名项。
   例外是在控制台「设置」视图里保存过的项——那会以 `.env` 为准强制刷新。
   所以某把密钥只存在于 shell/systemd 而不在 `.env` 里时，**别在设置页把它清空**：
   清空会写 `KEY=` 并立即生效，等效于抹掉这把密钥。
3. **改完怎么生效**：设置页保存即热重载；手工编辑 `.env` 要重启服务。

高德的两把 Key 要**分别创建**：同一个应用下加两个 Key，服务平台分别选「Web服务」和
「Web端(JS API)」，两者不通用。个人开发者免费配额有限，规划一次行程会打
2 次地理编码 + 1 次公交换乘。

不想碰文件也可以：起服务后在控制台「设置」视图，或移动端 PWA 的「系统设置 → 服务与密钥」里填。
写接口只接受本机请求，生产模式监听非本机地址时默认整个关闭（`AGENTS.md` 的「关键约定」写了这五道闸）。

## 6. 跑 RL 训练

```bash
pip install -e .[train]
pip install ./third_party/ml-agents-develop/ml-agents-develop/ml-agents-envs
zhishuxing train --mlagents_file "D:\Builds\HubTransfer\HubTransfer.exe" --behavior_name HubAgent --episode_limit 200 --max_train_steps 500000 --evaluate_freq 5000
```

`--mlagents_file` 留空则连 Unity Editor（Play 模式）。`--algorithm MATD3` 换 TD3 变体。
不带任何参数时全部走 `configs/training.json` 的默认值（含 `max_train_steps = 1000000`，
**第一次跑请先用 `--max_train_steps` 压到几千步验证链路**）。

产物三份：`data/outputs/*.npy`（评估奖励）、`data/model/integrated_hub_transfer/*.pth`（actor 权重）、
`data/runs/`（TensorBoard）。Unity 侧要配什么、观测与动作怎么对齐，看
[unity/README.md](../unity/README.md)。**本仓库不带可构建的 Unity 工程**，那三个目录里也没有权重。

### 没有 Unity 时会怎样

`zhishuxing train` 直接失败，报错发生在导入环境封装的那一行：

```
File "...\src\zhishuxing\rl\envs.py", line 12, in <module>
    from mlagents_envs.base_env import ActionTuple
ModuleNotFoundError: No module named 'mlagents_envs'
```

退出码 1。这不是配置错误：PyPI 上没有 `mlagents-envs` 的 1.x 版本，只能从源码装。
装完还需要一个训练场景（Behavior 名 `HubAgent`、连续动作 2 维）。

### 不训练也能把 RL 这条链路演示完

三件事按顺序做，全部离线：

```bash
zhishuxing simulate --max_steps 120 --agents_per_group 3
```

输出是一份 JSON 摘要，节选：

```
{
  "policy_source": "启发式回退",
  "steps_executed": 55,
  "agents_total": 9,
  "agents_arrived": 9,
  "avg_transfer_steps": 39.33,
  "free_flow_baseline_steps": 37.0,
  "collision_events": 0,
  ...
}
```

`policy_source` 明确写了这次用的是启发式而不是 MADDPG 策略——没有权重时系统不会假装是模型在引导。
控制台「RL 智能体」视图的「加载最新策略权重」按钮走同一套判定：扫到权重就加载，
扫不到就在状态里注明「启发式回退」和原因。奖励曲线（`zhishuxing analyze --report reward`）
读 `data/samples/` 里那 3 个样本 npy 就能出图。

`rl/` 的 6 个测试用例需要 torch，没装时是 skip 不是 fail。

## 7. 知识库入库与移动端 PWA

### 7.1 换乘经验知识库

```bash
zhishuxing kb-ingest --query "带老人 优先直梯"
```

预期输出：

```
{
  "files": 22,
  "docs": 22,
  "changed": 0,
  "out": "...\data\transfer_kb\corpus.jsonl"
}
语料 22 条,检索自检「带老人 优先直梯」:
  [7.583] 老人出行建议 <- 21_老人出行建议.md
  [5.122] 带小孩出行建议 <- 20_带小孩出行建议.md
  [3.098] 直梯与无障碍电梯分布 <- 03_直梯与无障碍电梯分布.md
```

`changed: 0` 说明源文档与语料已一致——入库是幂等的，按 `hub:相对路径:段序号` 覆盖写。
`data/transfer_kb/shenzhen_north/` 下 22 篇 `.md` 是**按公开出行攻略与站方指引手工整理的演示语料**，
不是站内实测数据，站内布局以现场为准。

换枢纽或收新文档：把 `.txt` / `.md` / `.html` 放进一个目录，
`zhishuxing kb-ingest --src <目录> --hub <枢纽标识> --query "<试搜>"`。输出语料默认就是
`data/transfer_kb/corpus.jsonl`（`--out` 可改），不需要改代码。

### 7.2 移动端 PWA

服务起来之后访问 `http://127.0.0.1:7860/mobile`。它与控制台同源，所以不用管 CORS，
ServiceWorker 的 scope 也正好覆盖页面。

- 只调两个接口：`POST /api/chat`、`GET /api/settings`。
- 后端不可用时回退本地演示数据，状态栏写「演示数据」并提示「后端不可用，已回退演示数据」。
- 「AR 实景导航」的素材 `AR.gif`（约 10 MB）不随仓库分发，缺失时显示占位说明而不是破图。
- 桌面版 Chrome/Edge 地址栏右侧会出现安装图标；手机上用「添加到主屏」。
  非 HTTPS 地址下浏览器会限制 ServiceWorker，真机验证以局域网 HTTP 访问为准。

细节（含 `sw.js` 缓存版本号改了要升）在 [web/README.md](../web/README.md)。

## 8. 常见故障

| 现象 / 报错原文 | 原因 | 怎么办 |
|---|---|---|
| Windows 控制台中文输出是乱码；用管道捕获时变 `UnicodeDecodeError` | Python 按 GBK(cp936) 写控制台 | `set PYTHONIOENCODING=utf-8` 后重跑 |
| `zhishuxing doctor` 退出码 1 | 3 项必需密钥没配 | 正常，离线演示不受影响；要在线能力就按输出里的「申请 / 填写」两行配 |
| 启动时出现 `Tip: There are .env files present. Install python-dotenv to use them.` | 那是 Flask 自己的提示 | 忽略。`.env` 由 `config.py` 解析（不依赖 python-dotenv），已经生效 |
| `.env` 里填了值但行为没变 | 同名 shell 环境变量优先，`.env` 不覆盖它 | 换 shell 里 unset 掉，或在设置页保存一次（那一项会以 `.env` 为准） |
| 设置了页把某项清空后在线能力突然没了 | 清空写入 `KEY=` 并热重载，等效抹掉该进程内这把密钥 | 重新填值，或从 `.env.bak`（每次写入前的备份）恢复后重启 |
| `高德地图初始化失败`，页面回退成折线 | 填了 `AMAP_JS_KEY` 但没填 `AMAP_SECURITY_CODE` | 两个必须成对，且服务平台类型要选对（Web端 JS API） |
| `/api/plan` 返回 400 `未配置 AMAP_REST_KEY 环境变量，无法使用真实路线规划。` | `engine=amap` 需要 Web 服务 Key | 配 `AMAP_REST_KEY`，或改用 `engine=hub`（内置枢纽引擎，离线可用） |
| `ModuleNotFoundError: No module named 'mlagents_envs'`（跑 `train` 时） | PyPI 无 1.x 版本 | 从源码装：`pip install ./third_party/ml-agents-develop/ml-agents-develop/ml-agents-envs` |
| `zhishuxing analyze --report reward` 报 `未找到 *_env_*.npy 奖励文件` | `data/outputs/` 与 `data/samples/` 里都没有奖励数组 | 正常 clone 时应能回退到 `samples/` 的 3 个样本；被删了就从备份或重跑训练取回 |
| `[reward] FAILED`、`[animation] FAILED` 之类出现在 analyze 输出里 | 单个报告异常被吞住继续跑，退出码最后统一给 1 | 看该行括号里的原因；多数是 `data/outputs/` 目录权限或 matplotlib 字体问题 |
| 打开 `/mobile` 返回 404 | 设了 `ZHISHUXING_WORKSPACE` 指向别的目录，那边没有 `web/mobile/` | 回仓库根起服务，或把 `web/` 一并拷过去 |
| 页面显示的路径/米数和想象差很多 | `cell_size_m = 30` 是示意换算系数，网格也是示意底图 | 换枢纽要同时改 `configs/hub_default.json` 的网格与地标，米数才有意义 |
| `python -m pytest -q` 跑完不显示通过数 | `pyproject.toml` 已有 `addopts = "-q"`，再 `-q` 就成了 `-qq` | 用 `python -m pytest`（不带额外 `-q`） |
| 起服务后局域网机器能访问 | `serve` 默认 `--host 0.0.0.0` | 显式 `--host 127.0.0.1`；写密钥接口本身还有一道 loopback 校验 |

## 9. 术语小词典

| 词 | 在这里指什么 |
|---|---|
| MADDPG | 多智能体深度确定性策略梯度：每个智能体一个自己的 actor（只看自己的局部观测）、训练期共用一个集中式 critic（看所有人的观测与动作）。本项目的 2 维连续动作是 `[前进/后退, 转向]`，转向按整档 90° 量化 |
| MATD3 | MADDPG 的 TD3 变体（双 critic 取小、目标策略平滑、延迟更新策略），`--algorithm MATD3` 启用，三个技巧在 `rl/agents.py` 里是独立类路径 |
| 策略（policy） | actor 网络从观测到动作的映射，也就是训练产物 `*_actor_*.pth` 里存的那个东西。运行时按 agent 各取 step 最大的一个加载 |
| 启发式回退 | 没加载到权重时用的规则策略：朝目标方向对齐、默认前进。它输出与 actor 同样的 2 维动作语义，所以链路不断；结果里 `policy_source` 会如实写「启发式回退」 |
| 奖励曲线 | 训练过程中定期**无噪声评估**得到的平均回报序列，存成 `{算法}_env_{环境名}_number_{n}_seed_{s}.npy`。`analyze --report reward` 画的就是它，默认还把 y 轴归一化到 [0,1] |
| 局部观测 | 单个智能体看得到的信息：目标相对位置、自身速度、最近 3 个邻近行人的相对位置（`OBS_DIM = 4 + 2×3 = 10`）。集中式 critic 才看全局 |
| BM25 | 知识库的检索打分函数：词频 + 逆文档频率 + 长度归一。中文按 2-gram 切、英文按小写单词切，标题重复一遍来加权。实现在 `llm/kb.py`，零第三方依赖，几十~几千条量级下毫秒返回 |
| 需求档案（`PassengerProfile`） | 把口语需求解析成结构化字段：4 维优先级 + 硬约束 + 软偏好 + 5 类乘客画像。画像是 `行李 / 老人 / 儿童 / 轮椅 / 赶时间`，每项会派生出隐含的偏好或约束 |
| 偏好感知 A\* | 在普通 A\* 上给每格加代价：进楼梯加价、带轮椅则楼梯禁行、要求「途经卫生间」时把它当软必经点（顺路才纳入，绕行超过 `soft_via_tolerance` 倍代价就如实说明没纳入） |
| 软必经点 | 不强制走、但值得顺路走的设施点。与 `via_landmarks` 那种硬必经（如「必经安检」）相对 |
| 高德路径规划 | 后端实际调的是两个 v3 接口：地理编码（地址→经纬度）与 `direction/transit/integrated` 公交换乘。「少步行」不是单独的步行规划接口，它是公交换乘 `strategy=3`（最少步行优先）这个参数 |
| waypoint | 仿真里智能体沿规划路径当前要抵达的那个路径点：`_SimAgent.waypoint` 是路径下标，走到该点就前进到下标，走完最后一个记为到达 |
| 动态客流 | 按各组乘客的释放时间沿路径撒点并随机扩散得到的归一化拥堵矩阵（`core/flow.py`），面板图与引导文案都读它 |
| editable 安装 | `pip install -e .`：装的是指向源码的链接，改 `src/` 立刻生效 |
| PWA | 可安装到桌面/主屏的网页应用，这里是 `web/mobile/mobile_app.html` + `manifest.webmanifest` + `sw.js`（ServiceWorker 预缓存） |
| 降级 | 在线能力缺依赖时的确定性替代：Mock 对话、Canvas 折线、内置枢纽引擎、启发式策略。四者都不需要网络 |

---

改完之后要跑什么，见 [AGENTS.md](../AGENTS.md) 的「改动后的验证」；
各目录各管哪一段，见该目录下的 `README.md`。
