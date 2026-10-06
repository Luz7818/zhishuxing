# HISTORY —— 版本更新记录

> 用途：记录 zhishuxing 的版本演进。只追加，禁止删除或改写既有条目；写错了就追加一条更正。
> 本文件从 v2.2.0 起向前记录（更早版本不追溯补写，历史见 git 提交记录）。
> 版本遵循语义化版本：破坏契约 = 主版本，加能力 = 次版本，修缺陷 = 补丁。
> v2.3.0 及以前的条目由原 `CHANGELOG.md` 逐条无损转入（2026-10-05），内容未改写。

## 2026-09-29 · v2.2.0 安全收口与 Web 控制台重构

### Security(安全)

- **git 历史重写收口(P0)**:剥离 `UI/`、`picture/`、`ml-agents-release_18_branch/`、
  `node_modules/`、`__pycache__/` 历史路径并替换全部密钥字面量
  (高德 Key 已轮换、DeepSeek Key 经平台侧实测 401 确认作废)。
  `.git` 体积 214 MB → 约 18 MB;全历史字节级检索密钥字面量 0 命中
- **`serve` 默认监听 `127.0.0.1`**(原 `0.0.0.0`):公网部署显式传 `--host 0.0.0.0`
- **密钥写接口按监听地址裁决**:监听非回环地址时默认关闭(须显式 `--allow-remote-settings`),
  不再因"开发模式"而放行;请求侧 loopback 校验保留,双层防线

### Added(新增)

- `LICENSE`(Proprietary 全文):教学/科研内部使用,第三方需书面授权
- Pillow 声明进 `[dev]` 依赖组(`scripts/render_brand_assets.py` 此前为未声明依赖)
- Web 控制台 v3.1 视觉精修:主操作渐变语言统一、卡片/KPI 悬浮质感、主区极光底光
- LLM 对话失败可见化:真实适配器异常记入会话并在响应中呈现(`real_adapter_error`,
  与 `load_llm` 同口径),不再静默吞掉

### Changed(变更 · 前端布局重构,同日追加)

- **智能换乘助手移到页面底部 Dock**(仿主流大模型对话形态):输入条常驻底部,
  发送或点按左侧圆形按钮即展开对话面板;原右侧竖轨 + 侧滑面板移除。
  面板展开后为**单列对话布局**(消息流为主体,分析详情限高内滚),
  对话引擎与微调实验收进底部「对话引擎 · 模型微调」折叠区;
  **顶边手柄可拖拽调整面板高度**(240px~78vh,记忆在浏览器,双击恢复默认)
- **侧栏可收起**:顶栏按钮把侧栏收成 64px 图标栏(状态记忆在浏览器),
  再点恢复;窄屏(<960px)下不收起
- **主栏排版精修(重构)**:RL/客流/报告/设置统一两列节奏(卡片等高),
  路线规划空状态入卡;**枢纽导航页重排(地图为中心)** —— 网格画布整幅居中放大(点击坐标
  按缩放比换算,交互不受影响)、图例居中;「规划参数」改为画布左上工具条
  按钮弹出的居中弹窗(规划成功自动收起),「规划结果」与网格画布**并排各占真空间**(工具条移到画布上方,
  布局留位零遮挡;窄屏上下堆叠),未规划时显示操作提示、规划后展示
  路径长度/途经点/完整坐标;
  移除仅此页使用的 `layout-split` 样式;修复 Dock「✕ 关不掉」—— `.chat-dock-body` 的
  `display:flex` 覆盖了 `hidden` 属性的浏览器默认样式,现显式声明
  `.chat-dock-body[hidden]{display:none}`;顺带修复重构引入的
  引导仿真卡片缺开标签与重复 `</section>`(全标签平衡校验通过)

### Added(新增 · 2.2 真实数据通路)

- **`data/real/` 真实客流数据约定目录**:`zhishuxing analyze` 自动发现 ——
  `congestion_before/after.csv` 供 heatmap、`transfer_summary.csv`(≥30 行)供
  transfer 与 efficiency;文件齐备即改用真实数据,输出行与图题标注
  `来源:真实数据:<文件名>`,缺文件自动回退合成并标注 `来源:合成(固定种子)`
- 格式模板与说明:`data/real/README.md` + 3 个 `*.example.csv`
  (真实 CSV 已进 `.gitignore` 不入库,防误提交敏感数据;模板入库)
- `run_congestion_report` / `run_transfer_time_report` 新增 `source_note` 参数

### Added(新增 · 3.0 起步与定位声明)

- **`unity/` 成为 UPM 本地包**(`package.json` + `HubTransferAgent.asmdef`,
  依赖 `com.unity.ml-agents@2.2.1-exp.1`):Package Manager → Add package from disk
  即可导入,脚本不再手工拷贝;仍不是可独立打开的工程(场景需自建,边界写进 AGENTS)
- **README「定位与边界」声明**:演示/教学工程 —— 会话仅内存、无账号、单实例、
  默认启发式回退与合成数据;走向「可用」的前置是会话持久化与多实例方案
- `render_reward_curve` 支持复用调用方已加载的奖励序列,`/api/rl/rewards`
  从两次全量 npy 加载降为一次

### Added(新增 · 智能体动作协议,同日追加)

- **类 MCP 工具调用**:对话助手可触发界面动作 —— 新增 `llm/actions.py`
  动作注册表(白名单:switch_tab × 7 个看板板块)+ JSON 动作信封解析
  (`{"say","action"}`,非法信封按原文返回、白名单外 tab 不下发)
- 双通道触发:① **规则短路** —— "打开/切换 + 板块名" 明确指令不经 LLM
  直接下发动作(离线确定、零延迟);② **LLM 信封** —— 自然语言泛化请求
  由真实模型按注入合成提示词的工具说明输出信封,服务端校验后下发
- 前端 `sendChat` 按动作执行 `switchTab` 并 toast;移动端 PWA 因 payload
  新增可选字段天然兼容(profile 容错已具备)
- 新增 9 个测试(动作协议 7 + API 快路径 2),全量 **193 passed**

### Added(新增 · 2.3 落地基建)

- **会话持久化(SQLite,零依赖)**:新增 `llm/session_store.py` —— 对话历史/需求档案
  写穿到 `data/runs/sessions.db`,服务重启后按 session_id 惰性回填;30 天不活跃自动清理
  (启动时 purge);档案序列化往返兼容、损坏记录按"无档案"处理;**持久化失败不阻断对话**
  (save/load_missing/delete 三处兜底,失败仅打印警告)
- **Docker 单容器部署**:`Dockerfile`(python:3.12-slim、非 root UID 1000、
  `/health` HEALTHCHECK、waitress 生产托管)+ `docker-compose.yml`(三数据卷:
  outputs/model/runs,端口只绑 127.0.0.1 交给反代)+ `.dockerignore`;
  DEPLOY.md 重写为 Docker 优先路线(本机无 Docker,镜像内契约已做静态校验,
  首次 `docker compose up --build` 请按 DEPLOY.md 验证)

### Fixed(修复 · LLM 自动挂载,同日追加)

- **密钥已配置即自动挂载真实适配器**:`zhishuxing serve` 启动时检测
  `SILICONFLOW_API_KEY` 并自动 `load_llm(prefer_real=True)`(模型留空回退
  `SILICONFLOW_MODEL`),不再要求每次重启后手动到对话面板点「加载模型」——
  此前密钥明明在、聊天却一直本地模板。自动挂载放在 serve 路径而非
  `create_app`:保证测试与 smoke 全程离线确定性(真实 `.env` 密钥不会在
  测试里发起网络调用)
- **适配器 HTTP 客户端直连化 + 失败重建重试**:`OpenAI(http_client=…,
  trust_env=False)` 忽略环境代理快照(校园端点直连可达,走代理出口会被
  网关按来源 IP 拦截);调用失败时重建客户端重试一次 —— 修复长驻服务进程
  反复拿到网关拦截页、而新起进程正常的环境态问题(实测同机同参:
  旧进程失败/新进程成功)
- **看板引导文案补降级路径**:`generate_guidance_text` 直接调 `llm.infer`
  无任何兜底,真实适配器 + 网络抖动时看板直接 500 —— 现失败即降级为
  确定性规则文案并打印原因(与 assistant 模板降级同口径,对齐
  「LLM 只有增强、必须可降级」约定)
- 模板回复尾注改写:两种 Mock 原因(未配置密钥 / 真实调用失败)都说清,
  指向 `real_adapter_error`

### Fixed(修复 · LLM 默认值清理,同日追加)

- 清理六处写死的 DeepSeek 旧默认:config 内置默认模型、设置页"留空降级"文案、
  Dock 模型输入框预填值(改为留空 = 服务器默认 `SILICONFLOW_MODEL`)、
  微调接口示例模型、微调模拟图标题、`.env.example` 注释 —— 全部统一为
  服务器配置的 `qwen3.8-27b`;设置页当前值本就正确(掩码 qw***)。

### Fixed(修复)

- `zhishuxing train` 在未装 `mlagents_envs` 的机器上给出带安装指引的报错
  (懒加载替代顶层 import;PyPI 无对应版本,需源码安装 `third_party/` 镜像)
- 启发式朝向策略双实现合并:`core/simulation.heuristic_action` 单一来源,
  `rl/runtime` 与仿真回退共用(此前逐行重复两份,改一漏一)
- `webapp.app` 模块级 `app` 改为 PEP 562 惰性构建:import 模块不再付出
  "加载导航图 + 扫描模型 + 建 BM25" 的全系统代价

### Changed(变更)

- 本地 LLM 端点配置示例:`SILICONFLOW_BASE_URL` 支持任意 OpenAI 兼容服务
  (当前指向 SEU openapi / qwen3.8-27b);高德 Key 与 API Key 均为部署者自填配置

## 2026-09-29 · v2.3.0 版本收纳

(本节收纳 2.2.0 tag 之后、原 CHANGELOG 创建时点之前落在 main 的全部变更:
LLM 自动挂载/直连化、动作协议、Dock 布局重构、导航页地图中心重排,以及 2.3 落地基建
——这些内容以 Added · 2.3 落地基建等小节形式并入了上一节,此处保留原说明。
更早内容见 v2.2.0 节与 git 历史。)

## 2026-10-05 · 文档规范体系落位（原 [Unreleased] 收纳）

- **移动端 PWA 对齐乘客侧功能 + 全面美化**:`web/mobile/mobile_app.html` 重写——新增真对话流
  (气泡消息、输入中动效,消费后端一直返回但此前被丢弃的 `kb_refs` 经验引用/`action` 动作信封/
  `real_adapter_error`,`POST /api/chat/reset` 新会话);结构化路线(步骤卡+沿途设施徽章+里程 KPI,
  替换 `<br>` 拼文本);站内导航页(13 地标选起终点/必经点,`/api/navigation/grid|plan` Canvas
  动画描线,演示场景一键填充);客流提示卡(`/api/dashboard/run` 三档拥挤度,失败静默收起)。
  免登录直达(移除纯前端演示登录);设计令牌与控制台同源(靛蓝紫渐变/阴影三档/缓动两族)+
  iOS 质感(毛玻璃顶栏底栏、按下即反馈、页面入场过渡、骨架屏、`prefers-reduced-motion` 降级),
  emoji 图标全部换内联 SVG;深色模式三态(浅/深/跟随系统,`data-theme` 对齐控制台机制,
  兼容旧 `darkMode` 键);补换乘方式偏好 UI,清理 quick-tag 与 aiConfig 死代码;SW 缓存 v5 → v7。
  质感与切换打磨:方向感知的双页滑动过渡(旧页滑出/新页滑入,返回保留滚动位)、底部导航滑动
  高亮药丸(弹簧曲线)、主题切换 View Transition 圆形揭示(不支持时退化为颜色交叉过渡)、
  顶栏滚动感知投影、卡片内高光、KPI/状态数值入场动效、导航画线 easeOutCubic、启动屏分层入场、
  hub 路线 KPI 智能收纳为里程+步数(不再以「-」占位)。
- **Windows 启动器** `启动器.bat`(仓库根):双击即用——7860 端口未监听时最小化窗口拉起
  `zhishuxing serve`,就绪后自动打开浏览器;已在运行则直接开浏览器,不重复起服务。
- **修复**：悬浮面板收回悬浮泡的「白球无图标」——`[hidden]` 的 `display` 过渡(allow-discrete)
  留 ~260ms 离场窗口,480px 高的对话体仍占布局,把输入条连同 FAB 挤出 56px 圆窗;现在泡泡态
  强制 `.chat-dock-body` 即时 `display: none; transition: none`,落位脉冲 `bubbleSettle`
  补 0→40% 透明度渐入。静态资源版本号 v5.8 → v5.9。
- **变更**：界面精修一轮（主题切换走 View Transitions、顶部导航滑动指示条、toast 图标语义、
  图标库补 2 个符号）；首页禁用启发式缓存（`Cache-Control: no-cache`）。
- **文档迁移**：五件套 → 九件体系（新增 ARCHITECTURE/CODE-STYLE/TESTING/GIT 与 HISTORY/TODO，
  `docs/getting-started.md` 更名 `docs/GET-START.md`，AGENTS 重写为规范入口，仓库地图/
  数据流/关键约定/迁移映射/历史问题表移入 ARCHITECTURE，改动后的验证移入 TESTING）；
  `CHANGELOG.md` 并入本文件后删除；README 规模行测试数 180 → 201 对齐实测。
- 变更缘由：落位《项目整体规范.md》九件必建。

## 2026-10-06 · 文档全仓核查修复（只动文档与 CHANGELOG 兑现删除）

- **packaging/ 落说明**：新建 `packaging/README.md`（exe_entry.py / app.ico / _build_version.txt
  三件的职责与门禁盲区），`目录说明.md` 树、一级目录表、快捷路径补 `packaging/`；
  README/GIT/ARCHITECTURE 的目录表同步收录。
- **数字对齐实测**：tests/README 180→201（补 test_actions/test_real_data/test_session_store 三行，
  test_api 19→21）；src/README 33→35 个 .py、llm 5→7（补 actions.py/session_store.py 两行）、
  版本 2.1.0→2.3.0；web/README 1830→2058 行、磁盘 14→12 个文件（VR.png/图标.png 已删）、
  「三个不入库的资源」改「不入库的资源」、别动段 v5→v7；data/README 入库 42→46、
  子目录表补 `real/`；GET-START 安装样例 2.2.0→2.3.0。
- **AGENTS**：HTTP 路由复核命令从表格内含 `|` 的交替写法改为两条无交替 grep（原样粘贴会得 0）。
- **GET-START kb-ingest 示例改为可直接粘贴**（以演示语料自指，幂等报 changed: 0）。
- **CHANGELOG.md 兑现删除**：上节声明的「并入后删除」至此执行，内容已逐条在档（复核：
  2.3.0 收纳节含 SILICONFLOW_BASE_URL 等全部条目）。
