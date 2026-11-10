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

## 2026-10-06 · v2.4.0 桌面版交付（不走浏览器的原生窗口 + 便携 zip）

### Added(新增)

- **桌面模式交付物**:`packaging/desktop_entry.py` —— PyInstaller onedir 入口,双击 exe 弹出
  pywebview 原生应用窗口(Edge WebView2 渲染),不再打开浏览器;后台 daemon 线程复用
  `serve --production`(waitress),品牌 loading 页(侧栏同族渐变 + 主题靛蓝)随端口就绪
  `load_url` 切入 `/?desktop=1` 控制台;关窗即进程退出。
- **单实例互斥**:Windows 命名互斥体 `Local\zhishuxing-desktop-singleton`——重复双击不再起
  第二个后端,而是新开窗口连接既有服务(viewer 模式);后端归首实例进程所有。
- **WebView2 运行时预检**:winreg 只读查询 Evergreen 运行时,缺失时原生 MessageBox 给出
  官方下载指引(替代黑控制台);桌面模式的致命错误同样走原生对话框。
- **便携分发包**:`scripts/package_zip.py` —— onedir 产物 + `使用说明.txt`(快速开始/数据
  位置/离线说明/常见问题,UTF-8 BOM) + LICENSE 组装为
  `dist/zhishuxing-<版本>-win64-portable.zip`,解压即用、删除即卸载。

### Changed(变更)

- **packaging/ 引导重构**:workspace 选择、资源解压、端口探测、致命错误兜底抽出为共享
  `packaging/bootstrap.py`;`exe_entry.py` 改为引用,浏览器模式(onefile)行为不变,降级为
  兼容回退链路。
- **构建脚本** `scripts/build_exe.py` 增加 `--mode onedir|onefile`(默认 onedir 桌面模式):
  桌面模式入口换 `desktop_entry.py` 并 `--collect-all webview/clr_loader/pythonnet` +
  `--hidden-import clr`;onefile 模式保持原 onefile + `exe_entry.py`。
- **前端桌面壳适配**:`app.js` 识别 `?desktop=1`(由 desktop_entry 注入)——禁右键菜单、
  禁 Ctrl+滚轮页面缩放与 Ctrl +/-/0 快捷键,WebView2 窗口行为对齐原生应用。
- **版本** `2.3.0 → 2.4.0`;`pyproject.toml` 增 `desktop = ["pywebview>=5.1"]` 可选依赖组。
- **文档**:packaging/README 重写为双模式说明;README 交付章节、`目录说明.md`、本节同步。

### 验证

- WebView2 真窗口冒烟(`loaded` 事件 + JS 求值)通过;onedir exe 冷启动:端口 7860 就绪、
  loading 页切入控制台、WM_CLOSE 干净退出;pytest 全量回归见当次提交说明。

## 2026-10-07 · 代码简洁清理（桶导出/重复方法/死参数收口）

- **包级桶导出删除**:`src/zhishuxing/__init__.py` 只保留 docstring 与 `__version__`,
  6 个子包 `__init__.py` 清空——全仓 grep 证实符号一律从具体子模块导入
  (`from zhishuxing.core.navigation import ...` 等),桶从未被消费;
  `import zhishuxing` 不再连带加载 core/llm/rl。**`__version__` 补齐为 `2.4.0`**
  (2.4.0 升版时只改了 pyproject,漏了这份)。
- **`HubTransferAnimator` 死方法删除**:`_local_density()` 与 `_social_forces()` 与
  `step()` 内联计算逐行重复(内联版复用 `_pairwise()` 结果,方法版会重算),全仓零调用。
- **`amap.plan_route` 死参数删除**:`llm_api_key/llm_model/llm_base_url` 三个形参从未被
  传入(唯一调用方 `service.plan_route` 只传 `question/prefs`),LLM 配置本就全部来自
  `siliconflow_config()`。
- **文档**:src/README 同步(`__init__` 职责、rl 惰性导出描述、envs.py 早已懒加载的
  过期描述);AGENTS/GET-START 版本样例 2.3.0 → 2.4.0。

## 2026-10-07 · 2.5.0 可发售私有化 v1(离线授权 / 部署验收 / 管理端鉴权 / 离线交付包)

### Added(新增)

- **离线授权机制** `src/zhishuxing/licensing.py`:HMAC-SHA256 签名 + 客户名 + 有效期的
  `license.lic`;状态机 missing(试用)/valid/expiring(剩≤30天)/grace(过期≤14天宽限)/
  expired(拒启)/invalid(篡改按试用,如实标注);状态并入 `settings.read_state()` →
  GET /api/settings、doctor、移动端 keyStatus 三端展示;CLI `license` 子命令(查看/
  `--file` 激活);`serve` 启动门禁(过期超宽限打印原因、退出码 3);厂商签发工具
  `scripts/make_license.py`(不进镜像与交付包);桌面顶栏 `#lic-chip` 与移动端授权提醒
  横幅(`web/mobile/sw.js` CACHE_NAME v8→v9)。HMAC 对称密钥防复制不防逆向,v1 明示边界,
  升级路径 Ed25519 记 TODO。
- **部署验收** `src/zhishuxing/verify.py` + CLI `verify` 子命令:本地体检(配置/授权/核心
  端点/密钥不泄漏/数据目录可写/管理鉴权自检/镜像无 samples 时 WARN)与远程模式(`--url`
  探测运行实例);PASS/WARN/FAIL 三态 + 末行 `VERIFY PASS` + 退出码 0/1,markdown 验收
  报告落 `data/outputs/`(容器内 `verify --url` 即部署后验收,客户机免 Python)。
- **管理端鉴权** `src/zhishuxing/webapp/auth.py` + `app.py` before_request 分域门禁:
  `ADMIN_PASSWORD` 设置才启用(未设置 = 历史行为不变);保护域仅管理动作 POST 9 条
  (换导航图/微调/模拟指标/LLM 加载/RL 权重/RL 仿真/面板/报告/改配置),乘客对话/规划、
  `/mobile`、演示 GET、`/health` 保持公开;PBKDF2 口令校验 + HMAC 会话令牌(Cookie 与
  Bearer 同一枚,12h)+ 登录失败 5 次锁 10 分钟;`/admin/login` 独立登录页 +
  `/api/admin/login|logout`;`POST /api/settings` 的 loopback 双闸不放宽。
- **离线交付包** `packaging/build_deploy.py` + `packaging/deploy-compose.yml`:docker
  build → save|gzip → 组装 `dist/zhishuxing-deploy-<版本>.zip`(镜像/compose/.env.example/
  LICENSE/EULA/DEPLOY-PRIVATE/交付说明);镜像名与 tar 对号入座,数据三卷,license 挂载位。
- **文档** `docs/DEPLOY-PRIVATE.md`(离线部署五步/安全模型/升级回滚/排障)、
  `docs/eula-template.md`(明示需法务复核);`.env.example` 增 `ADMIN_PASSWORD` 项。

### Changed(变更)

- 版本 2.4.0 → 2.5.0;AGENTS「当前状态」:测试 234、CLI 子命令 11、路由 27/26;
- `packaging/exe_entry.py`、`desktop_entry.py`:serve 启动前加授权门禁,过期弹原生
  错误窗(onefile 弹控制台/onedir 弹对话框),退出码 3。

### 验证

- `python -m pytest` 234 passed(含新增 test_licensing 14 例、test_auth 13 例、
  test_verify 6 例,全部离线确定性);`python -m pyflakes src/ scripts/ tests/` 0 告警;
- 签发→激活→判定往返(测试客户/365 天 → valid);过期超宽限 `serve` 退出码 3;
  分域鉴权:未设口令全端点行为不变、设口令后管理 POST 401/Cookie 与 Bearer 放行/
  锁定生效/公开域不受影响;verify 本地模式出报告、过期授权转 FAIL;
- **Docker 交付包真机实测(本机 Docker Desktop 29.8.2)**:`build_deploy.py` 出真包
  `dist/zhishuxing-deploy-2.5.0.zip`(120 MB)→ 删本地镜像后 `docker load` 交付 tar
  (5.8s)→ `compose up -d` 25s healthy → 容器内 `verify --url` **VERIFY PASS** →
  验收报告经 `/outputs/` 下载(200)→ 厂商签发 license 拷入容器即生效(WARN 清零,
  无需重启)→ 宿主机验证管理端鉴权(未登录 401 / Bearer 200)→ `compose down -v`
  清场。授权有效态 `license` 子命令的"文件:"字段空显示 bug 当场发现并修复
  (test_licensing 补一条断言钉住)。

## 2026-10-21 · 更正:存量乘客档案恢复失败留痕

- `llm/assistant.py` 从 SQLite 恢复序列化档案 dict 的路径里,`except Exception: pass` 按
  「解析失败即无档案」设计(对不可信存量数据防御本身是对的),但零痕迹——档案损坏只会
  表现为「对话怎么都不记得偏好」,无从排查。
- 修法:捕获后打印 `[profile] 存量乘客档案恢复失败,按无档案处理:<type>: <msg>`(一行,
  与 webapp/service.py `_autoload_real_llm` 的「降级可见」约定一致),行为不变。
- 验证:pyflakes 0(src/scripts/tests)、`python -m pytest -o addopts="" -q` 234 passed
  (测试数与基线一致)。

## 2026-10-31 · 2.6.0 前端体验精修(移动端 PWA + Web 控制台)

### Added(新增 · 移动端 PWA)

- **语音输入(Web Speech API)**:输入条新增麦克风键,识别实时填入输入框但不自动发送
  (让乘客先确认听到什么);不支持/无权限的浏览器入口不出现或如实提示,无硬依赖
- **PWA 安装引导**:个人中心新增「安装到主屏」,`beforeinstallprompt` 触发才露出,
  standalone 运行时自动隐藏;`appinstalled` 后入口消失
- **系统分享**:路线分享优先走 `navigator.share` 系统面板,不支持/用户取消再退回复制
- **触感反馈**:发送与路线落地时 `navigator.vibrate` 轻震(iOS 无此 API 静默忽略)
- **点气泡重问**:点击自己的历史气泡把原文填回输入框重新编辑

### Fixed(修复 · 移动端 PWA)

- **对话页输入条被悬浮胶囊底栏盖住(存量 bug)**:底栏改悬浮胶囊(60px+10px)时
  `#chatPage` 的 composer 底距没跟上,输入条(含麦克风键)整体被压在底栏之下;
  现 composer-wrap 底距对齐滚动页的 96px 基准(浏览器实测确认)
- 输入框长 placeholder 折行挤出 Windows 原生滚动条箭头:placeholder 缩短 +
  `scrollbar-width: none`(高度自适应仍由 autosize 负责)

### Changed(变更 · 移动端 PWA 动画/质感)

- 客流卡展开/收起改 `grid-template-rows 0fr→1fr` 过渡:高度连续、可中断,
  替代 display:none 硬切(快速连点不再闪跳)
- 「3D/AR 导航模式」与「主题三态分段」改滑动药丸指示器(与底部导航同一语言);
  页面隐藏时定位自动跳过,resize/切页即时落位
- 规划结果整组错峰入场(车票→KPI→预览→步骤,约 60ms 阶梯);欢迎卡分层入场;
  历史弹层行微错峰;恢复历史消息不再重放入场动画(只让新消息动)
- KPI/客流数字等宽排印(`font-variant-numeric: tabular-nums`);主按钮按压阴影收紧;
  发送键请求中变加载环;底部弹层加 iOS 式抓握条
- **新路线红点**:路线落地且用户不在路线页时「路线」tab 点亮红点,进入即消
- route-mini 卡补键盘 Enter/Space 支持;mode-tab 补键盘支持
- `sw.js` 缓存 `v9` → `v10`;`web/README.md` 行数/缓存名/行号引用同步

### Added(新增 · Web 控制台)

- **折线图描线入场**(奖励曲线/微调指标):坐标轴常驻、折线自左向右 0.56s ease-out
  落笔;同画布新数据到达即取消旧帧,reduced-motion 直出终态
- **画廊图片点击放大**(新标签页原图,`cursor: zoom-in`)
- 命令面板(Ctrl+K)图标从 emoji 统一为 sprite 体系;`icons.svg` 新增 `i-phone`,
  引用版本号 `?v=5.8` → `?v=6.0`,`style.css`/`app.js` 引用升 `?v=6.0`

### Changed(变更 · Web 控制台)

- KPI/规划/仿真/导航结果数值更新加轻换位 tick(与顶栏标题 swap 同语言)
- 站内路径落笔改 easeOutCubic(先快后慢,与移动端同语言)
- 待回复消息加内联 spinner;规划参数弹窗支持 Escape 关闭(与 cmdk/LLM 面板同习惯)

### Changed(变更 · 管理登录页)

- `admin_login.html` 品牌对齐控制台:靛蓝紫渐变主按钮/焦点环、logo 徽标、
  卡片入场、错误 shake、登录中 spinner、reduced-motion 降级

### 验证

- `python -m pytest` 234 passed;`python -m pyflakes src/ scripts/ tests/` 0 告警;
  `zhishuxing smoke` 通过;`tests/test_api.py -k mobile` 通过
- 浏览器实测(Chrome + 本机 serve):移动端对话流/客流卡展开/路线页 mode-tabs 药丸/
  高德真实底图/设置页 seg 药丸/输入条修复后布局全部走查通过;控制台欢迎页/
  命令面板 sprite 图标/RL 奖励曲线描线动画走查通过

## 2026-10-31 · 2.7.0 体验精修第二轮(动线底图 / 瞄准准星 / 图例高亮)

### Added(新增)

- **品牌「动线」底图母题落进两端背景**:虚线换乘路径 + 站点圆标的极淡 SVG 层,
  与 KPI 顶线、四步串联线同一设计语言;纯静态层零动画成本,明/暗主题各自取值
- **枢纽网格悬停准星格**(控制台):鼠标悬停高亮当前格(青色描边),拾取起终点有瞄准感;
  点击/悬停共用 `cellFromEvent`,触屏不触发无副作用
- **奖励曲线图例悬停高亮**:悬停某个 seed 图例键,对应曲线加粗高亮、其余淡化为灰色
  参考线,移出恢复(`drawLineChart` 支持 per-series 线宽;图例改按钮形态)
- 顶栏滚动投影(内容滑入顶栏之下时加深,与移动端 `.scrolled` 同款)

### Changed(变更 · 微交互)

- 画廊图片悬停轻抬 + 加载淡入(`img-fade`/`img-in`,缓存命中直显)
- Toast 状态徽章弹入;主按钮悬停图标微动;折叠原始数据展开淡入;Dock 引擎钮悬停缩放
- 侧栏项/地图钮/命令面板项/顶栏搜索钮按压物理补全;欢迎页统计卡悬停轻抬(精确指针门控)
- 移动端:客流卡头按压态、路线迷你卡悬停箭头轻推、场景卡悬停轻抬(hover 门控)、
  通知/预警开关触感 buzz(6)

### Fixed(修复)

- **侧栏滑动药丸在 View Transition 快照期间测量失真残留**(浏览器实测 pill 宽 1416px,
  约为窄屏断点侧栏宽度):交换完成后按真实布局补定位一次

### 验证

- `python -m pytest` 全量通过;`python -m pyflakes src/ scripts/ tests/` 0 告警;
  浏览器实测:动线底图/顶栏投影/网格准星/图例高亮/画廊悬停逐项目视确认
- `sw.js` 缓存 `v10` → `v11`;版本 2.6.0 → 2.7.0

## 2026-10-31 · 2.8.0 视觉高级化(活体动线 / 胶片颗粒 / 玻璃层级)

### Added(新增 · 控制台)

- **活体动线**:低透明度光点沿换乘路径巡航(CSS `offset-path` 合成器动画,
  常速 linear、首尾淡入淡出;欢迎页与工作台各一层,`reduced-motion` 自动消失)——
  枢纽"有乘客在流动"的标志性瞬间
- **全屏胶片颗粒**:固定合成层 + `feTurbulence` 噪声(3.2% overlay),大面积平色获得打印质感
- **侧栏质感**:青色底光球 + 颗粒纹理 + 滑动药丸辉光
- 顶栏发丝渐变线 + 玻璃强化(blur 16 saturate 165%);Dock 底条玻璃化 + 顶部发丝线
  (`color-mix` 带 `@supports` 回退)

### Changed(变更 · 动效)

- 主按钮/欢迎页主按钮**悬停光泽扫过**(transform 斜切高光带,离场瞬时归位)
- KPI 顶线改**沿线流动的数据脉冲**(渐变首尾相接平滑循环)
- 欢迎页主标题**流光**(em 渐变 7s 缓慢滑动);卡片悬停电致描边(1px 靛蓝光环);
  统计卡悬停辉光加强

### Changed(变更 · 移动端 PWA)

- 发送键**呼吸辉光**(CTA 心跳);底栏激活图标辉光;车票中心圆标路线"心脏"脉冲;
  拥挤徽章告警辉光(与客流脉冲柱同语言);欢迎卡标识入场后一次光泽扫过;
  用户气泡顶部内高光(玻璃弧面)
- `sw.js` 缓存 v12 → v13;资产引用 style/app.js 升 v6.3

### 验证

- `python -m pytest` 全量通过;`python -m pyflakes` 0 告警;
  浏览器实测:`hubTravel` 6 动画运行中(offset-distance 推进/透明度按关键帧变化),
  欢迎页流光标题、工作台 KPI 脉冲、顶栏/Dock 发丝线、移动端呼吸辉光逐项目视确认
- 版本 2.7.0 → 2.8.0

## 2026-10-31 · 2.8.1 移动端背景重设计「星穹站台」

### Changed(变更)

- **天穹纵深**:body 底色从平涂 `--bg` 改为垂直三段式渐变(顶部深→中部过渡→底部微亮,
  明/暗各自取值),替代原两层顶部极光
- **四球呼吸光层**:左上青/右上靛/右中紫/左下青绿四颗网格光球对角流动,
  整层 11s opacity 呼吸(纯合成动画);左下青绿球特意藏在毛玻璃输入条/底栏后面,
  磨砂玻璃透出流动颜色
- **纹理层合并**:点阵网格与「动线」母题合入 `body::after` 同一伪元素(明/暗双套);
  新增 `html::after` 胶片颗粒层(4.5% overlay,与控制台同源)
- **光点巡航**:两颗 offset-path 光点沿路径滑行,下段路径穿过毛玻璃化作一团柔光
  (与控制台 hubTravel 同语言;`reduced-motion` 下不出现)
- **启动屏场景化**:平涂渐变 → 双光球 + 超大白色动线母题斜贯 + 提示文字呼吸
  (hint 呼吸与入场动画合并声明,避免 shorthand 互相覆盖)

### 验证

- `tests/test_api.py -k mobile` 通过;浏览器实测:暗色/浅色对话页四球透光、
  毛玻璃透色、启动屏动线光带逐项目视确认
- `sw.js` 缓存 v13 → v14;版本 2.8.0 → 2.8.1

## 2026-11-01 · 安全补课批(ZX-01~05):桌面 XSS 全量转义/请求预算/会话主键恢复/产物 UUID 化/打包路径

> 10-26 专项审查(周报-2026-10-26)排定的安全批,11-01 全矩阵核查确认仍未修,本次一次性关闭五项 P1。
> 移动端 PWA 此前已 escapeHtml 全覆盖,本批只动桌面控制台与后端;SEC-07(Ed25519)仍随 TODO 任务 5 排期。

### Security(安全)

- **桌面端 DOM XSS 全量转义(SEC-01/ZX-01)**:`webapp/static/app.js` 渲染链路上所有
  裸拼 API 数据的 sink 统一过 `esc()`——`route_error`/`profile.summary`/路线
  `details`/`tips`/`kb_refs`(title/score/source)/`cost`/检查点表(算法与路径)/地标下拉
  与安检口 chips/RL 策略标签/客流分组表/需求徽章/微调结果/报告失败标题/报告画廊文件名,
  共 15 处;此前仅移动端全覆盖,桌面端 `renderChatAnalysis` 是唯一成片裸拼区。
- **管理令牌 localStorage 副本清除(ZX-01)**:`admin_login.html` 登录成功后不再把令牌
  写入 localStorage——主链路本就走 HttpOnly Cookie(auth.py COOKIE_NAME),该写入是
  历史遗留死代码,只构成 XSS 可达面;改为顺手清除存量旧副本。body 里的 token 保留给
  脚本/CI 的 Bearer 用法(app.py 登录路由注释口径)。
- **公开端点请求预算(SEC-02/ZX-02)**:应用级 `MAX_CONTENT_LENGTH=256KiB`(超限 413)+
  字段级预算——`message`/`question` 2k 字符、`prefs`/`config`/`via` 4k、`groups` 60k、
  `title` 120、`observations` 64 智能体×128 维;`session_id` 强制 1-64 位
  `[A-Za-z0-9_-]` 格式(防任意长串灌库,PERF-01 配套)。`/api/rl/act`、`/api/rl/simulate`、
  `/api/dashboard/run`、`/api/chat/reset`、`/api/navigation/plan` 补齐 ValueError→400
  细分(此前 ValueError 统一走 500)。

### Performance(性能)

- **会话恢复只走主键(PERF-01/ZX-03)**:`TransferAssistant.handle` 重启恢复从整表
  `load_missing()` 回填改为按 session_id 主键 `load()` 单行;`purge_expired()` 在
  TransferAssistant 构造(即服务启动)时真实调用,30 天 TTL 兑现;session_store.py 与
  assistant.py 的 docstring 同步转真(原「启动清理」「不做持久化」均为失实声明)。
- **Web 产物按请求 UUID 化(PERF-03/ZX-04)**:`run_dashboard`/`rl_simulate` 的固定
  `zhishuxing_web_dashboard.png`/`rl_guided_simulation.png` 与
  `rl/runtime.render_reward_curve` 的固定 `rl_reward_curve_web.png` 改为按请求 uuid
  后缀命名,并发请求不再互相覆盖;新增 `_sweep_outputs()` 同前缀产物只留最近 30 张,
  防 output_dir 无限增长(固定名历史产物不受影响)。

### Fixed(修复)

- **便携包构建路径(MAINT-01/ZX-05)**:`packaging/package_zip.py` 构建脚本引用从
  `scripts/build_exe.py`(已不存在,默认命令 100% 失败,exit 2 实测)修正为
  `packaging/build_exe.py`,提为模块常量 `BUILD_SCRIPT` 并以测试钉住。

### Tests(测试)

- 新增 6 条:`test_api.py` 请求预算三连(413/message 超限/session_id 格式)、
  `test_session_store.py` 会话主键恢复(替换空转断言——旧跨实例测试只验「内存里存在」,
  实际由 setdefault 兜底恒真且恢复路径从未真正断言;新测试 monkeypatch 封死
  `load_missing` 并断言历史内容从库恢复)、`test_packaging.py` 脚本存在性两查。
- 门禁:pytest 240 passed;`node --check app.js` 通过;pyflakes 0 告警;`zhishuxing smoke` 通过。

## 2026-11-02 · v2.9.0 动效体系定型与体验收口(设计纪律轮)

> 依据 2026-11-01 全矩阵周报「研究一 · 前端品味」对 zhishuxing 的落地建议(§7.5)与 Emil 动效
> 方法论:本轮不堆新装饰,做「受限的令牌系统 + 一处签名件 + 细节纪律」。随批关闭体检卡 P2/P3
> 四项诚实度问题(ZX-08/09/11/12);ZX-14(dist/build 磁盘残留)同步清理。

### Added(新增)

- **动效时长阶梯(两端)**:`style.css` 与 `mobile_app.html` 新增 `--dur-1(.12s 按压)/--dur-2(.18s
  悬停与焦点)/--dur-3(.24s 入场与弹层)/--dur-4(.3s 抽屉与大面板)`,高频交互过渡(按钮/输入/
  chips/侧栏/卡片/KPI/toast/命令面板/页面切换/抽屉)全部改从阶梯取值,替代原先约 19 个散落时长;
  欢迎页动效与悬浮球拖拽编排属 Rare 场景,保留自有节奏并注释说明。
- **空状态插画人格**:桌面 6 处空态(枢纽导航未规划/规划结果/RL 无权重/无仿真/报告画廊/微调预览)
  与移动端换乘历史空态,从「图标+虚线徽托」换为框线风格迷你 SVG 插画(与 sprite 同语言:圆头描边、
  动线虚线、站点圆标、accent 青色点);实线以 `pathLength=100` 归一做描线入场(0.7s ease-out,
  虚线与色点错峰淡入),`prefers-reduced-motion` 下直出终态(显式落 `stroke-dashoffset:0`,
  防全局禁动画规则把插画藏掉)。
- **报告画廊错峰入场**:7 类报告图片按 `--gi` 索引 60ms 阶梯上浮淡入,与移动端错峰语言同源。

### Changed(变更)

- **移动端输入控件 ≥16px**(iOS 聚焦缩放防护):对话输入框 14→16px、偏好备注 13→16px、
  站点下拉 13→16px。
- **移动端中文排版**:正文 `line-height 1.75`;`hanging-punctuation: allow-end`(Safari 生效,
  其余引擎忽略)。
- **天穹渐变防 banding**:`@supports` 渐进改用 `in oklch` 插值(明/暗两套),不支持处保持原 sRGB。
- **移动端 ServiceWorker 壳文档改网络优先**:`mobile_app.html` 与 manifest 拉网优先、离线回退缓存
  (旧策略 cache-first,漏 bump `CACHE_NAME` 用户就拿不到新壳);图标等静态资产维持 cache-first;
  缓存名 v14 → v15。
- 控制台/移动端资产引用升 `?v=6.4`;侧栏版本号 v2.9.0。

### Fixed(修复 · 诚实度与性能,随批搭车)

- **od_source 如实上报(ZX-08)**:`call_llm_extract_od` 失败静默回退正则时,返回值携带
  `_source=regex`,`plan_with_amap` 按真实来源上报——配了 LLM Key 但 LLM 挂掉时,响应不再谎称
  `od_source: "llm"`(降级可见约定在这条链路恢复)。
- **模拟微调不再冒充真实型号(ZX-11)**:`simulate_finetune_metrics` 的 `model_id` 从编造的
  `qwen3.8-27b` 改为 `demo-mock`(配置里的 `SILICONFLOW_MODEL` 默认值是真实服务器配置,不动)。
- **/api/rl/rewards 渲染记忆化(ZX-09)**:新增 `MADDPGRuntime.reward_fingerprint()`(文件名/大小/
  mtime),服务层按指纹复用上次响应——数据未变的公开 GET 不再每次读全部 NPY 并重绘 PNG。
- **/api/navigation/load 路径白名单(ZX-12)**:`file_path` 解析后必须位于 `configs/` 目录,
  否则 400——防换 token 后以管理域端点探测盘上任意路径。

### Removed(清理)

- 本机 `dist/`(883MB)与 `build/`(86MB)历史构建产物删除(ZX-14,均可由 packaging 脚本重建)。

### 验证

- 新增 4 条行为断言:`test_amap.py` LLM 回退上报 `_source=regex`/成功上报 `_source=llm`、
  `test_api.py` rewards 指纹记忆化(命中缓存不读 NPY,指纹变化才重绘)、simulate_metrics
  `model_id == "demo-mock"`、navigation/load 白名单外 400(含穿越路径)。
- 门禁:pytest 244 passed;pyflakes 0 告警;`zhishuxing smoke` 通过;`node --check` app.js/sw.js 通过。
- 浏览器实测(Chrome + 本机 serve,亮/暗双主题):欢迎页 → 工作台入场、枢纽导航/RL/报告三处空态
  描线插画终态正确、7/7 报告画廊 `--gi` 0-8 错峰落位、主题切换过渡正常;移动端 390×844——输入框
  computed 16px、`linear-gradient(in oklch, …)` 实际生效、正文行高 28px、历史空态时钟插画、
  对话流发送/气泡/spinner 正常。

## 2026-11-02 · 更正:ZX-08 两测补 openai 可选守卫(2.9.0 CI 红灯修正)

- 2.9.0 新增的 `test_call_llm_extract_od_fallback_reports_regex` / `..._success_reports_llm`
  经 `monkeypatch.setattr("openai.OpenAI", …)` 依赖 `openai` 可导入,而 openai 属可选 `[llm]`
  extra——CI 只装 `.[dev]` 无 openai,两测 ModuleNotFoundError 致 CI failure;本机全局装有
  openai 故本地绿,属环境差异漏检(2.9.0 提交信息声称的 pytest 244 在本机为真)
- 修正:两测开头 `pytest.importorskip("openai", …)`,与缺 torch 跳过 RL 用例同口径;
  本机 `pytest tests/test_amap.py` 11 项仍全绿(总数 244 不变),CI 口径为 242 passed + 2 skipped

## 2026-11-04 · v2.10.0 微交互与回读保护(体验精修第三轮)

> 2.9.0 动效体系的小步续航:不新增装饰,补两处真实的体验缺口(对话流拽人回读、分析卡硬切),
> 并把管理登录页纳入全站时长阶梯。含两项调查结论(见文末),均不产码。

### Added(新增)

- **对话流回读保护(两端)**:此前每条消息落地都无条件 `scrollTop = scrollHeight`,用户向上翻阅
  历史时会被新回复拽回底部。现仅当视口离底 ≤80px(本就在跟随)或消息来自用户本人发送时才滚底;
  桌面 `appendChatMessage` 与移动端 `chatNearBottom()` 同一口径——用户发送/typing 指示器/切页/
  新会话四处仍无条件滚底(均为主动行为或初始定位)。
- **对话分析卡换新轻入场(桌面)**:`renderChatAnalysis` 整块换新时挂 `ca-in` 走 0.18s ease-out
  淡入上浮,替代硬切 teleport;reduced-motion 由全局规则抑制为直出。

### Changed(变更)

- **管理登录页时长并入全站阶梯**:`admin_login.html` 补 `--dur-1/--dur-2`,输入框与按钮过渡取令牌;
  `cardIn(.4s)`/错误 `shake(.3s)` 属首次抵达与错误反馈的 Rare 场景,保留自有节奏。
- `app.js` 顶部补章节索引横幅(①API→⑬启动):单文件 PWA 不引打包器的前提下的最低成本可维护性
  改进(体检卡建议项);资产引用升 `?v=6.5`,侧栏版本号 v2.10.0。

### 调查结论(只记录,不改码)

- **欢迎页按钮命中核查**:上轮浏览器走查中 Playwright 对「进入系统/进入工作台」点击超时,
  本轮 `elementFromPoint` 复核两点均无覆盖、按钮自身命中——超时系自动化工具对无限装饰动画
  (heroSheen/光点巡航)的稳定性误判,非产品缺陷,真实鼠标不受影响。
- **欢迎页滚动显现迁移 CSS scroll-driven:评估后否决**——scroll timeline 的进度与滚动双向绑定,
  会把「进过视口定格,回读不再打扰」的一次性 reveal 变成可逆显现,与既有设计决策冲突;
  3 个元素的一次性 0.55s 过渡本就不构成主线程压力,不值得为合成器收益换语义。

### 验证

- 浏览器实测(Chrome + 本机 serve):桌面对话流撑满后(内容 1020px),请求期间上翻到顶,
  回复落地 `scrollTop` 保持 0;回到底部场景回复照常跟随(atBottom=true);`ca-in` 挂载确认;
  移动端同法验证保护生效。管理登录页走查过渡正常。
- 门禁:pytest 244 passed;pyflakes 0 告警;`zhishuxing smoke` 通过;`node --check` 通过。

## 2026-11-04 · v2.11.0 可访问性收尾 + 首访主题跟随(体验精修第四轮)

### Added(新增)

- **控制台首访跟随系统明暗**:`initTheme` 无记忆偏好时改按 `prefers-color-scheme` 初始化
  (原固定 light),与移动端三态「auto」同口径;已有记忆偏好则尊重记忆,切换行为不变。
- **对话流读屏播报**:桌面 `#chatMessages` 与移动端 `#chatLog` 加 `aria-live="polite"`,
  新回复对读屏用户可感知;桌面 `#toastHost` 同加(移动端 toast 已有 role="status")。
- **画廊键盘可达**:报告图片补 `tabindex=0`+`role="button"`+`aria-label`,Enter/Space 与点击
  同效打开原图(与移动端 route-mini 键盘支持同一先例);img 补 `alt=文件名`。

### Changed(变更)

- 六处 canvas 补 `role="img"` + 中文 `aria-label`(枢纽网格/奖励曲线/规划路线小图/对话内路线小图/
  微调损失/微调指标);dock 输入框补 `aria-label`(与移动端同文案)。
- 资产引用升 `?v=6.6`;侧栏版本号 v2.11.0。

### 验证

- 浏览器实测:清空记忆偏好后首访,环境系统偏好 dark → 页面正确落 dark(回退分支实测命中);
  `aria-live`/canvas role/label/dock aria-label 逐项读取确认;画廊 img 可聚焦,Enter/Space 打桩
  验证均打开对应原图 URL。
- 门禁:pytest 244 passed;pyflakes 0 告警;`zhishuxing smoke` 通过;`node --check` 通过。

## 2026-11-05 · v2.11.1 桌面+移动端主按钮纯色化(研究二轮 taste 批)

### Changed(变更)

- **主色渐变纯色化**(研究二轮口径,照 rr-studio 6163925 同决策):删除
  「靛蓝→紫」同族明度渐变 `--grad`/`--grad-h`(135deg #4f46e5→#7c3aed 族),
  主按钮 `.btn`/悬浮球 `.dock-fab`/用户气泡 `.chat-msg.user`/欢迎页
  `.btn-hero.primary` 一律改 `--primary` 纯色,hover 走 `--primary-h` 或
  `filter: brightness(1.08)` 明度深化。依据:品味研究第 1 条「同一族的明度
  渐变是最偷懒的渐变」。`--grad-glow` 随批更名 `--btn-glow`(值不变,名实一致)。
- **移动端 mobile_app.html 同款渐变一并纯色化**:`--grad` 两处定义(明/暗)
  改 `var(--primary)`,变量名沿用以减 diff(10 处引用零改动,行为等价)。
- 保留的两处 135deg(`.feat .f-ico`/`.hstat .hs-ico`)为靛蓝→青跨色相低透明
  洗底(动线母题双色),补品牌豁免注释,非同族明度渐变。
- 资产引用 `style.css?v=6.6` → `?v=6.7`(缓存穿透)。

### 验证

- 门禁:pytest 244 passed;`node --check app.js` 通过;浏览器不可开,
  代码层验证:style.css 花括号配平(684/684)、`grep --grad` 归零、
  `135deg` 仅剩 2 处品牌豁免注释在案。

## 2026-11-11 · 后端优化批之一:A* 热路径 blocked set 惰性缓存(PERF-05)

### Changed(变更)

- `NavigationMap._blocked_set` 由"每次访问重建 `set(tuple(...))`"改为加载/构造后惰性缓存
  一次(`__post_init__` 置空缓存,首访构建 `frozenset`,之后返回同一份)。A* 每展开一个邻居
  都走 `is_valid → _blocked_set`,不缓存时每次重建 25 格 set;前提已核:全仓 `blocked` 无
  append/extend/add 写入(只读引用仅 simulation.py:213、system.py:70 传渲染器与
  service.py 取数),缓存安全。
- cProfile 实测(20 次 `plan_with_preferences`,tag_penalties+soft_via 场景):总耗时
  0.389s → 0.092s(约 4.4×),`_blocked_set` 累计占比 68.6% → 2.9%。
- API 契约、产物命名、`configs/hub_default.json` 结构零变化。

### 验证

- 门禁:`pytest tests/test_navigation.py tests/test_navigation_prefs.py tests/test_simulation.py`
  21 passed;`python -m pytest` 245 passed(244→245,新增缓存行为断言 1 条);
  `python -m pyflakes src/ scripts/ tests/` 0 告警;`zhishuxing smoke` 通过
  (真实规划 [1,2]→[28,12] 含 via=security)。

## 2026-11-11 · 后端优化批之二:请求入口健壮性——校验前置 400 + 错误信封统一

### Changed(变更)

- **输入校验前置**(客户端可触发的 500 归零):
  - `parse_groups` 缺 name/start/goal、坐标非两项、组非对象一律 `ValueError`
    (原 KeyError/IndexError/TypeError 落 500),走路由层既有 ValueError→400 通道;
  - `plan_landmark_path` 未知 via 由 `KeyError` 改用户可读 `ValueError`(原 500→400),
    `test_unknown_landmark_raises` 同步改钉 ValueError;
  - `/api/chat` 与 `/api/plan` 的 prefs 非 dict(如字符串「行李多」)回 400,不再进业务层;
  - `/api/plan` 的 engine 非 amap/hub 回 400(原静默落 hub)。
- **错误信封统一**:9 处裸 `jsonify({"error": ...})` 全部补 `"ok": false`
  (`grep -c 'jsonify({"error"'` 归零);`fail()` 非 HTTPException 分支改回固定文案
  「服务器内部错误…」,异常原文不再 str(exc) 直传客户端,详情只进服务端日志
  (logger.exception 已在打)。
- **兼容性**:error 键读取不变——`static/app.js:44` 判 `response.ok || payload.ok===false`、
  `:45` 读 `payload.error`,本批为纯增量(成功路径契约零变化),已复核无需改前端。
- 缘由:此前 `/api/navigation/plan` 带 `via:["不存在"]`、`/api/dashboard/run` 缺字段等
  纯客户端输入错误会以 500+异常原文落响应,既污染故障观测又泄露内部信息。

### 验证

- 门禁:`zhishuxing smoke` 通过('Web smoke test passed.');`pytest tests/test_api.py`
  + `pytest tests/test_navigation.py` 40 passed;`python -m pytest` 251 passed
  (245→251,新增 6 条:prefs/engine 校验、未知 via 400、缺字段 400、500 固定信封);
  `python -m pyflakes src/ scripts/ tests/` 0 告警。

## 2026-11-11 · 后端优化批之三:对话会话运行期有界化(PERF-06)

### Changed(变更)

- **内存侧 LRU 上限**:`TransferAssistant._sessions` 加 `SESSION_CAPACITY = 512` 容量上限,
  新增 `_touched` 记账(sid → 最近活跃时刻,monotonic,不污染会话本体与落库 payload),
  每次 handle 记活跃,超限按最近活跃淘汰最久不动者;被淘汰会话已写穿 SQLite,
  再次到访走既有主键 load 恢复——命中恢复/重置语义(assistant.py)零改动。
  此前 `/api/chat` 的新 session_id 命中即常驻,内存随新 sid 无界增长直到重启。
- **DB 侧低频清理**:30 天 TTL `purge_expired()` 原「仅服务启动清一次」改为
  「启动一次 + 写入路径机会式清理(每 `PURGE_EVERY_N_SAVES = 50` 次落库触发一次)」;
  清理与落库同锁串行(`_purge_expired_locked` 内部实现体,避免 Lock 不可重入死锁),
  清理失败只打印告警不阻断本次落库。零新依赖、schema 不变。
- `session_store.py` 与 `assistant.py` 模块 docstring 同步新语义;
  AGENTS「对话会话已持久化」条目的「30 天不活跃自动清理」语义仍成立,核对未改。

### 验证

- 门禁:`pytest tests/test_session_store.py` 11 passed(新增上限淘汰与低频清理用例 2 条);
  `python -m pytest` 253 passed(251→253);`python -m pyflakes src/ scripts/ tests/` 0 告警;
  跑完 `git status` 干净(data/outputs/ 等运行时产物未入库)。
