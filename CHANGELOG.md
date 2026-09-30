# 更新日志

本文件从 2.2.0 起向前记录(更早版本不追溯补写,历史见 git 提交记录)。
版本遵循语义化版本:破坏契约 = 主版本,加能力 = 次版本,修缺陷 = 补丁。

## [2.2.0] - 2026-09-29

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
  按钮弹出的居中弹窗(规划成功自动收起),「规划结果」为地图右侧停靠面板
  (高德/百度 web 版路线详情形态:未规划时显示操作提示,规划后展示
  路径长度/途经点/完整坐标,窄屏折叠为底部抽屉);
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
