# 智枢星 v2 架构说明

## 设计原则

1. **逻辑平移不重设计**：合成数据数值特征、MADDPG 超参与产物命名、API 契约、前端交互全部保持
2. **单一来源**：字体配置/滑窗平均/CSV IO/合成数据生成器历史上分散 6+ 份，全部归一到 `analysis/`
3. **路径零 CWD 依赖**：所有产物锚定 workspace（`config.Paths`），密钥全部环境变量
4. **可选依赖隔离**：`torch`/`mlagents_envs`/`openai` 均为可选，Web/演示链路无硬依赖（`rl` 包惰性导出）

## 模块与数据流

```
                 ┌────────────┐   观测/动作契约   ┌──────────────┐
 Unity (exe/Editor)──▶ rl.envs ──▶ rl.runner ──▶ data/outputs/*.npy
                 │  (gym风格)    (MADDPG/MATD3)  data/model/*.pth
                 └────────────┘                      │
                                                     ▼
 core.navigation (A*) ──▶ core.simulation ◀── rl.runtime（策略加载/推理，无权重回退启发式）
        │                    │       ▲
        ▼                    ▼       └── PolicyProtocol（core 与 rl 解耦）
 core.flow ──▶ core.system（编排）──▶ analysis.plotting（热力/轨迹快照）
                    │
 webapp.service ────┴──▶ webapp.app（Flask）──▶ Web 控制台 / 移动端 PWA
        │
        ├── llm.assistant（对话编排）──▶ llm.profile（需求档案）+ llm.kb（BM25 经验检索）
        │                                   └─ 起终点命中地标 → 偏好规划；否则 ▼
        ├── planning.amap（高德真实规划）──▶ REST 高德 + LLM OD 提取
        └── analysis.reports（7 类报告）──▶ data/outputs/*.png|csv|gif
```

### 对话式助手链路（`/api/chat`）

```text
消息 → ① llm.profile：规则关键词优先、LLM JSON 兜底解析 → PassengerProfile
     → ② 与会话档案增量合并（多轮累积，会话态在内存）
     → ③ OD 分流：起终点命中枢纽地标 → core.navigation.plan_with_preferences（偏好加权 A* + 软必经点）
                   否则 → planning.amap（高德引擎，无 Key 时答话术不 500）
     → ④ llm.kb BM25 检索 top-3 站内经验（注明来源）
     → ⑤ llm.adapters 合成回答；Mock 模式确定性模板，全链路离线可演示
```

## 关键决策

| 决策 | 说明 |
|---|---|
| Flask 保留 | 历史端点契约与 waitress 部署延续；前端 fetch 无需改动 |
| core.simulation 与 rl 解耦 | 仿真通过 `PolicyProtocol` 注入策略，MADDPGRuntime/HeuristicPolicy 可互换；无 torch 也能跑 |
| MATD3 并入 agents.py | TD3 三技巧作为独立类路径，`--algorithm MATD3` 真正生效（历史版本分支被注释，属死代码） |
| 合成数据单一来源 | `analysis/synthetic.py`：拥堵矩阵（12区×17时段、三峰、分层缓解率）、换乘时间分布（520→398 收敛）、安检排队（sigmoid 分流、双种子）、微调指标（历史两套发散实现合并，`monotonic` 参数兼顾两种展示形态） |
| 报告函数化 | `analysis/reports.py` 每个 `run_*_report()` 可 import 可 CLI，返回结构化结果；webapp 的“既有功能联动”从 subprocess+吞异常 改为进程内调用+逐报告状态 |
| PWA 同源服务 | `/mobile`、`/mobile/<file>` 同前缀，ServiceWorker scope 覆盖页面；`planRoute` 先 fetch `/api/plan`，失败回退 mock 并标注“演示数据” |
| 高德底图双通道 | `home()` 经 Jinja 注入 `ZSX_CONFIG`（`AMAP_JS_KEY`/`AMAP_SECURITY_CODE`，均来自环境变量）；前端动态加载高德 JS API 2.0 渲染真实底图（深色样式 + 方向折线 + 起终点标注），脚本加载/初始化失败或未配置 Key 时自动回退 Canvas 折线示意并在说明栏注明原因 |
| 对话助手 Mock 闭环 | `TransferAssistant` 每一环（解析/规划/检索/合成）在无密钥时都有确定性降级：规则解析兜底 LLM、内置枢纽引擎兜底高德、模板回答兜底对话合成——断网可完整演示 |
| .env 自动加载 | `config.py` 启动时解析 workspace 根 `.env`（键=值、行注释，零第三方依赖），不覆盖已存在环境变量；`.env.example` 提供模板 |

## 迁移映射（旧 → 新）

| 旧位置 | 新位置 |
|---|---|
| `program/MADDPG/MADDPG_main.py` | `src/zhishuxing/rl/runner.py` + `cli.py train` |
| `program/MADDPG/{maddpg,matd3}.py` | `src/zhishuxing/rl/agents.py` |
| `program/MADDPG/networks.py` / `replay_buffer.py` / `environment.py` | `src/zhishuxing/rl/{networks,buffer,envs}.py` |
| `program/MADDPG/zhishuxing/rl_bridge.py` | `src/zhishuxing/rl/runtime.py` + `core/simulation.py` |
| `program/MADDPG/zhishuxing/{navigation,system,visualization}.py` | `core/{navigation,system}.py` + `analysis/plotting.py` |
| `program/MADDPG/zhishuxing/adapters.py` | `llm/adapters.py`（新增 SiliconFlow 真实实现） |
| `program/MADDPG/webapp/*` | `src/zhishuxing/webapp/*`（新增 `/api/plan`、`/mobile`） |
| `program/MADDPG/plot_*.py` / `animate_transfer_env.py` | `analysis/{synthetic,reports}.py` + `core/animation.py` |
| `program/fine/*` | `analysis/reports.py`（效率对比）+ `synthetic.py`（微调指标） |
| `program/UnityTemplate/` | `unity/` |
| `UI/mobile_app.html` 等 | `web/mobile/` |
| `UI/jiaohu*.py` | `legacy/ui/`（密钥移除，走环境变量） |
| `program/data_train/` | `data/samples/`（参考产物）/ `data/outputs/`（运行时） |

## 修复的历史问题

- 安检排队图南区“无引导/MADDPG”两条线数据与标签互换
- 奖励曲线 `find_latest_npy` 会误选任意 npy → 仅匹配 `*_env_*.npy`
- `fine/plot_finetune_metrics.py` 无 Agg 后端且 `plt.show()` 无头挂起
- `mlagents-envs>=1.1.0` 不可安装的版本 pin；未使用的 `gym` 依赖
- MATD3 `save_model` 缺目录创建；ReplayBuffer 样本不足时的报错信息
- Streamlit 原型偏好参数收集后未传入规划（现 `resolve_strategy` 真实生效）
- 移动端 sw.js 预缓存引用不存在的 `VR1.png` 导致安装失败、页面从未注册 SW
- manifest theme 色与页面 meta 不一致
- 无 .gitignore 导致 node_modules(187M)/pyc/大媒体/密钥入库（已治理，历史需单独清理）
- 新版高德 v3 换乘响应嵌套结构（`walking.steps`/`bus.buslines`）导致折线解析为空 → 新旧两代结构兼容解析（单测覆盖）
- 地图容器零尺寸时创建高德地图渲染空白 → 容器先显示后再建图
- 训练 TensorBoard 日志目录 `format` 多传一个 `args.algorithm` 导致 seed 错位 → 对齐 npy 产物命名
- 移动端引用未入库的 9.9MB `AR.gif` 导致克隆后破图 → 缺失时展示占位说明；`sw.js` 预缓存无引用的 1.8MB `VR.png` → 移除并升级缓存版本号
- 全仓 pyflakes 告警（未定义名 `Any`、未使用导入/死变量）→ 清零，CI 固化静态检查

## 性能优化

行人仿真 `step()` 已向量化：默认规模（44人×1000帧）**19.25x**、大规模（500人×300帧）**8.69x** 加速，统计行为等价。基准与报告见 `code_optimization/report.md`。

## 已知边界

- git 历史中仍可见已泄漏密钥与大文件（前向治理；彻底清史需 `git filter-repo` + force push，属破坏性操作需单独确认）
- `ml-agents` 镜像移至 `third_party/`（gitignore，本机参考）；训练环境需从源码安装 `mlagents-envs`
- 移动端 PWA 的登录/定位仍为演示语义（无账号体系），历史记录存 localStorage
