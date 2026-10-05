# 智枢星 代码风格

> 用途：给改本仓代码的人与 AI。静态检查是 pyflakes（0 告警门禁）；约定来自四条设计原则与
> 既有实践（详见 `docs/ARCHITECTURE.md`）。

## 代码风格

- Python ≥3.10；分层：`core/`（领域内核，不依赖 torch 与网络）→ `rl/`（算法与环境）→
  `llm/`（对话链路）→ `planning/`（高德）→ `analysis/`（报告与绘图）→ `webapp/`（Flask）。
- 单一来源纪律：字体配置、`moving_average`、CSV IO、合成数据生成器只在 `analysis/`；
  别处再写一份就是回归（已修过的历史问题，见 ARCHITECTURE）。

## 命名与结构约定

- 产物命名是契约：`{algorithm}_env_{env_name}_number_{n}_seed_{s}.npy` 与
  `{algorithm}_actor_number_{n}_step_{k}k_agent_{id}.pth`，读写双方都依赖，不能改。
- 配置键走 `SETTINGS` 注册表；退出码与降级状态写进注册表（`degrades_to`）。
- 可选依赖隔离：`torch`/`mlagents_envs`/`openai` 全部懒加载或可选组，core 对它们无硬依赖。

## 错误处理与已知陷阱

- 在线能力必须可降级：LLM 失败 → 确定性模板（`real_adapter_error` 如实上报），不许 500 糊脸。
- 对外输出一律掩码（`mask_value()`），任何新接口不得回显明文密钥。
- 路径只由 `config.paths` 推导，禁止相对 CWD 的 `Path("data/...")` 写法。
- `matplotlib.use("Agg")` 在 `analysis/plotting.py` 顶层——绘图模块必须先 import 它再 pyplot。
- Windows 控制台 cp936：中文输出先 `set PYTHONIOENCODING=utf-8`。
- 全链路都是请求/响应，没有 SSE/WebSocket（`/api/chat` 一次返回完整回答）。
