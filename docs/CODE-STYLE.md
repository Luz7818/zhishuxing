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

## 前端动效规范(2.12.0 动效升级批定型)

适用面:控制台(templates/index.html + static/style.css + app.js)与移动端 PWA
(web/mobile/mobile_app.html);令牌事实来源是两份文件的 `:root` 注释,本章只写口径。

- **语义判据(一句话)**:每个动效必须能翻译成交通语义——到站、开门、发车、沿线行进;
  翻译不出来的就不做。悬停 = 到站微抬升(translateY(-1px)+shadow-sm,transform `--dur-1`、
  阴影 `--dur-2`);面板/侧栏/弹层开合 = 开门关门(退场比进场快约 20%);页面切换 = 旧页
  发车离站、新页到站定格;进度与光效 = 沿动线行进(既有 hubTravel/lineFlow 母题)。
- **三缓动分工**:进场/出现/到站 = `--ease-out`;在屏内移动(滑动药丸、标签滑轨) =
  `--ease-in-out`;几何开合(侧栏/抽屉/页面 push)= `--ease-drawer`。时长一律取四档
  `--dur-*` 阶梯,全站交互动画 <300ms;Rare 场景(欢迎页编排、悬浮球、空态描线、
  图片加载淡入)保留自有节奏并就地注释豁免原因。
- **页面切换口径(两端各有空间隐喻,定为规范不再另起)**:控制台 = 内容层纵向到站
  (新页子元素 `pageIn` stagger `--dur-3`)+ 旧页发车淡出(`.leaving`,opacity+上行
  `--dur-2`,absolute 豁免回流);移动端 = 页面层方向性 push(±64px,`pageInR/L` +
  `pageOutL/R` `--dur-4`)。退场一律快于进场约 20%,一律不从 scale(0) 起。
- **主题切换规范(定为规范不再另起)**:桌面 = `theme-fading` 全站颜色过渡 `--dur-3`
  (不用页面级 View Transition);移动端 = 优先 `vtCircle` 圆形揭示 .45s,不支持/
  reduced-motion 回退 `theme-fading` .26s(该 .26s 是主题过渡规范值,不算未归阶梯的
  散数)。两处都必须先查 `prefers-reduced-motion`。
- **不回潮红线**:2.10.0 已否决滚动耦合 reveal(进过视口即定格,git c20ccda);
  主按钮保持 2.11.1 纯色口径不回升渐变;hover 明度反馈单一机制 = `--primary-h`/
  `--surface-2` 背景换色,不引入 brightness。
- **scopeNote(下一批待收)**:移动端三处滑动药丸(mode/seg/nav)曲线已统一
  `--ease-in-out`,但 .3s/.34s 时长仍硬编码,未归 `--dur-*` 阶梯(控制台 nav-pill 已是
  `--dur-3`);历史 sheet 出入场 .24s~.32s 与 toast 时长两端不一致,同样留待下一批。
