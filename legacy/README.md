# legacy/ —— 历史原型归档

> 用途：说明这里留着什么、为什么留着、以及为什么它不参与测试与静态检查范围之外的任何门禁。

`legacy/ui/` 是 v2 重构前的 **Streamlit 单页原型**。它的逻辑已经逐段迁进
`src/zhishuxing/planning/amap.py`（OD 提取、地理编码、换乘规划、折线解析）与
`src/zhishuxing/webapp/`（界面与交互），这里只保留原始形态用于对照"迁移前是什么样"。

## 文件清单

| 文件 | 干什么 | 是否入库 |
|---|---|---|
| `ui/jiaohu.py` | 换乘引导原型主页：OD 提取、高德地理编码与公交换乘、折线渲染、偏好汇总 | ✓ |
| `ui/jiaohu_sj.py` | 第二个原型页（深色调 CSS 与状态机独立实现），与上页共享 `config_direct` | ✓ |
| `ui/config_direct.py` | 早期集中放密钥与端点的配置模块 | ✗ 被 `.gitignore` 的 `config_direct.py` 规则排除 |
| `ui/mobile_app.zip` | 旧版移动端整包，11.9 MB | ✗ 被 `*.zip` 排除 |
| `ui/*.mp4`（演示录屏） | 参赛演示录屏，50.8 MB | ✗ 被 `*.mp4` 排除 |

也就是说：`git ls-files legacy/` 只有 2 个 `.py`，另外三个只存在于本机。

## 为什么它不参与测试

1. **依赖没有声明**：这两个文件 `import streamlit`，而 streamlit 不在 `pyproject.toml`
   的任何依赖组里（复核：`python -c "import importlib.metadata as m; print(m.requires('zhishuxing'))"`）。
   给它们写测试等于要先把 streamlit 变成运行时依赖。
2. **函数与新实现重名但语义不同**：`extract_od_locally()`、`geocode()`、`transit_route()`、
   `polyline_from_transit()` 在 `planning/amap.py` 里是各自的另一份。同名函数有两份，
   一旦收集进测试就会出现"测的是哪一份"说不清的情况。
3. 它是**归档**，不是待修的实现。发现它有 bug 时的正确动作是确认新实现有没有同样的问题，
   而不是改这里。

它同样不在 `python -m pyflakes src/ scripts/ tests/` 的门禁范围内 —— 命令里没写 `legacy/`。
这是有意的：`jiaohu_sj.py` 单独就有 10 条 pyflakes 告警（7 个未使用导入、3 个无占位符的
f-string；复核：`python -m pyflakes legacy/ui/jiaohu_sj.py`），
纳入检查只会逼人为归档代码做无意义改动。

## 关于密钥的一件事

`ui/config_direct.py` 的**当前本机内容**已经没有密钥字面量（复核：
`grep -cE '"[0-9a-f]{32}"|sk-[A-Za-z0-9]{20,}' legacy/ui/config_direct.py` 输出 `0`）。
但它的**前身** `UI/config_direct.py` 在首次提交 `93656291` 里含 3 处密钥形状的字面量，
而 git 历史没有被重写（复核：
`git show 93656291:UI/config_direct.py | grep -cE '"[0-9a-f]{32}"|sk-[A-Za-z0-9]{20,}'`）。
`.gitignore` 只能阻止后续提交，不能抹掉历史。已泄漏的密钥要在平台侧作废重发，
清理历史是另一件事（`git filter-repo` + 强推，属破坏性操作）。

## 别动

- 不要"顺手清理"这两个 `.py`：它们是迁移映射的对照物，删掉后 `AGENTS.md` 里
  「旧 → 新」那条表就无从核对。
- 不要把这里的代码 import 回 `src/`：需要那段逻辑就照 `planning/amap.py` 的写法重写并补测试。
- 不要为了通过某个扫描工具把 `config_direct.py` 加进跟踪，它被排除正是为了断掉这条路径。
