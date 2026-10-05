# 智枢星 Git 规范

> 用途：给提交本仓代码的人。分支、CI、入库边界与历史教训都在这里。

## 分支与提交策略

- 单 `main` 分支，推 `main` 触发 CI（ubuntu × Python 3.11/3.12，pyflakes → pytest）。
- **一批一提交**（Traffic 2026-10-02 事故的直接教训——大工作区攒批差点全损）；提交前门禁见
  [TESTING.md](TESTING.md)。

## 必须入库 / 禁止上传

| 判定 | 规则 |
|---|---|
| 必须入库 | `src/`、`configs/`、`data/transfer_kb/` 与 `data/samples/`、`data/real/` 的模板 CSV、`web/`、`unity/`（UPM 本地包）、`scripts/`、`tests/`、`code_optimization/` 的基准脚本与报告、`legacy/`、九件文档 |
| 禁止上传 | `data/outputs/`、`data/model/`、`data/runs/`（运行时产物）；`data/real/` 的真实 CSV（可能含敏感客流数据，只有 `*.example.csv` 模板入库）；`.env*`（`!.env.example` 例外必须在 `.env.*` 规则**之后**，gitignore 最后匹配赢——顺序换了它就静默不入库）；`third_party/`（上游镜像）；`*.egg-info/` |
| 密钥 | 任何真实密钥（高德/LLM）永不入库；2.2.0 已做全历史重写收口（`.git` 214 MB → 18 MB，全历史密钥字面量 0 残留），引用旧 SHA 的文档需复核 |

## CI

- `.github/workflows/ci.yml`：ubuntu-latest × Python 3.11/3.12，装 `.[dev]` + CPU torch，
  先 pyflakes 再 pytest（`MPLBACKEND=Agg`）。
- 不要为 `unity/` 写 CI（不是可构建工程，见 ARCHITECTURE）。

## commit message

- 风格沿用既有历史：`<type>(<scope>): 中文一句话` 或 `<type>: 中文一句话`
  （复核：`git log --oneline -10`）。
- tag：历史 tag 见 `git tag`；2.2.0 历史重写后旧 SHA 已变。
