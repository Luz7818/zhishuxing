# data/real/ —— 真实客流数据约定目录(2.2.0)

把真实客流数据按下表命名放进本目录,`zhishuxing analyze` 会自动发现并使用;
**文件缺失不是错误** —— 对应报告自动回退合成数据(固定种子),并在输出里标注来源。

| 文件名 | 消费的报告 | 格式 |
| --- | --- | --- |
| `congestion_before.csv` + `congestion_after.csv` | heatmap(拥堵热力三联图) | 矩阵 CSV:首列区域名、首行时段标签,两个文件的行列标签必须一致 |
| `transfer_summary.csv` | transfer(换乘时间分布)+ efficiency(场景效率对比) | 列:`iteration,p50,p90,max`(iteration=训练迭代号) |

来源可追溯:使用真实数据时,输出行标注 `来源:真实数据:<文件名>`,
heatmap/transfer 的图题也会带同一标注;合成运行标注 `来源:合成(固定种子)`。

## 格式细节

- 编码 UTF-8(带 BOM 亦可,Excel 直接另存即可);逗号分隔。
- 矩阵 CSV 数值 = 该区域该时段的拥堵指数/客流强度(与你的业务口径一致即可,
  报告只做相对对比与归一化展示)。
- `transfer_summary.csv` 的行数建议 ≥ 30(太少时效率报告会以 `min_rows` 拒绝)。
- 本目录两个 `*.example.csv` 是格式模板:复制、去掉 `.example`、填入真实值后即生效。
- 数据含敏感信息时不要提交本目录(`.gitignore` 已排除除 README/模板外的文件,
  以提交前 `git status` 为准)。

## 判据(改自版本路线 2.2)

放入真实文件后运行 `zhishuxing analyze --report all`:输出应逐行标注
`来源:真实数据:...`(heatmap/transfer/efficiency)且 7 行全部 OK;
清空本目录后同一命令回退 `来源:合成(固定种子)`。
