# 智枢星 Web 部署说明

本模块将现有“智枢星”能力封装为可部署网页，支持：
- 导航图加载接口
- 路径规划接口
- 大模型加载与微调接口
- 动态客流可视化面板生成
- 运行既有功能脚本（奖励曲线、热力图、动图）

## 1. 目录
- `MADDPG/webapp/app.py`：Flask 后端与 API
- `MADDPG/webapp/templates/index.html`：网页前端
- `MADDPG/webapp/static/*`：静态资源
- `MADDPG/run_zhishuxing_web.py`：服务启动脚本
- `MADDPG/webapp/smoke_test.py`：冒烟测试

## 2. 启动（开发）
在工作区根目录执行：

```powershell
(E:\Software\Anaconda\shell\condabin\conda-hook.ps1)
conda activate my
python .\MADDPG\run_zhishuxing_web.py --host 0.0.0.0 --port 7860
```

浏览器打开：`http://127.0.0.1:7860`

## 3. 启动（生产部署）
使用 `waitress` 生产托管：

```powershell
(E:\Software\Anaconda\shell\condabin\conda-hook.ps1)
conda activate my
python .\MADDPG\run_zhishuxing_web.py --host 0.0.0.0 --port 7860 --production
```

## 4. API 清单
- `GET /health`
- `POST /api/navigation/load`
- `POST /api/navigation/plan`
- `POST /api/llm/load`
- `POST /api/llm/fine_tune`
- `POST /api/dashboard/run`
- `POST /api/features/run_existing`
- `GET /outputs/<filename>`

## 5. 冒烟测试

```powershell
(E:\Software\Anaconda\shell\condabin\conda-hook.ps1)
conda activate my
python .\MADDPG\webapp\smoke_test.py
```

通过后会输出：`Web smoke test passed.`
