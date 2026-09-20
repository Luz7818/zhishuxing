from __future__ import annotations

from zhishuxing import config as cfg


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert payload["navigation"]


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    # 欢迎页 + 应用壳 + 各功能分区(harness 式布局:tab-* 分区 / AI 面板 / 命令面板)
    for key in (
        "view-welcome",
        "tab-overview",
        "tab-navigation",
        "tab-rl",
        "tab-flow",
        "tab-plan",
        "tab-reports",
        "aipanel",
        "chatInput",
        "cmdk-mask",
        "theme-toggle",
        "navCanvas",
        "rewardCanvas",
        "btnRLSim",
        "btnRealPlan",
        "btnPanel",
        "btnReportsAll",
        "app.js",
    ):
        assert key in html
    # 高德底图配置注入与双通道容器
    assert "ZSX_CONFIG" in html
    assert "amapContainer" in html


def test_index_page_injects_amap_js_key(client, monkeypatch):
    monkeypatch.setenv("AMAP_JS_KEY", "test-js-key-123")
    html = client.get("/").data.decode("utf-8")
    assert "test-js-key-123" in html

    monkeypatch.delenv("AMAP_JS_KEY", raising=False)
    monkeypatch.delenv("AMAP_REST_KEY", raising=False)
    html = client.get("/").data.decode("utf-8")
    assert 'amapJsKey: ""' in html


def test_navigation_grid_endpoint(client):
    resp = client.get("/api/navigation/grid")
    assert resp.status_code == 200
    grid = resp.get_json()["data"]
    assert grid["width"] == 30 and grid["height"] == 16
    assert len(grid["blocked"]) == 25  # 含换乘通道竖墙(直梯/扶梯/楼梯门洞之外)
    assert "entry_a" in grid["landmarks"]
    # 设施语义层与地标中文标签(schema v2)
    assert set(grid["cell_tags"]) >= {"stairs", "escalator", "elevator", "crowd"}
    assert grid["cell_size_m"] == 30
    assert grid["landmark_labels"]["restroom_a"] == "卫生间A"


def test_scenarios_endpoint(client):
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    groups = resp.get_json()["data"]
    assert len(groups) == 3
    for group in groups:
        assert isinstance(group["start"], list) and len(group["start"]) == 2
        assert group["passengers"] > 0


def test_navigation_load_and_plan(client):
    load = client.post("/api/navigation/load", json={"file_path": str(cfg.paths.navigation_config)})
    assert load.status_code == 200
    data = load.get_json()["data"]
    assert data["width"] == 30 and data["blocked_count"] > 0

    plan = client.post("/api/navigation/plan", json={"start": [1, 2], "goal": [28, 12], "via": ["security"]})
    assert plan.status_code == 200
    route = plan.get_json()["data"]["route"]
    assert route[0] == [1, 2] and route[-1] == [28, 12]

    bad = client.post("/api/navigation/load", json={})
    assert bad.status_code == 400


def test_rl_endpoints(client):
    status = client.get("/api/rl/status")
    assert status.status_code == 200
    assert "policy_source" in status.get_json()["data"]

    act = client.post("/api/rl/act", json={"observations": [[0.5, 0.25, 1.0, 0.0, 0.2, 0.1, -0.3, 0.4, 0.0, -0.2]]})
    assert act.status_code == 200
    assert len(act.get_json()["data"]["actions"][0]) == 2

    act_empty = client.post("/api/rl/act")
    assert act_empty.status_code == 400

    simulate = client.post("/api/rl/simulate", json={"config": {"max_steps": 120, "agents_per_group": 3}})
    assert simulate.status_code == 200
    data = simulate.get_json()["data"]
    assert data["agents_total"] > 0 and data["agents_arrived"] > 0
    image_url = data["image_url"]
    assert client.get(image_url).status_code == 200


def test_rewards_endpoint(client):
    resp = client.get("/api/rl/rewards")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert "series_count" in data


def test_dashboard_endpoint(client, service):
    groups = service.default_groups()
    resp = client.post("/api/dashboard/run", json={"groups": groups, "title": "pytest"})
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["image_url"].startswith("/outputs/")
    assert client.get(data["image_url"]).status_code == 200

    missing = client.post("/api/dashboard/run", json={"groups": []})
    assert missing.status_code == 400


def test_llm_endpoints(client):
    load = client.post("/api/llm/load", json={"model_id": "test-model"})
    assert load.status_code == 200

    tune = client.post("/api/llm/fine_tune", json={"config": {"method": "LoRA", "epochs": 1}})
    assert tune.status_code == 200
    assert tune.get_json()["data"]["status"] in ("fine_tuned", "metadata_recorded")

    metrics = client.post("/api/llm/simulate_metrics", json={"config": {"epochs": 12, "seed": 7}})
    assert metrics.status_code == 200
    data = metrics.get_json()["data"]
    assert client.get(data["image_url"]).status_code == 200
    assert client.get(data["csv_url"]).status_code == 200


def test_plan_hub_engine(client):
    resp = client.post("/api/plan", json={"question": "从A口经主安检到地铁闸机", "engine": "hub"})
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["engine"] == "hub"
    assert data["route"][0] == [1, 2]  # entry_a
    assert data["simulation"]["agents_total"] > 0

    bad = client.post("/api/plan", json={"question": "从火星到月球", "engine": "hub"})
    assert bad.status_code == 400


def test_plan_hub_engine_consumes_preferences(client):
    # 自由文本偏好(note)必须被解析生效:路径避开楼梯、途经卫生间
    resp = client.post(
        "/api/plan",
        json={
            "question": "从A口到地铁闸机",
            "engine": "hub",
            "prefs": {"note": "优先直梯,先上趟卫生间"},
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    profile = data["profile"]
    assert "prefer_elevator" in profile["soft"]
    assert "need_restroom" in profile["hard"]
    details_text = " ".join(data["details"])
    assert "需求理解" in details_text
    assert any("直梯" in d or "卫生间" in d for d in data["details"])

    # 无偏好时也返回档案字段(空档案),前端渲染保持一致
    plain = client.post("/api/plan", json={"question": "从A口到地铁闸机", "engine": "hub"})
    assert plain.status_code == 200
    assert "profile" in plain.get_json()["data"]


def test_chat_endpoint_understands_preferences_and_plans(client):
    resp = client.post(
        "/api/chat",
        json={"message": "带老人行李多,优先直梯,去地铁前先上趟卫生间,从A口出发"},
    )
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["session_id"]
    assert data["engine"] == "hub"  # 起终点命中枢纽地标 → 枢纽内偏好规划
    assert "need_restroom" in data["profile"]["hard"]
    assert "prefer_elevator" in data["profile"]["soft"]
    assert "elderly" in data["profile"]["persona"]
    assert data["route"]["route"][0] == [1, 2]
    assert data["kb_refs"], "站内经验引用不应为空"
    assert "已理解您的需求" in data["reply"]

    # 第二轮:增量合并需求档案(会话状态)
    second = client.post(
        "/api/chat",
        json={"message": "我现在有点赶时间", "session_id": data["session_id"]},
    )
    assert second.status_code == 200
    second_data = second.get_json()["data"]
    assert second_data["profile"]["source"] == "merged"
    assert "rushed" in second_data["profile"]["persona"]
    assert second_data.get("od_incomplete") is True  # 无完整 OD 时引导补充


def test_chat_endpoint_resets_session(client):
    first = client.post("/api/chat", json={"message": "从A口到地铁闸机"}).get_json()["data"]
    reset = client.post("/api/chat/reset", json={"session_id": first["session_id"]})
    assert reset.status_code == 200
    assert reset.get_json()["data"]["reset"] is True

    # 重置后同 session_id 重新开始:档案不再累积
    fresh = client.post(
        "/api/chat", json={"message": "从A口到地铁闸机", "session_id": first["session_id"]}
    ).get_json()["data"]
    assert fresh["profile"]["source"] != "merged"


def test_chat_endpoint_requires_message(client):
    bad = client.post("/api/chat", json={})
    assert bad.status_code == 400


def test_chat_endpoint_amap_fallback_without_key(client, monkeypatch):
    # 市际 OD(未命中枢纽地标)走高德引擎;无 Key 时应答话术而非 500
    monkeypatch.delenv("AMAP_REST_KEY", raising=False)
    resp = client.post("/api/chat", json={"message": "从深圳北站到宝安机场,赶时间"})
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["engine"] == "amap"
    assert data.get("route_error")  # 无 Key 明确报错但对话不中断
    assert data["reply"]


def test_plan_amap_without_key(client, monkeypatch):
    monkeypatch.delenv("AMAP_REST_KEY", raising=False)
    resp = client.post("/api/plan", json={"question": "从深圳北站到宝安机场", "engine": "amap"})
    assert resp.status_code == 400
    assert "AMAP_REST_KEY" in resp.get_json()["error"]


def test_mobile_pages(client):
    page = client.get("/mobile")
    assert page.status_code == 200
    html = page.data.decode("utf-8")
    assert "serviceWorker" in html and "/api/chat" in html

    sw = client.get("/mobile/sw.js")
    assert sw.status_code == 200
    # 预缓存清单应包含必需资源,且不再包含已移出仓库的无引用大图
    sw_text = sw.data.decode("utf-8")
    assert "mobile_app.html" in sw_text and "icon-192.png" in sw_text
    assert "VR.png" not in sw_text

    manifest = client.get("/mobile/manifest.webmanifest")
    assert manifest.status_code == 200


def test_run_existing_reports(client):
    resp = client.post("/api/features/run_existing")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["count"] > 0
    reports = data["reports"]
    # 安检排队报告（含历史标签互换 bug 修复）应成功
    assert reports["security_queue_comparison"]["ok"] is True
    assert reports["congestion_heatmap"]["ok"] is True
