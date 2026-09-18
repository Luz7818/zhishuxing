from __future__ import annotations

import json
from pathlib import Path

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
    # 应用外壳与关键交互元素
    for key in (
        "view-overview",
        "view-navigation",
        "view-rl",
        "view-flow",
        "view-plan",
        "view-llm",
        "view-reports",
        "navCanvas",
        "rewardCanvas",
        "btnRLSim",
        "btnRealPlan",
        "btnPanel",
        "btnReportsAll",
        "app.js",
    ):
        assert key in html


def test_navigation_grid_endpoint(client):
    resp = client.get("/api/navigation/grid")
    assert resp.status_code == 200
    grid = resp.get_json()["data"]
    assert grid["width"] == 30 and grid["height"] == 16
    assert len(grid["blocked"]) == 20
    assert "entry_a" in grid["landmarks"]


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


def test_plan_amap_without_key(client, monkeypatch):
    monkeypatch.delenv("AMAP_REST_KEY", raising=False)
    resp = client.post("/api/plan", json={"question": "从深圳北站到宝安机场", "engine": "amap"})
    assert resp.status_code == 400
    assert "AMAP_REST_KEY" in resp.get_json()["error"]


def test_mobile_pages(client):
    page = client.get("/mobile")
    assert page.status_code == 200
    html = page.data.decode("utf-8")
    assert "serviceWorker" in html and "/api/plan" in html

    sw = client.get("/mobile/sw.js")
    assert sw.status_code == 200
    assert "VR.png" in sw.data.decode("utf-8")

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
