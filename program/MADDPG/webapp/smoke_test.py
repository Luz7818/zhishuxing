from __future__ import annotations

from pathlib import Path
import json
import sys

CURRENT_DIR = Path(__file__).resolve().parent
MADDPG_DIR = CURRENT_DIR.parent
if str(MADDPG_DIR) not in sys.path:
    sys.path.insert(0, str(MADDPG_DIR))

from webapp.app import create_app


def run_smoke_test() -> None:
    app = create_app()
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200

    nav_path = str(MADDPG_DIR / "zhishuxing" / "sample_navigation.json")
    load_nav = client.post("/api/navigation/load", json={"file_path": nav_path})
    assert load_nav.status_code == 200, load_nav.data

    plan = client.post(
        "/api/navigation/plan",
        json={"start": [1, 2], "goal": [28, 12], "via": ["security"]},
    )
    assert plan.status_code == 200, plan.data

    groups = [
        {
            "name": "A口进站->地铁",
            "start": [1, 2],
            "goal": [28, 12],
            "via_landmarks": ["security"],
            "release_time": 0,
            "passengers": 15,
        },
        {
            "name": "B口进站->高铁",
            "start": [1, 13],
            "goal": [28, 3],
            "via_landmarks": ["security_backup"],
            "release_time": 8,
            "passengers": 18,
        },
    ]
    run_dash = client.post("/api/dashboard/run", json={"groups": groups, "title": "smoke-test"})
    assert run_dash.status_code == 200, run_dash.data

    payload = json.loads(run_dash.data.decode("utf-8"))
    image_url = payload["data"]["image_url"]
    image_name = image_url.split("/")[-1]

    image_resp = client.get(f"/outputs/{image_name}")
    assert image_resp.status_code == 200

    sim_metrics = client.post(
        "/api/llm/simulate_metrics",
        json={"config": {"epochs": 12, "seed": 7, "loss_noise": 0.03, "metric_noise": 0.01}},
    )
    assert sim_metrics.status_code == 200, sim_metrics.data

    sim_payload = json.loads(sim_metrics.data.decode("utf-8"))
    sim_image_url = sim_payload["data"]["image_url"]
    sim_csv_url = sim_payload["data"]["csv_url"]

    sim_image_name = sim_image_url.split("/")[-1]
    sim_csv_name = sim_csv_url.split("/")[-1]

    sim_image_resp = client.get(f"/outputs/{sim_image_name}")
    assert sim_image_resp.status_code == 200

    sim_csv_resp = client.get(f"/outputs/{sim_csv_name}")
    assert sim_csv_resp.status_code == 200

    rl_status = client.get("/api/rl/status")
    assert rl_status.status_code == 200, rl_status.data
    rl_payload = json.loads(rl_status.data.decode("utf-8"))
    assert "policy_source" in rl_payload["data"]

    rl_act = client.post(
        "/api/rl/act",
        json={"observations": [[0.5, 0.25, 1.0, 0.0, 0.2, 0.1, -0.3, 0.4, 0.0, -0.2]]},
    )
    assert rl_act.status_code == 200, rl_act.data
    act_payload = json.loads(rl_act.data.decode("utf-8"))
    assert len(act_payload["data"]["actions"][0]) == 2

    rl_rewards = client.get("/api/rl/rewards")
    assert rl_rewards.status_code == 200, rl_rewards.data

    rl_sim = client.post(
        "/api/rl/simulate",
        json={"groups": groups, "config": {"max_steps": 120, "agents_per_group": 3}},
    )
    assert rl_sim.status_code == 200, rl_sim.data
    sim_payload = json.loads(rl_sim.data.decode("utf-8"))
    assert sim_payload["data"]["agents_total"] > 0

    rl_sim_image_url = sim_payload["data"]["image_url"]
    rl_sim_image_resp = client.get(f"/outputs/{rl_sim_image_url.split('/')[-1]}")
    assert rl_sim_image_resp.status_code == 200

    print("Web smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
