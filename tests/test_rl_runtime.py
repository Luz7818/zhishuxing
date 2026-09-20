from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="RL 运行时测试需要 torch")

from zhishuxing.rl.networks import Actor  # noqa: E402
from zhishuxing.rl.runtime import MADDPGRuntime  # noqa: E402
from zhishuxing import config as cfg  # noqa: E402


@pytest.fixture()
def runtime(tmp_path):
    return MADDPGRuntime(model_dir=tmp_path, data_dir=cfg.paths.samples)


def test_status_without_checkpoints(runtime):
    status = runtime.get_status()
    assert status["policy_loaded"] is False
    assert status["policy_source"] == "未加载"
    assert status["checkpoint_count"] == 0
    assert status["torch_available"] is True


def test_load_policy_falls_back_when_empty(runtime):
    status = runtime.load_policy()
    assert status["policy_loaded"] is False
    assert status["policy_source"] == "启发式回退"
    assert "未找到" in status["load_error"]


def test_load_policy_from_real_checkpoints(runtime, tmp_path):
    """构造符合训练命名格式的 actor 权重，验证加载→推理回环。"""
    ckpt_dir = tmp_path / "integrated_hub_transfer"
    ckpt_dir.mkdir(parents=True)
    args = SimpleNamespace(
        obs_dim_n=[10, 10, 10],
        action_dim_n=[2, 2, 2],
        hidden_dim=64,
        max_action=1.0,
        use_orthogonal_init=False,
    )
    for agent_id in range(3):
        actor = Actor(args, agent_id)
        torch.save(actor.state_dict(), ckpt_dir / f"MADDPG_actor_number_1_step_500k_agent_{agent_id}.pth")

    status = runtime.load_policy(str(tmp_path))
    assert status["policy_loaded"] is True
    assert status["policy_source"] == "MADDPG"
    assert status["checkpoint_count"] == 3
    assert status["load_error"] == ""

    result = runtime.act([[0.5, 0.25, 1.0, 0.0, 0.2, 0.1, -0.3, 0.4, 0.0, -0.2]] * 3)
    assert result["policy_used"] == ["MADDPG"] * 3
    assert all(len(a) == 2 for a in result["actions"])
    assert all(abs(v) <= 1.0 for a in result["actions"] for v in a)


def test_act_obs_padding(runtime, tmp_path):
    """观测维度不足时补零、超出时截断，仍能输出动作。"""
    ckpt_dir = tmp_path / "integrated_hub_transfer"
    ckpt_dir.mkdir(parents=True)
    args = SimpleNamespace(
        obs_dim_n=[10],
        action_dim_n=[2],
        hidden_dim=32,
        max_action=1.0,
        use_orthogonal_init=False,
    )
    torch.save(Actor(args, 0).state_dict(), ckpt_dir / "MADDPG_actor_number_1_step_10k_agent_0.pth")
    runtime.load_policy(str(tmp_path))

    short = runtime.act([[0.1, 0.2, 0.3, 0.4]])          # 4 维 → 补零到 10
    long = runtime.act([[0.1] * 20])                      # 20 维 → 截断到 10
    assert len(short["actions"][0]) == 2
    assert len(long["actions"][0]) == 2


def test_reward_series_from_samples(runtime):
    payload = runtime.reward_series()
    assert payload["series"], "data/samples 中应有奖励 npy"
    first = payload["series"][0]
    assert first["points"] == 1000
    assert "x" in first and "y" in first
    assert len(first["x"]) <= 200  # 降采样


def test_render_reward_curve(runtime, tmp_path):
    result = runtime.render_reward_curve(str(tmp_path / "curve.png"))
    assert result["series_count"] >= 1
    assert Path(result["image"]).exists()
