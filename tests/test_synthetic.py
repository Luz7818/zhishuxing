from __future__ import annotations

import numpy as np

from zhishuxing.analysis.synthetic import (
    generate_congestion_matrices,
    generate_finetune_metrics,
    generate_security_queue,
    generate_transfer_distribution,
)


def test_congestion_matrices_shape_and_determinism():
    a = generate_congestion_matrices(seed=20260319)
    b = generate_congestion_matrices(seed=20260319)
    zones, time_slots, before, after, delta = a
    assert len(zones) == 12 and len(time_slots) == 17
    assert before.shape == (12, 17) and after.shape == (12, 17)
    # 固定种子必须完全一致
    assert np.array_equal(before, b[2]) and np.array_equal(after, b[3])
    # delta = before - after
    assert np.allclose(delta, before - after)


def test_congestion_relief_direction():
    _, _, before, after, _ = generate_congestion_matrices(seed=20260319)
    # 训练后整体拥堵应下降
    assert float(after.mean()) < float(before.mean())
    # 拥堵强度裁剪下限
    assert float(before.min()) >= 20 and float(after.min()) >= 15


def test_transfer_distribution_convergence():
    iters, p50, p90, max_v = generate_transfer_distribution(seed=20260319, points=1000)
    assert len(iters) == 1000
    # 前 10% vs 后 10%：换乘时间应显著下降
    head = float(np.mean(p50[:100]))
    tail = float(np.mean(p50[-100:]))
    assert (head - tail) / head > 0.10
    # 约束关系：p90 >= p50 + 35, max >= p90 + 30
    assert bool(np.all(p90 >= p50 + 35 - 1e-3))
    assert bool(np.all(max_v >= p90 + 30 - 1e-3))


def test_transfer_distribution_seed_stable():
    a = generate_transfer_distribution(seed=7, points=100)
    b = generate_transfer_distribution(seed=7, points=100)
    assert np.array_equal(a[1], b[1])


def test_security_queue_guidance_reduces_queues():
    time_slots, no_guide, maddpg = generate_security_queue(seed=20260324)
    assert len(time_slots) == 193  # 6:00-22:00 每 5 分钟
    # 动态引导把主口压力分流到备用口：主口排队大幅下降，备用口承担分流
    assert float(np.mean(maddpg["queue_main"])) < float(np.mean(no_guide["queue_main"]))
    assert float(np.max(maddpg["queue_main"])) < float(np.max(no_guide["queue_main"]))
    assert float(np.mean(maddpg["queue_backup"])) > float(np.mean(no_guide["queue_backup"]))
    # 系统总排队（主+备）仍应下降
    total_no = float(np.mean(no_guide["queue_main"])) + float(np.mean(no_guide["queue_backup"]))
    total_maddpg = float(np.mean(maddpg["queue_main"])) + float(np.mean(maddpg["queue_backup"]))
    assert total_maddpg < total_no


def test_finetune_metrics_ranges():
    m = generate_finetune_metrics(epochs=36, seed=42)
    assert set(m) == {"epoch", "loss", "bleu4", "rouge1", "rougeL"}
    assert len(m["loss"]) == 36
    # loss 单调趋势下降（末值 < 首值），指标在 [0,1]
    assert float(m["loss"][-1]) < float(m["loss"][0])
    for key in ("bleu4", "rouge1", "rougeL"):
        assert float(m[key].min()) >= 0.0 and float(m[key].max()) <= 1.0
    # 末值 ROUGE-1 应高于 BLEU-4
    assert float(m["rouge1"][-1]) > float(m["bleu4"][-1])


def test_finetune_metrics_monotonic_mode():
    m = generate_finetune_metrics(epochs=20, seed=1, monotonic=True)
    for key in ("bleu4", "rouge1", "rougeL"):
        diffs = np.diff(m[key])
        assert bool(np.all(diffs >= -1e-9)), f"{key} 应单调不减"
