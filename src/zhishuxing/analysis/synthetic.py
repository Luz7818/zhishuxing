"""合成数据生成器：全仓库唯一来源（数值行为与历史脚本逐行一致）。

包括：深圳北站 12区域×17时段 拥堵矩阵、换乘时间 P50/P90/Max 分布、
安检排队微观仿真、LLM 微调指标曲线。固定默认种子保证可复现。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------- 拥堵矩阵（原 plot_congestion_heatmap.generate_synthetic_data）

def default_zones() -> List[str]:
    return [
        "东广场进站口",
        "西广场进站口",
        "地铁4号线换乘通道",
        "地铁5号线换乘通道",
        "高铁出站口A",
        "高铁出站口B",
        "北1检票闸机群",
        "北2检票闸机群",
        "南1检票闸机群",
        "南2检票闸机群",
        "出租车上客区",
        "网约车接驳区",
    ]


def default_time_slots() -> List[str]:
    return [
        "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00",
        "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00",
    ]


def generate_congestion_matrices(seed=20260319):
    rng = np.random.default_rng(seed)
    zones = default_zones()
    time_slots = default_time_slots()

    z, t = len(zones), len(time_slots)
    before = np.zeros((z, t), dtype=np.float32)

    # 基础客流强度（不同区域常态差异）
    zone_base = np.array([
        95, 90, 110, 120, 85, 80, 100, 96, 92, 88, 78, 75
    ], dtype=np.float32)

    # 早晚高峰 + 午间次峰
    peak_morning = np.exp(-0.5 * ((np.arange(t) - 2.2) / 1.2) ** 2)   # 8点附近
    peak_evening = np.exp(-0.5 * ((np.arange(t) - 12.2) / 1.3) ** 2)  # 18点附近
    peak_midday = np.exp(-0.5 * ((np.arange(t) - 6.3) / 1.6) ** 2)    # 12点附近

    # 训练前：拥堵更严重，热点更集中
    for i in range(z):
        congestion_factor = 0.9 + 0.3 * rng.random()
        temporal = (
            0.55 * peak_morning +
            0.45 * peak_evening +
            0.20 * peak_midday
        )

        # 换乘通道与闸机群更拥堵
        hotspot_bonus = 0.0
        if i in [2, 3, 6, 7, 8, 9]:
            hotspot_bonus = 0.22
        if i in [0, 1]:
            hotspot_bonus += 0.12

        curve = zone_base[i] * (0.70 + congestion_factor * temporal + hotspot_bonus)
        noise = rng.normal(0, zone_base[i] * 0.06, size=t)
        before[i] = np.clip(curve + noise, 20, None)

    # 训练后：整体拥堵下降 + 高峰削峰填谷（但保留自然波动）
    after = before.copy()
    relief = np.zeros_like(before)

    for i in range(z):
        if i in [2, 3, 6, 7, 8, 9]:
            base_relief = 0.24
        elif i in [0, 1, 4, 5]:
            base_relief = 0.16
        else:
            base_relief = 0.12

        # 高峰时段减幅更大
        peak_weight = 0.6 * peak_morning + 0.7 * peak_evening + 0.25 * peak_midday
        relief_ratio = base_relief + 0.22 * peak_weight
        relief_ratio += rng.normal(0, 0.02, size=t)
        relief_ratio = np.clip(relief_ratio, 0.05, 0.55)

        relief[i] = before[i] * relief_ratio

        # 同时加入“分流后波动”
        redistribution_noise = rng.normal(0, zone_base[i] * 0.04, size=t)
        after[i] = np.clip(before[i] - relief[i] + redistribution_noise, 15, None)

    # 差值（正值表示缓解）
    delta = before - after

    return zones, time_slots, before, after, delta


# ---------------------------------------------------------------- 换乘时间分布（原 plot_transfer_time_distribution.generate_synthetic_distribution）

def generate_transfer_distribution(seed=20260319, points=1000):
    rng = np.random.default_rng(seed)
    iterations = np.arange(1, points + 1, dtype=np.int32)

    # 目标：1000次迭代、总体降幅约20%，最大值降幅略高，后期进入平台期
    progress = 1.0 - np.exp(-iterations / 240.0)  # 逐步收敛

    p50_start, p50_end = 520.0, 398.0   # 目标降幅约20%上下
    p90_start, p90_end = 760.0, 578.0   # 目标降幅约20%上下
    max_start, max_end = 980.0, 650.0   # 最大值降幅略高

    p50_base = p50_start - (p50_start - p50_end) * progress
    p90_base = p90_start - (p90_start - p90_end) * progress
    max_base = max_start - (max_start - max_end) * progress

    # 平台期：约650次后仅小幅变化
    plateau_factor = np.clip((iterations - 650) / 300.0, 0.0, 1.0)
    p50_base = p50_base * (1.0 - 0.015 * plateau_factor)
    p90_base = p90_base * (1.0 - 0.012 * plateau_factor)
    max_base = max_base * (1.0 - 0.010 * plateau_factor)

    # 非平稳噪声：前期大、后期中等，保留真实波动
    sigma = 20.0 * np.exp(-iterations / 360.0) + 6.8
    p50 = p50_base + rng.normal(0, sigma * 0.55)
    p90 = p90_base + rng.normal(0, sigma * 0.75)

    # 最大值加入独立AR漂移，避免与P90同步
    max_ar = np.zeros(points, dtype=np.float32)
    for i in range(1, points):
        max_ar[i] = 0.88 * max_ar[i - 1] + rng.normal(0, float(sigma[i]) * 0.42)
    max_values = max_base + rng.normal(0, sigma * 0.95) + max_ar

    # 阶段性退化/突发拥堵：三条线峰谷位置和幅度略有差异
    base_centers = np.array([140, 280, 430, 610, 790, 920], dtype=np.float32)
    p50_centers = base_centers + np.array([-6, -3, 0, 2, 4, -2], dtype=np.float32)
    p90_centers = base_centers + np.array([0, 3, -2, 4, 1, 3], dtype=np.float32)
    max_centers = np.array([110, 255, 392, 562, 742, 878], dtype=np.float32)

    for center50, center90, center_max in zip(p50_centers, p90_centers, max_centers):
        width50 = rng.uniform(10, 17)
        width90 = rng.uniform(11, 18)
        width_max = rng.uniform(12, 19)

        amp50 = rng.uniform(8, 18)
        amp90 = rng.uniform(11, 22)
        amp_max = rng.uniform(12, 26)

        bump50 = amp50 * np.exp(-0.5 * ((iterations - center50) / width50) ** 2)
        bump90 = amp90 * np.exp(-0.5 * ((iterations - center90) / width90) ** 2)
        bump_max = amp_max * np.exp(-0.5 * ((iterations - center_max) / width_max) ** 2)

        p50 += bump50
        p90 += bump90
        max_values += bump_max

    # 最大值专属谷值事件（极值回落），与P90进一步错位
    max_dip_centers = np.array([205, 485, 705, 948], dtype=np.float32)
    for center in max_dip_centers:
        dip_width = rng.uniform(8, 15)
        dip_amp = rng.uniform(10, 20)
        dip = dip_amp * np.exp(-0.5 * ((iterations - center) / dip_width) ** 2)
        max_values -= dip

    # 轻微相位差振荡，避免三条线峰谷过于同步
    p50 += 3.0 * np.sin(iterations / 46.0 + 0.15)
    p90 += 4.0 * np.sin(iterations / 44.0 + 0.55)
    max_values += 6.5 * np.sin(iterations / 37.0 + 1.15) + 2.8 * np.sin(iterations / 23.0 + 0.4)

    # 稀疏异常点：分别采样，位置略错开
    pool = np.arange(50, points - 20)
    idx50 = rng.choice(pool, size=max(14, points // 70), replace=False)
    idx90 = rng.choice(pool, size=max(16, points // 65), replace=False)
    idx_max = rng.choice(pool, size=max(24, points // 52), replace=False)

    p50[idx50] += rng.uniform(4, 16, size=idx50.shape[0])
    p90[idx90] += rng.uniform(8, 24, size=idx90.shape[0])
    max_values[idx_max] += rng.uniform(10, 42, size=idx_max.shape[0])
    idx_max_down = rng.choice(pool, size=max(14, points // 80), replace=False)
    max_values[idx_max_down] -= rng.uniform(8, 26, size=idx_max_down.shape[0])

    # 按需求整体上移
    p50 += 150.0
    p90 += 80.0

    p50 = np.clip(p50, 80, None)
    p90 = np.maximum(p90, p50 + 35)
    dynamic_gap = 42.0 + 9.0 * np.sin(iterations / 57.0 + 0.2) + rng.normal(0, 2.0, size=points)
    dynamic_gap = np.clip(dynamic_gap, 30.0, 62.0)
    max_values = np.maximum(max_values, p90 + dynamic_gap)

    return iterations, p50.astype(np.float32), p90.astype(np.float32), max_values.astype(np.float32)


# ---------------------------------------------------------------- 安检排队（原 plot_security_queue_comparison）

def build_time_slots(start_hour: int = 6, end_hour: int = 22, step_min: int = 5):
    slots = []
    minutes = []
    for h in range(start_hour, end_hour + 1):
        for m in range(0, 60, step_min):
            if h == end_hour and m > 0:
                break
            slots.append(f"{h:02d}:{m:02d}")
            minutes.append((h - start_hour) * 60 + m)
    return slots, np.asarray(minutes, dtype=np.float32)


def _gaussian(x: np.ndarray, center: float, sigma: float):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def generate_demand_and_capacity(minutes: np.ndarray, seed: int):
    """单位：人/5分钟。按“小规模排队（数量级约10）”进行校准。"""
    rng = np.random.default_rng(seed)

    morning_peak = _gaussian(minutes, center=130, sigma=45)
    midday_peak = _gaussian(minutes, center=360, sigma=55)
    evening_peak = _gaussian(minutes, center=720, sigma=65)

    base_arrival = 6.8 + 1.0 * np.sin(minutes / 38.0)
    arrival_rate = (
        base_arrival
        + 3.2 * morning_peak
        + 1.8 * midday_peak
        + 4.0 * evening_peak
        + rng.normal(0, 0.5, size=minutes.shape[0])
    )

    # 高峰时增配主安检资源，服务能力整体接近需求，形成小幅波动排队
    main_capacity = (
        7.0
        + 1.6 * morning_peak
        + 0.9 * midday_peak
        + 1.9 * evening_peak
        + 0.4 * np.sin(minutes / 55.0 + 0.5)
    )

    # 备用口基础开口较少，动态引导可临时增开
    backup_capacity = 2.8 + 0.4 * np.sin(minutes / 63.0 + 1.3)

    arrival_rate = np.clip(arrival_rate, 3.5, None)
    main_capacity = np.clip(main_capacity, 5.5, None)
    backup_capacity = np.clip(backup_capacity, 2.0, None)

    return arrival_rate, main_capacity, backup_capacity


def _sigmoid(x: float):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_security_strategy(strategy: str, arrival_rate: np.ndarray, main_capacity: np.ndarray, backup_capacity: np.ndarray, seed: int):
    """逐步 Poisson 到达 + Gaussian 容量抖动的排队递推。

    “无引导”：固定低分流比例，主口拥挤时小幅上升；
    “MADDPG动态引导”：按排队压力 sigmoid 分流，高压时临时增开通道。
    """
    rng = np.random.default_rng(seed)
    n = arrival_rate.shape[0]

    q_main = 0.0
    q_backup = 0.0

    queue_main = np.zeros(n, dtype=np.float32)
    queue_backup = np.zeros(n, dtype=np.float32)

    arrival_main_arr = np.zeros(n, dtype=np.float32)
    arrival_backup_arr = np.zeros(n, dtype=np.float32)
    served_main_arr = np.zeros(n, dtype=np.float32)
    served_backup_arr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        total_arrival = float(rng.poisson(max(arrival_rate[i], 0.0)))
        cap_main = float(max(rng.normal(main_capacity[i], 0.55), 0.0))
        cap_backup = float(max(rng.normal(backup_capacity[i], 0.35), 0.0))

        if strategy == "无引导":
            backup_share = 0.08
            if q_main > 9:
                backup_share = 0.12
            if q_main > 16:
                backup_share = 0.18
        elif strategy == "MADDPG动态引导":
            pressure = (q_main - 0.9 * q_backup) / 6.0
            backup_share = 0.20 + 0.34 * _sigmoid(pressure)
            backup_share = float(np.clip(backup_share, 0.18, 0.58))

            if q_main > 7:
                cap_backup += 0.8
            if q_main > 12:
                cap_backup += 1.1
                cap_main += 0.5
            if q_backup > 10:
                backup_share = max(0.20, backup_share - 0.08)
        else:
            raise ValueError(f"未知策略: {strategy}")

        arr_backup = float(rng.binomial(int(total_arrival), backup_share))
        arr_main = total_arrival - arr_backup

        served_main = min(q_main + arr_main, cap_main)
        served_backup = min(q_backup + arr_backup, cap_backup)

        q_main = max(0.0, q_main + arr_main - served_main)
        q_backup = max(0.0, q_backup + arr_backup - served_backup)

        queue_main[i] = q_main
        queue_backup[i] = q_backup
        arrival_main_arr[i] = arr_main
        arrival_backup_arr[i] = arr_backup
        served_main_arr[i] = served_main
        served_backup_arr[i] = served_backup

    return {
        "queue_main": queue_main,
        "queue_backup": queue_backup,
        "arrival_main": arrival_main_arr,
        "arrival_backup": arrival_backup_arr,
        "served_main": served_main_arr,
        "served_backup": served_backup_arr,
    }


def generate_security_queue(seed: int = 20260324) -> Tuple[List[str], Dict, Dict]:
    """默认安检排队对比实验：主数据 seed，两策略分别用 seed+101 / seed+202。"""
    time_slots, minutes = build_time_slots()
    arrival_rate, main_capacity, backup_capacity = generate_demand_and_capacity(minutes, seed=seed)
    no_guide = simulate_security_strategy(
        strategy="无引导",
        arrival_rate=arrival_rate,
        main_capacity=main_capacity,
        backup_capacity=backup_capacity,
        seed=seed + 101,
    )
    maddpg = simulate_security_strategy(
        strategy="MADDPG动态引导",
        arrival_rate=arrival_rate,
        main_capacity=main_capacity,
        backup_capacity=backup_capacity,
        seed=seed + 202,
    )
    return time_slots, no_guide, maddpg


# ---------------------------------------------------------------- LLM 微调指标（历史两套发散实现合并为单一来源）

def _correlated_noise(rng: np.random.Generator, size: int, scale: float = 1.0):
    """平滑相关抖动，模拟真实训练波动。"""
    white = rng.normal(0, 1, size=size)
    smooth = np.convolve(white, np.array([0.2, 0.6, 0.2]), mode="same")
    return smooth * scale


def generate_finetune_metrics(
    epochs: int = 36,
    seed: int = 42,
    loss_noise: float = 0.018,
    metric_noise: float = 0.006,
    monotonic: bool = False,
) -> Dict[str, np.ndarray]:
    """模拟 LoRA 微调的 loss / BLEU-4 / ROUGE-1 / ROUGE-L 曲线。

    monotonic=True 时对上升型指标做单调累加（Web 端展示更平稳），
    False 时保留 correlated-noise 的真实感回撤（历史 fine/ 版本行为）。
    """
    rng = np.random.default_rng(seed)
    x = np.arange(1, epochs + 1)
    p = (x - 1) / max(epochs - 1, 1)

    # Loss：快速下降 + 学习率 bump + 相关噪声
    loss_base = 1.7 * np.exp(-4.3 * p) + 0.23
    loss_wave = 0.035 * np.sin(6.2 * np.pi * p) * np.exp(-1.2 * p)
    lr_bump_1 = 0.06 * np.exp(-((p - 0.28) ** 2) / 0.002)
    lr_bump_2 = 0.035 * np.exp(-((p - 0.62) ** 2) / 0.003)
    loss_rand = _correlated_noise(rng, epochs, scale=loss_noise) + rng.normal(0, loss_noise * 0.45, size=epochs)
    loss = np.clip(loss_base + loss_wave + lr_bump_1 + lr_bump_2 + loss_rand, 0.16, None)

    def rising(trend: np.ndarray, wave: np.ndarray, dip: np.ndarray, noise_scale: float) -> np.ndarray:
        curve = trend + wave + dip + _correlated_noise(rng, epochs, scale=noise_scale) + rng.normal(0, noise_scale * 0.6, size=epochs)
        curve = np.clip(curve, 0.0, 1.0)
        if monotonic:
            curve = np.maximum.accumulate(curve)
        return curve

    bleu4 = rising(
        0.09 + 0.35 * (1 - np.exp(-3.8 * p)),
        0.012 * np.sin(5.1 * np.pi * p) * np.exp(-0.7 * p),
        -0.018 * np.exp(-((p - 0.42) ** 2) / 0.0025),
        metric_noise,
    )
    rouge1 = rising(
        0.28 + 0.46 * (1 - np.exp(-4.0 * p)),
        0.01 * np.sin(4.5 * np.pi * p) * np.exp(-0.8 * p),
        0.0,
        metric_noise + 0.001,
    )

    rouge_l_gap = 0.055 + 0.01 * (1 - np.exp(-2.3 * p))
    rouge_l = np.clip(rouge1 - rouge_l_gap + _correlated_noise(rng, epochs, scale=metric_noise), 0.0, 1.0)
    if monotonic:
        rouge_l = np.maximum.accumulate(rouge_l)

    return {"epoch": x, "loss": loss, "bleu4": bleu4, "rouge1": rouge1, "rougeL": rouge_l}
