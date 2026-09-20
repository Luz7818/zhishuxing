"""分析报告：可 import 的报告构建函数（CLI 只是薄封装）。

每类报告返回 {"ok", "files", ...摘要}，输出默认锚定 data/outputs（不依赖 CWD）。
已修复的历史 bug：
- 奖励曲线 find_latest_npy 误选非奖励 npy → 仅匹配 *_env_*.npy 奖励格式；
- 安检排队图南 区“无引导/MADDPG”两条线数据标签互换；
- fine/plot_finetune_metrics 无 Agg 后端且调用 plt.show()（无头环境挂起）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .. import config as cfg
from ..core.animation import make_animation
from .io_utils import read_matrix_csv, read_summary_csv, write_csv_rows, write_summary_csv
from .plotting import ensure_parent, min_max_normalize, moving_average, setup_chinese_font
from .synthetic import (
    generate_congestion_matrices,
    generate_finetune_metrics,
    generate_security_queue,
    generate_transfer_distribution,
)


def _out(name: str) -> Path:
    return cfg.paths.outputs / name


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# ---------------------------------------------------------------- 奖励曲线

def run_reward_curve_report(
    data_dir: Optional[Path] = None,
    file: Optional[str] = None,
    window: int = 30,
    normalize_y: bool = True,
    output: Optional[Path] = None,
) -> Dict:
    if data_dir is not None:
        search_dirs = [Path(data_dir)]
    else:
        # 默认先找运行时输出，再回退参考样本
        search_dirs = [cfg.paths.outputs, cfg.paths.samples]
    data_dir = search_dirs[0]

    reward_file = None
    for directory in search_dirs:
        if not directory.exists():
            continue
        if file is not None:
            candidate = directory / file
        else:
            # 仅匹配训练产出的奖励格式 {algo}_env_{env}_number_{}_seed_{}.npy
            candidates = sorted(directory.glob("*_env_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
            candidate = candidates[0] if candidates else None
        if candidate is not None and candidate.exists():
            reward_file = candidate
            data_dir = directory
            break

    if reward_file is None:
        raise FileNotFoundError("未找到 *_env_*.npy 奖励文件，请先运行训练或指定 --file。")

    rewards = np.asarray(np.load(reward_file), dtype=np.float32).reshape(-1)

    y_label = "奖励值"
    normalize_info = None
    if normalize_y:
        r_min, r_max = float(rewards.min()), float(rewards.max())
        rewards = min_max_normalize(rewards)
        y_label = "归一化奖励 [0, 1]"
        normalize_info = (r_min, r_max)

    win = int(window)
    if win <= 1:
        win = max(3, min(10, len(rewards) // 20 if len(rewards) >= 20 else 3))
    rewards_smooth = moving_average(rewards, win)

    setup_chinese_font()
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="评估奖励", alpha=0.35, color="tab:blue", linewidth=1.5)
    plt.plot(rewards_smooth, label=f"平滑曲线（窗口={win}）", linewidth=2.5, color="tab:orange")
    plt.title("MADDPG奖励曲线")
    plt.xlabel("训练次数")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    save_path = ensure_parent(output or _out("reward_curve.png"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    result = {"ok": True, "files": [str(save_path)], "source": str(reward_file), "window": win}
    if normalize_info is not None:
        result["normalize"] = {"min": float(normalize_info[0]), "max": float(normalize_info[1])}
    return result


# ---------------------------------------------------------------- 拥堵热力图

def run_congestion_report(
    before_csv: Optional[Path] = None,
    after_csv: Optional[Path] = None,
    seed: int = 20260319,
    output: Optional[Path] = None,
    rank_output: Optional[Path] = None,
    peak_output: Optional[Path] = None,
    title_prefix: str = "深圳北站综合枢纽",
) -> Dict:
    if before_csv and after_csv:
        zones_b, times_b, before = read_matrix_csv(Path(before_csv))
        zones_a, times_a, after = read_matrix_csv(Path(after_csv))
        if zones_b != zones_a or times_b != times_a:
            raise ValueError("before/after CSV 的行列标签不一致，请对齐后再绘图")
        zones, time_slots = zones_b, times_b
        delta = before - after
    else:
        zones, time_slots, before, after, delta = generate_congestion_matrices(seed=seed)

    setup_chinese_font()
    output_path = ensure_parent(output or _out("shenzhen_north_congestion_heatmap.png"))
    rank_path = ensure_parent(rank_output or _out("zone_improvement_ranking.png"))
    peak_path = ensure_parent(peak_output or _out("peak_shaving_by_timeslot.png"))

    # 1) 三联热力图
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=True)
    vmin = min(float(before.min()), float(after.min()))
    vmax = max(float(before.max()), float(after.max()))

    im0 = axes[0].imshow(before, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix} 训练前拥堵热力图")
    axes[0].set_xlabel("时间")
    axes[0].set_ylabel("站点区域")
    axes[1].imshow(after, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix} 训练后拥堵热力图")
    axes[1].set_xlabel("时间")
    vmax_delta = float(np.max(np.abs(delta)))
    im2 = axes[2].imshow(delta, aspect="auto", cmap="RdYlGn", vmin=-vmax_delta, vmax=vmax_delta)
    axes[2].set_title(f"{title_prefix} 拥堵变化(前-后)")
    axes[2].set_xlabel("时间")

    for ax in axes:
        ax.set_xticks(np.arange(len(time_slots)))
        ax.set_xticklabels(time_slots, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(zones)))
        ax.set_yticklabels(zones, fontsize=9)

    cbar0 = fig.colorbar(im0, ax=axes[:2], fraction=0.025, pad=0.02)
    cbar0.set_label("拥堵强度 (人流密度/单位面积)")
    cbar1 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar1.set_label("变化量 (正值=缓解)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    # 2) 区域平均改善率排名
    improvement_rate = np.mean((before - after) / np.maximum(before, 1e-6), axis=1)
    order = np.argsort(improvement_rate)[::-1]
    zones_sorted = [zones[i] for i in order]
    rates_sorted = improvement_rate[order] * 100.0

    fig, ax = plt.subplots(figsize=(12, 7.5))
    colors = plt.cm.YlGn(np.linspace(0.45, 0.85, len(zones_sorted)))
    bars = ax.barh(zones_sorted, rates_sorted, color=colors, edgecolor="#2F4F4F", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_title(f"{title_prefix} 区域平均改善率排名")
    ax.set_xlabel("平均改善率 (%)")
    ax.set_ylabel("站点区域")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars, rates_sorted):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(rank_path, dpi=180)
    plt.close(fig)

    # 3) 分时段削峰效果
    before_total = np.sum(before, axis=0)
    after_total = np.sum(after, axis=0)
    x = np.arange(len(time_slots))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.bar(x - width / 2, before_total, width=width, label="训练前", color="#E67E22", alpha=0.88)
    ax.bar(x + width / 2, after_total, width=width, label="训练后", color="#2E86C1", alpha=0.88)
    peak_labels = ["08:00", "18:00"]
    for peak_label in peak_labels:
        if peak_label in time_slots:
            idx = time_slots.index(peak_label)
            ax.axvspan(idx - 0.5, idx + 0.5, color="#F7DC6F", alpha=0.20)
    ax.set_title(f"{title_prefix} 分时段削峰效果（Before/After）")
    ax.set_xlabel("时间段")
    ax.set_ylabel("全区域拥堵强度总量")
    ax.set_xticks(x)
    ax.set_xticklabels(time_slots, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(peak_path, dpi=180)
    plt.close(fig)

    peak_rates = {}
    for peak_label in peak_labels:
        if peak_label not in time_slots:
            continue
        idx = time_slots.index(peak_label)
        base = max(float(before_total[idx]), 1e-6)
        peak_rates[peak_label] = (float(before_total[idx]) - float(after_total[idx])) / base

    return {
        "ok": True,
        "files": [str(output_path), str(rank_path), str(peak_path)],
        "peak_shaving": peak_rates,
    }


# ---------------------------------------------------------------- 换乘时间分布

def run_transfer_time_report(
    input_csv: Optional[Path] = None,
    seed: int = 20260319,
    points: int = 1000,
    smooth_window: int = 12,
    output: Optional[Path] = None,
    save_sim_csv: Optional[Path] = None,
) -> Dict:
    if input_csv:
        iterations, p50, p90, max_values = read_summary_csv(Path(input_csv))
        csv_file = None
    else:
        iterations, p50, p90, max_values = generate_transfer_distribution(seed=seed, points=points)
        csv_file = write_summary_csv(save_sim_csv or _out("transfer_time_distribution_simulated.csv"), iterations, p50, p90, max_values)

    win = max(2, int(smooth_window))
    output_path = ensure_parent(output or _out("transfer_time_distribution.png"))

    setup_chinese_font()
    p50_s = moving_average(p50, win)
    p90_s = moving_average(p90, win)
    max_s = moving_average(max_values, win)

    plt.figure(figsize=(11, 6))
    plt.plot(iterations, p50, color="#4C78A8", alpha=0.25, linewidth=1.2)
    plt.plot(iterations, p90, color="#F58518", alpha=0.23, linewidth=1.2)
    plt.plot(iterations, max_values, color="#E45756", alpha=0.20, linewidth=1.2)
    plt.plot(iterations, p50_s, color="#4C78A8", linewidth=2.4, label=f"P50（平滑窗口={win}）")
    plt.plot(iterations, p90_s, color="#F58518", linewidth=2.4, label=f"P90（平滑窗口={win}）")
    plt.plot(iterations, max_s, color="#E45756", linewidth=2.6, label=f"最大值（平滑窗口={win}）")
    plt.title("换乘时间分布随训练迭代变化")
    plt.xlabel("训练迭代")
    plt.ylabel("换乘时间（秒）")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()

    def drop(series: np.ndarray) -> float:
        head = slice(0, max(50, len(series) // 10))
        tail = slice(max(0, len(series) - max(100, len(series) // 8)), len(series))
        return (1.0 - float(np.mean(series[tail])) / float(np.mean(series[head]))) * 100.0

    files = [str(output_path)] + ([str(csv_file)] if csv_file else [])
    return {
        "ok": True,
        "files": files,
        "points": int(len(iterations)),
        "drop_pct": {
            "p50": round(drop(p50), 2),
            "p90": round(drop(p90), 2),
            "max": round(drop(max_values), 2),
        },
    }


# ---------------------------------------------------------------- 安检排队对比

def run_security_queue_report(seed: int = 20260324, output: Optional[Path] = None, csv_output: Optional[Path] = None) -> Dict:
    time_slots, no_guide, maddpg = generate_security_queue(seed=seed)
    output_path = ensure_parent(output or _out("security_queue_comparison.png"))
    csv_path = write_security_queue_csv(csv_output or _out("security_queue_comparison.csv"), time_slots, no_guide, maddpg)

    setup_chinese_font()
    x = np.arange(len(time_slots))

    # 历史版本此处将南区两条线的数据与标签互换，已修正
    north_no = no_guide["queue_main"]
    north_maddpg = maddpg["queue_main"]
    south_no = no_guide["queue_backup"]
    south_maddpg = maddpg["queue_backup"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)

    ax1.plot(x, north_no, color="#D35400", linewidth=2.2, label="无引导")
    ax1.plot(x, north_maddpg, color="#1F77B4", linewidth=2.2, label="MADDPG动态引导")
    ax1.set_title("北区闸机群平均排队长度变化")
    ax1.set_ylabel("排队长度（人）")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left")

    ax2.plot(x, south_no, color="#D35400", linewidth=2.0, label="无引导")
    ax2.plot(x, south_maddpg, color="#2E86C1", linewidth=2.0, label="MADDPG动态引导")
    ax2.set_title("南区闸机群平均排队长度变化")
    ax2.set_ylabel("排队长度（人）")
    ax2.set_xlabel("时段")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left")

    tick_step = 12
    tick_idx = np.arange(0, len(time_slots), tick_step)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([time_slots[i] for i in tick_idx], rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    def reduction(a: float, b: float) -> float:
        return (a - b) / max(a, 1e-6) * 100.0

    summary = {
        "north_avg_reduction_pct": round(reduction(float(np.mean(north_no)), float(np.mean(north_maddpg))), 2),
        "north_peak_reduction_pct": round(reduction(float(np.max(north_no)), float(np.max(north_maddpg))), 2),
        "south_avg_reduction_pct": round(reduction(float(np.mean(south_no)), float(np.mean(south_maddpg))), 2),
        "south_peak_reduction_pct": round(reduction(float(np.max(south_no)), float(np.max(south_maddpg))), 2),
    }
    return {"ok": True, "files": [str(output_path), str(csv_path)], "summary": summary}


def write_security_queue_csv(csv_path, time_slots, no_guide: Dict, maddpg: Dict) -> Path:
    rows = []

    def append_rows(strategy_name: str, data: Dict):
        for i, t in enumerate(time_slots):
            rows.append(
                {"time": t, "node": "北区闸机群", "strategy": strategy_name,
                 "queue_length": data["queue_main"][i], "arrivals": data["arrival_main"][i], "served": data["served_main"][i]}
            )
            rows.append(
                {"time": t, "node": "南区闸机群", "strategy": strategy_name,
                 "queue_length": data["queue_backup"][i], "arrivals": data["arrival_backup"][i], "served": data["served_backup"][i]}
            )

    append_rows("无引导", no_guide)
    append_rows("MADDPG动态引导", maddpg)
    return write_csv_rows(
        csv_path,
        fieldnames=["time", "node", "strategy", "queue_length", "arrivals", "served"],
        rows=rows,
        fmt={"queue_length": "{:.4f}", "arrivals": "{:.4f}", "served": "{:.4f}"},
    )


# ---------------------------------------------------------------- 场景效率对比（原 fine/compare_transfer_efficiency_scenarios）

def run_efficiency_report(
    transfer_csv: Optional[Path] = None,
    seed: int = 20260319,
    output_csv: Optional[Path] = None,
    output_png: Optional[Path] = None,
) -> Dict:
    csv_path = Path(transfer_csv) if transfer_csv else _first_existing(
        _out("transfer_time_distribution_simulated.csv"),
        cfg.paths.samples / "transfer_time_distribution_simulated.csv",
    )
    _, p50, _, _ = read_summary_csv(csv_path, min_rows=30)
    base_before, _, global_improve_ratio = estimate_global_transfer_change(p50)
    period_adjustments = derive_period_adjustments(seed=seed)
    rows = build_scenario_rows(
        base_before_time=base_before,
        global_improve_ratio=global_improve_ratio,
        period_adjustments=period_adjustments,
    )

    out_csv = write_csv_rows(
        output_csv or _out("transfer_efficiency_scenario_comparison.csv"),
        fieldnames=[
            "period", "profile", "before_time_sec", "after_time_sec",
            "time_drop_pct", "before_eff_index", "after_eff_index", "eff_gain_pct",
        ],
        rows=rows,
        fmt={
            "before_time_sec": "{:.2f}",
            "after_time_sec": "{:.2f}",
            "time_drop_pct": "{:.2f}",
            "before_eff_index": "{:.4f}",
            "after_eff_index": "{:.4f}",
            "eff_gain_pct": "{:.2f}",
        },
    )

    setup_chinese_font()
    out_png = ensure_parent(output_png or _out("transfer_efficiency_scenario_comparison.png"))
    labels = [f"{r['period']}\n{r['profile']}" for r in rows]
    before_vals = [r["before_time_sec"] for r in rows]
    after_vals = [r["after_time_sec"] for r in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    ax.bar(x - width / 2, before_vals, width=width, label="训练前", color="#E67E22", alpha=0.9)
    ax.bar(x + width / 2, after_vals, width=width, label="训练后", color="#2E86C1", alpha=0.9)
    ax.set_title("不同场景下训练前后换乘时间对比")
    ax.set_xlabel("场景")
    ax.set_ylabel("平均换乘时间（秒，越低越好）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    for i, row in enumerate(rows):
        ax.text(x[i] + width / 2, after_vals[i] + 3, f"-{row['time_drop_pct']:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {"ok": True, "files": [str(out_csv), str(out_png)], "rows": rows}


def estimate_global_transfer_change(p50: np.ndarray, head_ratio: float = 0.1, tail_ratio: float = 0.1):
    n = len(p50)
    head_n = max(20, int(n * head_ratio))
    tail_n = max(20, int(n * tail_ratio))
    before = float(np.mean(p50[:head_n]))
    after = float(np.mean(p50[-tail_n:]))
    improve_ratio = (before - after) / max(before, 1e-6)
    return before, after, improve_ratio


def derive_period_adjustments(seed: int = 20260319) -> Dict[str, float]:
    _, time_slots, before_mat, after_mat, _ = generate_congestion_matrices(seed=seed)
    total_before = np.sum(before_mat, axis=0)
    total_after = np.sum(after_mat, axis=0)
    total_rate = (total_before - total_after) / np.maximum(total_before, 1e-6)

    peak_labels = {"08:00", "18:00"}
    peak_idx = [i for i, t in enumerate(time_slots) if t in peak_labels]
    offpeak_idx = [i for i in range(len(time_slots)) if i not in peak_idx]

    peak_improve = float(np.mean(total_rate[peak_idx]))
    offpeak_improve = float(np.mean(total_rate[offpeak_idx]))
    mean_improve = float(np.mean(total_rate))

    peak_load = float(np.mean(total_before[peak_idx]) / np.mean(total_before))
    offpeak_load = float(np.mean(total_before[offpeak_idx]) / np.mean(total_before))

    return {
        "peak_improve_mult": peak_improve / max(mean_improve, 1e-6),
        "offpeak_improve_mult": offpeak_improve / max(mean_improve, 1e-6),
        "peak_time_mult": peak_load,
        "offpeak_time_mult": offpeak_load,
    }


def build_scenario_rows(base_before_time: float, global_improve_ratio: float, period_adjustments: Dict[str, float]) -> List[Dict]:
    # 行李人群通常基础换乘时间更长，且受优化收益略小
    profile_cfg = {
        "普通旅客": {"time_mult": 1.00, "improve_mult": 1.00},
        "携带大件行李旅客": {"time_mult": 1.22, "improve_mult": 0.85},
    }
    period_cfg = {
        "高峰时段": {
            "time_mult": period_adjustments["peak_time_mult"],
            "improve_mult": period_adjustments["peak_improve_mult"],
        },
        "平峰时段": {
            "time_mult": period_adjustments["offpeak_time_mult"],
            "improve_mult": period_adjustments["offpeak_improve_mult"],
        },
    }

    rows = []
    for period_name, p_cfg in period_cfg.items():
        for profile_name, u_cfg in profile_cfg.items():
            before_time = base_before_time * p_cfg["time_mult"] * u_cfg["time_mult"]
            improve_ratio = global_improve_ratio * p_cfg["improve_mult"] * u_cfg["improve_mult"]
            improve_ratio = float(np.clip(improve_ratio, 0.03, 0.45))
            after_time = before_time * (1.0 - improve_ratio)

            before_eff = 1000.0 / max(before_time, 1e-6)
            after_eff = 1000.0 / max(after_time, 1e-6)
            eff_gain = (after_eff - before_eff) / max(before_eff, 1e-6)

            rows.append(
                {
                    "period": period_name,
                    "profile": profile_name,
                    "before_time_sec": before_time,
                    "after_time_sec": after_time,
                    "time_drop_pct": improve_ratio * 100.0,
                    "before_eff_index": before_eff,
                    "after_eff_index": after_eff,
                    "eff_gain_pct": eff_gain * 100.0,
                }
            )
    return rows


# ---------------------------------------------------------------- 微调指标与动画

def run_finetune_metrics_report(
    epochs: int = 36,
    seed: int = 42,
    output: Optional[Path] = None,
    csv_output: Optional[Path] = None,
) -> Dict:
    metrics = generate_finetune_metrics(epochs=epochs, seed=seed)
    output_path = ensure_parent(output or _out("finetune_metrics_simulated.png"))
    csv_path = write_csv_rows(
        csv_output or _out("finetune_metrics_simulated.csv"),
        fieldnames=["epoch", "loss", "bleu4", "rouge1", "rougeL"],
        rows=[
            {k: metrics[k][i] for k in ("epoch", "loss", "bleu4", "rouge1", "rougeL")}
            for i in range(len(metrics["epoch"]))
        ],
        fmt={"epoch": "{:d}", "loss": "{:.6f}", "bleu4": "{:.6f}", "rouge1": "{:.6f}", "rougeL": "{:.6f}"},
    )

    setup_chinese_font()
    x = metrics["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DeepSeek-R1-Distill-Qwen-7B 微调效果（模拟数据）", fontsize=14)

    panels = [
        ("loss", "损失值（Loss）", "损失值", None, "#e76f51"),
        ("bleu4", "BLEU-4 指标", "得分", (0, 1), "#2a9d8f"),
        ("rouge1", "ROUGE-1 指标", "得分", (0, 1), "#457b9d"),
        ("rougeL", "ROUGE-L 指标", "得分", (0, 1), "#8d99ae"),
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (key, title, ylabel, ylim, color), (row, col) in zip(panels, positions):
        axes[row, col].plot(x, metrics[key], marker="o", markersize=3.5, linewidth=2, color=color)
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel("训练轮次（Epoch）")
        axes[row, col].set_ylabel(ylabel)
        if ylim:
            axes[row, col].set_ylim(*ylim)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        "ok": True,
        "files": [str(output_path), str(csv_path)],
        "final": {k: round(float(metrics[k][-1]), 4) for k in ("loss", "bleu4", "rouge1", "rougeL")},
    }


def run_animation_report(
    frames: int = 60,
    fps: int = 20,
    n_agents: int = 44,
    seed: int = 42,
    output: Optional[Path] = None,
) -> Dict:
    output_path = make_animation(
        output_path=output or _out("transfer_env_demo.gif"),
        frames=frames,
        fps=fps,
        n_agents=n_agents,
        seed=seed,
    )
    return {"ok": True, "files": [str(output_path)], "n_agents": n_agents, "frames": max(60, frames)}
