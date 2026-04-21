import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from MADDPG.plot_congestion_heatmap import generate_synthetic_data


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_transfer_distribution(csv_path: Path):
    iterations = []
    p50 = []
    p90 = []
    max_values = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"iteration", "p50", "p90", "max"}
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError("输入CSV必须包含列: iteration,p50,p90,max")

        for row in reader:
            iterations.append(int(row["iteration"]))
            p50.append(float(row["p50"]))
            p90.append(float(row["p90"]))
            max_values.append(float(row["max"]))

    if len(iterations) < 30:
        raise ValueError("样本点过少，无法稳定估计训练前后区间")

    return (
        np.asarray(iterations, dtype=np.int32),
        np.asarray(p50, dtype=np.float32),
        np.asarray(p90, dtype=np.float32),
        np.asarray(max_values, dtype=np.float32),
    )


def estimate_global_transfer_change(p50: np.ndarray, head_ratio: float = 0.1, tail_ratio: float = 0.1):
    n = len(p50)
    head_n = max(20, int(n * head_ratio))
    tail_n = max(20, int(n * tail_ratio))

    before = float(np.mean(p50[:head_n]))
    after = float(np.mean(p50[-tail_n:]))
    improve_ratio = (before - after) / max(before, 1e-6)
    return before, after, improve_ratio


def derive_period_adjustments(seed: int = 20260319):
    _, time_slots, before_mat, after_mat, _ = generate_synthetic_data(seed=seed)

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


def build_scenario_rows(base_before_time: float, global_improve_ratio: float, period_adjustments: dict):
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


def save_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "period",
        "profile",
        "before_time_sec",
        "after_time_sec",
        "time_drop_pct",
        "before_eff_index",
        "after_eff_index",
        "eff_gain_pct",
    ]

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "period": row["period"],
                    "profile": row["profile"],
                    "before_time_sec": f"{row['before_time_sec']:.2f}",
                    "after_time_sec": f"{row['after_time_sec']:.2f}",
                    "time_drop_pct": f"{row['time_drop_pct']:.2f}",
                    "before_eff_index": f"{row['before_eff_index']:.4f}",
                    "after_eff_index": f"{row['after_eff_index']:.4f}",
                    "eff_gain_pct": f"{row['eff_gain_pct']:.2f}",
                }
            )


def save_plot(rows, output_png: Path):
    setup_chinese_font()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"{r['period']}\n{r['profile']}" for r in rows]
    before = [r["before_time_sec"] for r in rows]
    after = [r["after_time_sec"] for r in rows]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    ax.bar(x - width / 2, before, width=width, label="训练前", color="#E67E22", alpha=0.9)
    ax.bar(x + width / 2, after, width=width, label="训练后", color="#2E86C1", alpha=0.9)

    ax.set_title("不同场景下训练前后换乘时间对比")
    ax.set_xlabel("场景")
    ax.set_ylabel("平均换乘时间（秒，越低越好）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    for i, row in enumerate(rows):
        ax.text(x[i] + width / 2, after[i] + 3, f"-{row['time_drop_pct']:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close(fig)


def print_summary(rows):
    print("=== 场景对比（训练前 -> 训练后）===")
    for row in rows:
        print(
            f"{row['period']} | {row['profile']}: "
            f"{row['before_time_sec']:.2f}s -> {row['after_time_sec']:.2f}s, "
            f"时间下降 {row['time_drop_pct']:.2f}%, 效率提升 {row['eff_gain_pct']:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser("Compare transfer efficiency before/after training across scenarios")
    parser.add_argument(
        "--transfer_csv",
        type=str,
        default="./data_train/transfer_time_distribution_simulated.csv",
        help="换乘时间分布CSV（iteration,p50,p90,max）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./data_train/transfer_efficiency_scenario_comparison.csv",
        help="对比结果CSV输出路径",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="./data_train/transfer_efficiency_scenario_comparison.png",
        help="对比图输出路径",
    )
    parser.add_argument("--seed", type=int, default=20260319, help="拥堵时段模拟种子")
    args = parser.parse_args()

    _, p50, _, _ = read_transfer_distribution(Path(args.transfer_csv))
    base_before, _, global_improve_ratio = estimate_global_transfer_change(p50)
    period_adjustments = derive_period_adjustments(seed=args.seed)
    rows = build_scenario_rows(
        base_before_time=base_before,
        global_improve_ratio=global_improve_ratio,
        period_adjustments=period_adjustments,
    )

    save_csv(rows, Path(args.output_csv))
    save_plot(rows, Path(args.output_png))
    print_summary(rows)
    print(f"已保存CSV: {args.output_csv}")
    print(f"已保存图片: {args.output_png}")


if __name__ == "__main__":
    main()