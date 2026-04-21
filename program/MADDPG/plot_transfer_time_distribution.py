import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def moving_average(values: np.ndarray, window: int):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0], dtype=np.float32)
    return np.concatenate([pad, smoothed]).astype(np.float32)


def read_summary_csv(csv_path: Path):
    iterations = []
    p50 = []
    p90 = []
    max_values = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"iteration", "p50", "p90", "max"}
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV 需要包含列: iteration,p50,p90,max")

        for row in reader:
            iterations.append(int(row["iteration"]))
            p50.append(float(row["p50"]))
            p90.append(float(row["p90"]))
            max_values.append(float(row["max"]))

    if len(iterations) == 0:
        raise ValueError("CSV 为空，无法绘图")

    return (
        np.asarray(iterations, dtype=np.int32),
        np.asarray(p50, dtype=np.float32),
        np.asarray(p90, dtype=np.float32),
        np.asarray(max_values, dtype=np.float32),
    )


def generate_synthetic_distribution(seed=20260319, points=1000):
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


def plot_curves(iterations, p50, p90, max_values, output_path: Path, smooth_window: int):
    setup_chinese_font()

    p50_s = moving_average(p50, smooth_window)
    p90_s = moving_average(p90, smooth_window)
    max_s = moving_average(max_values, smooth_window)

    plt.figure(figsize=(11, 6))

    plt.plot(iterations, p50, color="#4C78A8", alpha=0.25, linewidth=1.2)
    plt.plot(iterations, p90, color="#F58518", alpha=0.23, linewidth=1.2)
    plt.plot(iterations, max_values, color="#E45756", alpha=0.20, linewidth=1.2)

    plt.plot(iterations, p50_s, color="#4C78A8", linewidth=2.4, label=f"P50（平滑窗口={smooth_window}）")
    plt.plot(iterations, p90_s, color="#F58518", linewidth=2.4, label=f"P90（平滑窗口={smooth_window}）")
    plt.plot(iterations, max_s, color="#E45756", linewidth=2.6, label=f"最大值（平滑窗口={smooth_window}）")

    plt.title("换乘时间分布随训练迭代变化")
    plt.xlabel("训练迭代")
    plt.ylabel("换乘时间（秒）")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_summary_csv(csv_path: Path, iterations, p50, p90, max_values):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "p50", "p90", "max"])
        for i, a, b, c in zip(iterations, p50, p90, max_values):
            writer.writerow([int(i), float(a), float(b), float(c)])


def main():
    parser = argparse.ArgumentParser("Plot P50/P90/Max transfer-time distribution over training iterations")
    parser.add_argument("--input_csv", type=str, default=None, help="输入CSV（列: iteration,p50,p90,max）")
    parser.add_argument("--output", type=str, default="./data_train/transfer_time_distribution.png", help="输出图片路径")
    parser.add_argument("--smooth_window", type=int, default=12, help="平滑窗口")
    parser.add_argument("--seed", type=int, default=20260319, help="模拟数据随机种子")
    parser.add_argument("--points", type=int, default=1000, help="模拟数据点数")
    parser.add_argument("--save_sim_csv", type=str, default="./data_train/transfer_time_distribution_simulated.csv", help="模拟数据CSV保存路径")
    args = parser.parse_args()

    if args.input_csv:
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"输入CSV不存在: {csv_path}")
        iterations, p50, p90, max_values = read_summary_csv(csv_path)
    else:
        iterations, p50, p90, max_values = generate_synthetic_distribution(seed=args.seed, points=args.points)
        save_summary_csv(Path(args.save_sim_csv), iterations, p50, p90, max_values)

    window = max(2, int(args.smooth_window))
    output_path = Path(args.output)
    plot_curves(iterations, p50, p90, max_values, output_path=output_path, smooth_window=window)

    print(f"Saved figure: {output_path}")
    print(f"Points: {len(iterations)}")
    print(f"P50 range: {float(np.min(p50)):.2f} ~ {float(np.max(p50)):.2f}")
    print(f"P90 range: {float(np.min(p90)):.2f} ~ {float(np.max(p90)):.2f}")
    print(f"Max range: {float(np.min(max_values)):.2f} ~ {float(np.max(max_values)):.2f}")

    head_slice = slice(0, max(50, len(iterations) // 10))
    tail_slice = slice(max(0, len(iterations) - max(100, len(iterations) // 8)), len(iterations))
    p50_drop = (1.0 - float(np.mean(p50[tail_slice])) / float(np.mean(p50[head_slice]))) * 100.0
    p90_drop = (1.0 - float(np.mean(p90[tail_slice])) / float(np.mean(p90[head_slice]))) * 100.0
    max_drop = (1.0 - float(np.mean(max_values[tail_slice])) / float(np.mean(max_values[head_slice]))) * 100.0
    print(f"Drop ratio -> P50: {p50_drop:.2f}%, P90: {p90_drop:.2f}%, Max: {max_drop:.2f}%")


if __name__ == "__main__":
    main()
