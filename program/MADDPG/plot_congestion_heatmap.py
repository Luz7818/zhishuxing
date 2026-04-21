import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def default_zones():
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


def default_time_slots():
    return [
        "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00",
        "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00",
    ]


def read_matrix_csv(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"CSV 格式错误: {csv_path}")

    header = rows[0]
    time_slots = header[1:]
    zones = []
    values = []

    for row in rows[1:]:
        if not row:
            continue
        zones.append(row[0])
        values.append([float(x) for x in row[1:]])

    matrix = np.asarray(values, dtype=np.float32)
    return zones, time_slots, matrix


def generate_synthetic_data(seed=20260319):
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


def plot_heatmaps(zones, time_slots, before, after, delta, output_path: Path, title_prefix="深圳北站综合枢纽"):
    setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=True)

    vmin = min(float(before.min()), float(after.min()))
    vmax = max(float(before.max()), float(after.max()))

    im0 = axes[0].imshow(before, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix} 训练前拥堵热力图")
    axes[0].set_xlabel("时间")
    axes[0].set_ylabel("站点区域")

    im1 = axes[1].imshow(after, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_avg_improvement_rate(before: np.ndarray, after: np.ndarray):
    # 按区域计算平均改善率：mean((before-after)/before)
    eps = 1e-6
    ratio = (before - after) / np.maximum(before, eps)
    return np.mean(ratio, axis=1)


def plot_improvement_ranking(zones, improvement_rate, output_path: Path, title_prefix="深圳北站综合枢纽"):
    setup_chinese_font()

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_peak_shaving_by_time(time_slots, before, after, output_path: Path, title_prefix="深圳北站综合枢纽"):
    setup_chinese_font()

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    peak_rates = {}
    for peak_label in peak_labels:
        if peak_label not in time_slots:
            continue
        idx = time_slots.index(peak_label)
        base = max(float(before_total[idx]), 1e-6)
        rate = (float(before_total[idx]) - float(after_total[idx])) / base
        peak_rates[peak_label] = rate

    return peak_rates


def main():
    parser = argparse.ArgumentParser("Plot spatiotemporal congestion heatmaps (before vs after training)")
    parser.add_argument("--before_csv", type=str, default=None, help="训练前拥堵矩阵CSV路径，可选")
    parser.add_argument("--after_csv", type=str, default=None, help="训练后拥堵矩阵CSV路径，可选")
    parser.add_argument("--output", type=str, default="./data_train/shenzhen_north_congestion_heatmap.png", help="输出图片路径")
    parser.add_argument("--rank_output", type=str, default="./data_train/zone_improvement_ranking.png", help="区域平均改善率排名图输出路径")
    parser.add_argument("--peak_output", type=str, default="./data_train/peak_shaving_by_timeslot.png", help="分时段削峰效果图输出路径")
    parser.add_argument("--seed", type=int, default=20260319, help="模拟数据随机种子")
    args = parser.parse_args()

    output_path = Path(args.output)
    rank_output_path = Path(args.rank_output)
    peak_output_path = Path(args.peak_output)

    if args.before_csv and args.after_csv:
        before_csv = Path(args.before_csv)
        after_csv = Path(args.after_csv)
        if not before_csv.exists() or not after_csv.exists():
            raise FileNotFoundError("before_csv 或 after_csv 不存在")

        zones_b, times_b, before = read_matrix_csv(before_csv)
        zones_a, times_a, after = read_matrix_csv(after_csv)

        if zones_b != zones_a or times_b != times_a:
            raise ValueError("before/after CSV 的行列标签不一致，请对齐后再绘图")

        zones, time_slots = zones_b, times_b
        delta = before - after
    else:
        zones, time_slots, before, after, delta = generate_synthetic_data(seed=args.seed)

    plot_heatmaps(
        zones=zones,
        time_slots=time_slots,
        before=before,
        after=after,
        delta=delta,
        output_path=output_path,
    )

    improvement_rate = compute_avg_improvement_rate(before=before, after=after)
    plot_improvement_ranking(
        zones=zones,
        improvement_rate=improvement_rate,
        output_path=rank_output_path,
    )

    peak_rates = plot_peak_shaving_by_time(
        time_slots=time_slots,
        before=before,
        after=after,
        output_path=peak_output_path,
    )

    print(f"Saved heatmap: {output_path}")
    print(f"Saved ranking: {rank_output_path}")
    print(f"Saved peak chart: {peak_output_path}")
    if "08:00" in peak_rates:
        print(f"Morning peak shaving rate (08:00): {peak_rates['08:00'] * 100:.2f}%")
    if "18:00" in peak_rates:
        print(f"Evening peak shaving rate (18:00): {peak_rates['18:00'] * 100:.2f}%")


if __name__ == "__main__":
    main()
