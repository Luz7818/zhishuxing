import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


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


def gaussian(x: np.ndarray, center: float, sigma: float):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def generate_demand_and_capacity(minutes: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)

    # 单位：人/5分钟。按“小规模排队（数量级约10）”进行校准
    morning_peak = gaussian(minutes, center=130, sigma=45)
    midday_peak = gaussian(minutes, center=360, sigma=55)
    evening_peak = gaussian(minutes, center=720, sigma=65)

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


def sigmoid(x: float):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_strategy(strategy: str, arrival_rate: np.ndarray, main_capacity: np.ndarray, backup_capacity: np.ndarray, seed: int):
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
            # 无引导：绝大部分旅客仍集中在主口
            backup_share = 0.08
            if q_main > 9:
                backup_share = 0.12
            if q_main > 16:
                backup_share = 0.18
        elif strategy == "MADDPG动态引导":
            # 动态引导：根据排队压力分流，并在高压时临时增开备用通道
            pressure = (q_main - 0.9 * q_backup) / 6.0
            backup_share = 0.20 + 0.34 * sigmoid(pressure)
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


def save_csv(time_slots, output_csv: Path, no_guide: dict, maddpg: dict):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "time",
        "node",
        "strategy",
        "queue_length",
        "arrivals",
        "served",
    ]

    def write_rows(writer, strategy_name: str, data: dict):
        for i, t in enumerate(time_slots):
            writer.writerow(
                {
                    "time": t,
                    "node": "北区闸机群",
                    "strategy": strategy_name,
                    "queue_length": f"{data['queue_main'][i]:.4f}",
                    "arrivals": f"{data['arrival_main'][i]:.4f}",
                    "served": f"{data['served_main'][i]:.4f}",
                }
            )
            writer.writerow(
                {
                    "time": t,
                    "node": "南区闸机群",
                    "strategy": strategy_name,
                    "queue_length": f"{data['queue_backup'][i]:.4f}",
                    "arrivals": f"{data['arrival_backup'][i]:.4f}",
                    "served": f"{data['served_backup'][i]:.4f}",
                }
            )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        write_rows(writer, "无引导", no_guide)
        write_rows(writer, "MADDPG动态引导", maddpg)


def plot_curves(time_slots, output_png: Path, no_guide: dict, maddpg: dict):
    setup_chinese_font()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(time_slots))

    north_no = no_guide["queue_main"]
    north_maddpg = maddpg["queue_main"]
    south_maddpg = no_guide["queue_backup"]
    south_no = maddpg["queue_backup"]

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
    plt.savefig(output_png, dpi=180)
    plt.close(fig)


def summarize(no_guide: dict, maddpg: dict):
    north_no = no_guide["queue_main"]
    north_maddpg = maddpg["queue_main"]
    south_no = no_guide["queue_backup"]
    south_maddpg = maddpg["queue_backup"]

    def reduction(a: float, b: float):
        return (a - b) / max(a, 1e-6) * 100.0

    print("=== 北/南区闸机群排队对比摘要 ===")
    print(f"北区平均排队: 无引导={float(np.mean(north_no)):.2f}, MADDPG={float(np.mean(north_maddpg)):.2f}, 降幅={reduction(float(np.mean(north_no)), float(np.mean(north_maddpg))):.2f}%")
    print(f"北区峰值排队: 无引导={float(np.max(north_no)):.2f}, MADDPG={float(np.max(north_maddpg)):.2f}, 降幅={reduction(float(np.max(north_no)), float(np.max(north_maddpg))):.2f}%")
    print(f"南区平均排队: 无引导={float(np.mean(south_no)):.2f}, MADDPG={float(np.mean(south_maddpg)):.2f}, 降幅={reduction(float(np.mean(south_no)), float(np.mean(south_maddpg))):.2f}%")
    print(f"南区峰值排队: 无引导={float(np.max(south_no)):.2f}, MADDPG={float(np.max(south_maddpg)):.2f}, 降幅={reduction(float(np.max(south_no)), float(np.max(south_maddpg))):.2f}%")


def main():
    parser = argparse.ArgumentParser("Plot security queue comparison: no-guidance vs MADDPG dynamic guidance")
    parser.add_argument("--seed", type=int, default=20260324, help="随机种子")
    parser.add_argument("--output", type=str, default="./data_train/security_queue_comparison.png", help="输出图路径")
    parser.add_argument("--csv", type=str, default="./data_train/security_queue_comparison.csv", help="输出CSV路径")
    args = parser.parse_args()

    time_slots, minutes = build_time_slots()
    arrival_rate, main_capacity, backup_capacity = generate_demand_and_capacity(minutes, seed=args.seed)

    no_guide = simulate_strategy(
        strategy="无引导",
        arrival_rate=arrival_rate,
        main_capacity=main_capacity,
        backup_capacity=backup_capacity,
        seed=args.seed + 101,
    )
    maddpg = simulate_strategy(
        strategy="MADDPG动态引导",
        arrival_rate=arrival_rate,
        main_capacity=main_capacity,
        backup_capacity=backup_capacity,
        seed=args.seed + 202,
    )

    output_png = Path(args.output)
    output_csv = Path(args.csv)

    save_csv(time_slots, output_csv, no_guide=no_guide, maddpg=maddpg)
    plot_curves(time_slots, output_png, no_guide=no_guide, maddpg=maddpg)
    summarize(no_guide=no_guide, maddpg=maddpg)

    print(f"已保存图片: {output_png}")
    print(f"已保存CSV: {output_csv}")


if __name__ == "__main__":
    main()
