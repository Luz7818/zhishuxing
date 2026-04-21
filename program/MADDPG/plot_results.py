import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def find_latest_npy(data_dir: Path):
    npy_files = sorted(data_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    return npy_files[0] if npy_files else None


def moving_average(values: np.ndarray, window: int):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def min_max_normalize(values: np.ndarray):
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if np.isclose(v_max, v_min):
        return np.zeros_like(values, dtype=np.float32), v_min, v_max
    normalized = (values - v_min) / (v_max - v_min)
    return normalized.astype(np.float32), v_min, v_max


def main():
    setup_chinese_font()
    parser = argparse.ArgumentParser("Plot MADDPG evaluation reward curve")
    parser.add_argument("--data_dir", type=str, default="./data_train", help="Directory containing reward .npy files")
    parser.add_argument("--file", type=str, default=None, help="Specific .npy file name (optional)")
    parser.add_argument("--window", type=int, default=30, help="Moving average window size")
    parser.add_argument("--save", type=str, default="./data_train/reward_curve.png", help="Output image path")
    parser.add_argument("--normalize_y", dest="normalize_y", action="store_true", help="Apply min-max normalization to y-axis")
    parser.add_argument("--no_normalize_y", dest="normalize_y", action="store_false", help="Use original reward scale on y-axis")
    parser.set_defaults(normalize_y=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if args.file is not None:
        reward_file = data_dir / args.file
    else:
        reward_file = find_latest_npy(data_dir)

    if reward_file is None or not reward_file.exists():
        raise FileNotFoundError("No .npy reward file found. Please run training first.")

    rewards = np.load(reward_file)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)

    y_label = "奖励值"
    normalize_info = None
    if args.normalize_y:
        rewards, r_min, r_max = min_max_normalize(rewards)
        y_label = "归一化奖励 [0, 1]"
        normalize_info = (r_min, r_max)

    window = args.window
    if window <= 1:
        window = max(3, min(10, len(rewards) // 20 if len(rewards) >= 20 else 3))
    rewards_smooth = moving_average(rewards, window)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="评估奖励", alpha=0.35, color="tab:blue", linewidth=1.5)
    plt.plot(rewards_smooth, label=f"平滑曲线（窗口={window}）", linewidth=2.5, color="tab:orange")
    plt.title("MADDPG奖励曲线")
    plt.xlabel("训练次数")
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loaded: {reward_file}")
    if normalize_info is not None:
        print(f"Y normalized with min={normalize_info[0]:.4f}, max={normalize_info[1]:.4f}")
    print(f"Saved figure: {save_path}")


if __name__ == "__main__":
    main()
