import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def set_chinese_font():
    """Configure Chinese-capable fonts with fallbacks for Windows/Matplotlib."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Arial Unicode MS",
    ]

    installed = {f.name for f in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _correlated_noise(rng: np.random.Generator, size: int, scale: float = 1.0):
    """Generate smooth, correlated jitter to mimic realistic training fluctuation."""
    white = rng.normal(0, 1, size=size)
    smooth = np.convolve(white, np.array([0.2, 0.6, 0.2]), mode="same")
    return smooth * scale


def generate_simulated_metrics(epochs: int = 36, seed: int = 42):
    """Generate realistic simulated fine-tuning metrics for visualization."""
    rng = np.random.default_rng(seed)
    x = np.arange(1, epochs + 1)
    p = (x - 1) / (epochs - 1)

    # Loss: rapid early drop, slower late convergence, with minor bumps.
    loss_base = 1.7 * np.exp(-4.3 * p) + 0.23
    loss_wave = 0.035 * np.sin(6.2 * np.pi * p) * np.exp(-1.2 * p)
    lr_bump_1 = 0.06 * np.exp(-((p - 0.28) ** 2) / 0.002)
    lr_bump_2 = 0.035 * np.exp(-((p - 0.62) ** 2) / 0.003)
    loss_noise = _correlated_noise(rng, epochs, scale=0.018) + rng.normal(0, 0.008, size=epochs)
    loss = np.clip(loss_base + loss_wave + lr_bump_1 + lr_bump_2 + loss_noise, 0.16, None)

    # BLEU-4: rises quickly then saturates, with slight temporary regressions.
    bleu_trend = 0.09 + 0.35 * (1 - np.exp(-3.8 * p))
    bleu_wave = 0.012 * np.sin(5.1 * np.pi * p) * np.exp(-0.7 * p)
    bleu_dip = -0.018 * np.exp(-((p - 0.42) ** 2) / 0.0025)
    bleu_noise = _correlated_noise(rng, epochs, scale=0.006) + rng.normal(0, 0.004, size=epochs)
    bleu4 = np.clip(bleu_trend + bleu_wave + bleu_dip + bleu_noise, 0, 1)

    # ROUGE-1: stronger keyword coverage than BLEU, steadily improving.
    rouge1_trend = 0.28 + 0.46 * (1 - np.exp(-4.0 * p))
    rouge1_wave = 0.01 * np.sin(4.5 * np.pi * p) * np.exp(-0.8 * p)
    rouge1_noise = _correlated_noise(rng, epochs, scale=0.007) + rng.normal(0, 0.004, size=epochs)
    rouge1 = np.clip(rouge1_trend + rouge1_wave + rouge1_noise, 0, 1)

    # ROUGE-L: coherence score grows with similar pattern and lower absolute value.
    rouge_l_gap = 0.055 + 0.01 * (1 - np.exp(-2.3 * p))
    rouge_l_noise = _correlated_noise(rng, epochs, scale=0.006) + rng.normal(0, 0.0035, size=epochs)
    rouge_l = np.clip(rouge1 - rouge_l_gap + rouge_l_noise, 0, 1)

    return x, loss, bleu4, rouge1, rouge_l


def plot_metrics(save_path: str = "finetune_metrics_simulated.png"):
    x, loss, bleu4, rouge1, rouge_l = generate_simulated_metrics()

    # Fall back gracefully for environments with older matplotlib style sets.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    set_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DeepSeek-R1-Distill-Qwen-7B 微调效果（模拟数据）", fontsize=14)

    axes[0, 0].plot(x, loss, marker="o", markersize=3.5, linewidth=2, color="#e76f51")
    axes[0, 0].set_title("损失值（Loss）")
    axes[0, 0].set_xlabel("训练轮次（Epoch）")
    axes[0, 0].set_ylabel("损失值")

    axes[0, 1].plot(x, bleu4, marker="o", markersize=3.5, linewidth=2, color="#2a9d8f")
    axes[0, 1].set_title("BLEU-4 指标")
    axes[0, 1].set_xlabel("训练轮次（Epoch）")
    axes[0, 1].set_ylabel("得分")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(x, rouge1, marker="o", markersize=3.5, linewidth=2, color="#457b9d")
    axes[1, 0].set_title("ROUGE-1 指标")
    axes[1, 0].set_xlabel("训练轮次（Epoch）")
    axes[1, 0].set_ylabel("得分")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(x, rouge_l, marker="o", markersize=3.5, linewidth=2, color="#8d99ae")
    axes[1, 1].set_title("ROUGE-L 指标")
    axes[1, 1].set_xlabel("训练轮次（Epoch）")
    axes[1, 1].set_ylabel("得分")
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_metrics()
