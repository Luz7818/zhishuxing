import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


class HubTransferAnimator:
    def __init__(self, width=140, height=85, n_agents=44, seed=42):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        self.frame_count = 0

        self.wall_blocks = [
            (0, 0, 140, 2),
            (0, 83, 140, 2),
            (0, 0, 2, 85),
            (138, 0, 2, 85),
            (22, 26, 10, 12),
            (22, 50, 10, 16),
            (56, 18, 8, 16),
            (56, 42, 8, 10),
            (56, 60, 8, 18),
            (100, 22, 14, 18),
            (100, 52, 14, 14),
        ]

        self.station_zones = {
            "东广场进站口": (4, 58, 16, 20),
            "西广场进站口": (4, 8, 16, 20),
            "安检闸机群": (32, 34, 16, 18),
            "安检闸机群(中部)": (72, 34, 22, 16),
            "地铁4号线": (116, 58, 18, 20),
            "地铁5号线": (116, 8, 18, 20),
        }

        self.gates = np.array([
            [21.0, 40.0], [21.0, 46.0], [33.0, 40.0], [33.0, 46.0],
            [60.0, 36.0], [60.0, 38.0],
            [60.0, 55.0], [60.0, 57.0],
            [80.0, 38.0], [80.0, 42.0], [80.0, 46.0],
            [88.0, 38.0], [88.0, 42.0], [88.0, 46.0],
            [99.0, 42.0], [99.0, 50.0], [115.0, 42.0], [115.0, 50.0],
        ], dtype=np.float32)

        self.targets = {
            "east_entry": np.array([10.0, 68.0], dtype=np.float32),
            "west_entry": np.array([10.0, 18.0], dtype=np.float32),
            "security": np.array([40.0, 44.0], dtype=np.float32),
            "security_mid": np.array([84.0, 42.0], dtype=np.float32),
            "metro4": np.array([126.0, 68.0], dtype=np.float32),
            "metro5": np.array([126.0, 18.0], dtype=np.float32),
            "exit_east": np.array([8.0, 72.0], dtype=np.float32),
            "exit_west": np.array([8.0, 12.0], dtype=np.float32),
            "left_pass_top": np.array([30.0, 56.0], dtype=np.float32),
            "left_pass_bottom": np.array([30.0, 36.0], dtype=np.float32),
            "right_pass_top": np.array([108.0, 56.0], dtype=np.float32),
            "right_pass_bottom": np.array([108.0, 44.0], dtype=np.float32),
        }

        self.agent_types = self._sample_agent_types()
        self.positions = self._sample_starts_by_type()
        self.velocities = np.zeros_like(self.positions)
        self.reached = np.zeros(self.n_agents, dtype=bool)
        self.current_stage = np.zeros(self.n_agents, dtype=np.int32)
        self.routes = self._build_routes()
        self.stage_wait = np.zeros(self.n_agents, dtype=np.int32)
        self.guidance_interventions = 0
        self.start_groups = self._build_start_groups()
        self.release_frames = self._build_release_frames()

    def _sample_agent_types(self):
        type_names = np.array(["进站换乘", "地铁出站", "站内换乘"], dtype=object)
        probs = np.array([0.42, 0.26, 0.32], dtype=np.float32)
        return self.rng.choice(type_names, size=self.n_agents, p=probs)

    def _inside_wall(self, x, y):
        for ox, oy, ow, oh in self.wall_blocks:
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True
        return False

    def _sample_point_in_zone(self, zone_name):
        zx, zy, zw, zh = self.station_zones[zone_name]
        while True:
            x = self.rng.uniform(zx + 1.0, zx + zw - 1.0)
            y = self.rng.uniform(zy + 1.0, zy + zh - 1.0)
            if not self._inside_wall(x, y):
                return np.array([x, y], dtype=np.float32)

    def _sample_starts_by_type(self):
        starts = np.zeros((self.n_agents, 2), dtype=np.float32)
        for i, t in enumerate(self.agent_types):
            if t == "进站换乘":
                starts[i] = self._sample_point_in_zone("东广场进站口") if self.rng.random() < 0.55 else self._sample_point_in_zone("西广场进站口")
            elif t == "地铁出站":
                starts[i] = self._sample_point_in_zone("地铁4号线") if self.rng.random() < 0.5 else self._sample_point_in_zone("地铁5号线")
            else:
                starts[i] = self._sample_point_in_zone("安检闸机群")
        return starts

    def _build_start_groups(self):
        groups = np.empty(self.n_agents, dtype=object)
        for i, t in enumerate(self.agent_types):
            x, y = self.positions[i]
            if t == "进站换乘":
                groups[i] = "东广场" if y >= 43 else "西广场"
            elif t == "地铁出站":
                groups[i] = "地铁4号线" if y >= 43 else "地铁5号线"
            else:
                groups[i] = "安检区内"
        return groups

    def _build_release_frames(self):
        base_release = {
            "东广场": (0, 10),
            "西广场": (8, 18),
            "地铁4号线": (16, 30),
            "地铁5号线": (24, 38),
            "安检区内": (4, 14),
        }
        release = np.zeros(self.n_agents, dtype=np.int32)
        for i in range(self.n_agents):
            lo, hi = base_release.get(self.start_groups[i], (0, 12))
            release[i] = int(self.rng.integers(lo, hi + 1))
        return release

    def _build_routes(self):
        routes = []
        for t in self.agent_types:
            if t == "进站换乘":
                if self.rng.random() < 0.5:
                    route = ["left_pass_top", "security", "security_mid", "metro4"]
                else:
                    route = ["left_pass_bottom", "security", "security_mid", "metro5"]
            elif t == "地铁出站":
                if self.rng.random() < 0.5:
                    route = ["right_pass_top", "security_mid", "security", "exit_east"]
                else:
                    route = ["right_pass_bottom", "security_mid", "security", "exit_west"]
            else:
                if self.rng.random() < 0.5:
                    route = ["security", "security_mid", "right_pass_top", "metro4"]
                else:
                    route = ["security", "security_mid", "right_pass_bottom", "metro5"]
            routes.append(route)
        return routes

    def _project_to_free_space(self, p):
        candidate = np.array(p, dtype=np.float32)
        candidate[0] = np.clip(candidate[0], 1.5, self.width - 1.5)
        candidate[1] = np.clip(candidate[1], 1.5, self.height - 1.5)
        if not self._inside_wall(candidate[0], candidate[1]):
            return candidate

        for radius in [1.5, 2.5, 3.5, 5.0, 7.0, 9.0]:
            for _ in range(16):
                ang = self.rng.uniform(0, 2 * np.pi)
                trial = candidate + np.array([np.cos(ang), np.sin(ang)], dtype=np.float32) * radius
                trial[0] = np.clip(trial[0], 1.5, self.width - 1.5)
                trial[1] = np.clip(trial[1], 1.5, self.height - 1.5)
                if not self._inside_wall(trial[0], trial[1]):
                    return trial

        # 极端情况回退到安检区中心
        return np.array([40.0, 44.0], dtype=np.float32)

    def _current_target(self, agent_idx):
        stage = self.current_stage[agent_idx]
        route = self.routes[agent_idx]
        if stage >= len(route):
            return None
        return self.targets[route[stage]]

    def _wall_repulse(self, p):
        repulse = np.zeros(2, dtype=np.float32)
        for ox, oy, ow, oh in self.wall_blocks:
            cx = np.clip(p[0], ox, ox + ow)
            cy = np.clip(p[1], oy, oy + oh)
            vec = p - np.array([cx, cy], dtype=np.float32)
            dist = np.linalg.norm(vec)
            if dist < 6.5:
                repulse += vec / (dist + 1e-4) * (6.5 - dist) * 0.26
        return repulse

    def _social_repulse(self, idx):
        p = self.positions[idx]
        diff = p - self.positions
        dist_all = np.linalg.norm(diff, axis=1)
        close_idx = np.where((dist_all > 1e-6) & (dist_all < 4.8))[0]
        if len(close_idx) == 0:
            return np.zeros(2, dtype=np.float32)
        rep = np.sum(diff[close_idx] / (dist_all[close_idx][:, None] + 1e-4), axis=0)
        return rep * 0.08

    def _local_density(self):
        density = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            d = np.linalg.norm(self.positions[i] - self.positions, axis=1)
            density[i] = np.sum((d < 5.2) & (d > 1e-6))
        return density

    def step(self, dt=1.25):
        self.frame_count += 1
        local_density = self._local_density()

        for i in range(self.n_agents):
            if self.reached[i]:
                continue

            if self.frame_count < self.release_frames[i]:
                continue

            p = self.positions[i]
            target = self._current_target(i)
            if target is None:
                self.reached[i] = True
                continue

            to_goal = target - p
            goal_dir = to_goal / (np.linalg.norm(to_goal) + 1e-6)

            repulse = self._wall_repulse(p)
            social = self._social_repulse(i)
            noise = self.rng.normal(0, 0.06, size=2)

            direction = goal_dir + repulse + social + noise
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            density_penalty = min(0.45, 0.04 * local_density[i])
            type_bias = 0.0
            if self.agent_types[i] == "地铁出站":
                type_bias = 0.12
            elif self.agent_types[i] == "站内换乘":
                type_bias = 0.06

            speed = 1.45 + type_bias + self.rng.uniform(0.15, 0.75) - density_penalty
            speed = max(0.38, speed)
            v = direction * speed

            new_p = p + v * dt
            new_p = self._project_to_free_space(new_p)

            if self._inside_wall(new_p[0], new_p[1]):
                jitter = self.rng.normal(0, 1.0, size=2)
                new_p = self._project_to_free_space(p - 0.22 * v + 0.55 * jitter)

            self.positions[i] = new_p
            self.velocities[i] = v
            self.stage_wait[i] += 1

            if np.linalg.norm(self.positions[i] - target) < 4.5:
                self.current_stage[i] += 1
                self.stage_wait[i] = 0
                if self.current_stage[i] >= len(self.routes[i]):
                    self.reached[i] = True

            # 长时间卡滞时，触发引导系统介入，直接引导到当前阶段目标附近
            if (not self.reached[i]) and self.stage_wait[i] > 18:
                guided_pos = self._project_to_free_space(target + self.rng.normal(0, 0.8, size=2))
                self.positions[i] = guided_pos
                self.current_stage[i] += 1
                self.stage_wait[i] = 0
                self.guidance_interventions += 1
                if self.current_stage[i] >= len(self.routes[i]):
                    self.reached[i] = True

    def congestion_index(self):
        dense_count = 0
        for i in range(self.n_agents):
            if self.frame_count < self.release_frames[i]:
                continue
            d = np.linalg.norm(self.positions[i] - self.positions, axis=1)
            nearby = np.sum((d < 4.5) & (d > 1e-6))
            if nearby >= 4:
                dense_count += 1
        active_count = max(1, int(np.sum(self.release_frames <= self.frame_count)))
        return dense_count / active_count


def _draw_station_layout(ax, env: HubTransferAnimator):
    zone_colors = {
        "东广场进站口": "#E8F1FF",
        "西广场进站口": "#E8F1FF",
        "安检闸机群": "#FFF2CC",
        "安检闸机群(中部)": "#FFE7B3",
        "换乘主通道": "#F2F2F2",
        "地铁4号线": "#EAF8EA",
        "地铁5号线": "#EAF8EA",
        "高铁候车区": "#F5ECFF",
    }

    for name, (zx, zy, zw, zh) in env.station_zones.items():
        ax.add_patch(
            patches.Rectangle(
                (zx, zy), zw, zh,
                facecolor=zone_colors.get(name, "#F7F7F7"),
                edgecolor="#9A9A9A",
                linewidth=1.2,
                alpha=0.82,
            )
        )
        ax.text(zx + zw / 2, zy + zh / 2, name, ha="center", va="center", fontsize=8.7, color="#333333")

    for ox, oy, ow, oh in env.wall_blocks:
        ax.add_patch(
            patches.Rectangle((ox, oy), ow, oh, facecolor="#2D6CDF", edgecolor="#1E4EA8", alpha=0.9)
        )

    ax.scatter(env.gates[:, 0], env.gates[:, 1], s=36, c="#6B7280", marker="s", alpha=0.9, label="闸机/通道节点")



def make_animation(output_path: Path, frames=100, fps=20, n_agents=26, seed=42):
    setup_chinese_font()
    env = HubTransferAnimator(n_agents=n_agents, seed=seed)

    fig, ax = plt.subplots(figsize=(12.6, 7.5))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("综合交通枢纽强化学习环境（真实布局示意）", fontsize=14)
    ax.set_xlabel("空间X")
    ax.set_ylabel("空间Y")

    _draw_station_layout(ax, env)

    ax.scatter(
        [env.targets["metro4"][0], env.targets["metro5"][0], env.targets["exit_east"][0], env.targets["exit_west"][0]],
        [env.targets["metro4"][1], env.targets["metro5"][1], env.targets["exit_east"][1], env.targets["exit_west"][1]],
        s=115,
        c="#2FA84F",
        marker="o",
        label="目标点",
    )

    scat = ax.scatter(env.positions[:, 0], env.positions[:, 1], s=40, c="#D62828", edgecolors="white", linewidths=0.25, label="行人")

    text_box_style = dict(facecolor="white", edgecolor="#888888", alpha=0.9, boxstyle="round,pad=0.28")
    info_text = ax.text(2, env.height - 2, "", fontsize=10, va="top", bbox=text_box_style)

    obstacle_proxy = patches.Patch(facecolor="#2D6CDF", edgecolor="#1E4EA8", label="阻挡区域")
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles=handles + [obstacle_proxy],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=4,
        frameon=True,
    )
    legend.get_frame().set_alpha(0.9)

    def update(frame_idx):
        env.step()
        scat.set_offsets(env.positions)
        reached_count = int(np.sum(env.reached))
        released_count = int(np.sum(env.release_frames <= env.frame_count))
        reach_rate = reached_count / env.n_agents
        congestion = env.congestion_index()
        info_text.set_text(
            f"帧: {frame_idx:03d} | 已放行: {released_count}/{env.n_agents} | 已完成换乘: {reached_count}/{env.n_agents}\n"
            f"到达率: {reach_rate * 100:.1f}% | 拥堵指数: {congestion:.2f} | 引导介入: {env.guidance_interventions}"
        )
        return scat, info_text

    anim = FuncAnimation(fig, update, frames=frames, interval=int(1000 / fps), blit=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(bottom=0.16)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser("生成综合交通枢纽换乘环境动图")
    parser.add_argument("--output", type=str, default="./data_train/transfer_env_demo.gif", help="输出GIF路径")
    parser.add_argument("--frames", type=int, default=100, help="动图帧数")
    parser.add_argument("--fps", type=int, default=20, help="帧率")
    parser.add_argument("--n_agents", type=int, default=44, help="行人数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    output_path = Path(args.output)
    make_animation(
        output_path=output_path,
        frames=max(60, args.frames),
        fps=max(5, args.fps),
        n_agents=max(8, args.n_agents),
        seed=args.seed,
    )

    print(f"Saved GIF: {output_path}")


if __name__ == "__main__":
    main()
