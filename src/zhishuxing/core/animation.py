"""综合交通枢纽微观行人仿真与换乘环境动图生成。

逻辑自历史 animate_transfer_env.py 平移：布局、三类行人路线、放行窗口、
社会力模型与“卡滞 >18 帧触发引导介入”机制全部保持。

性能：step() 已向量化（成对距离矩阵 + 墙体斥力广播 + 阶段目标查表 +
活跃集压缩），44人×1000帧约 17x 加速、500人×300帧约 8x 加速；
社会力改为帧首快照（群体仿真标准做法），统计行为与原实现等价，
基准数据见 code_optimization/report.md。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.plotting import ensure_parent, setup_chinese_font


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

        # ---- 向量化辅助表 ----
        self._max_stage = max(len(r) for r in self.routes)
        stage_targets = np.zeros((self.n_agents, self._max_stage, 2), dtype=np.float32)
        for i, route in enumerate(self.routes):
            for s, key in enumerate(route):
                stage_targets[i, s] = self.targets[key]
        self._stage_targets = stage_targets

        bias = np.zeros(self.n_agents, dtype=np.float32)
        bias[self.agent_types == "地铁出站"] = 0.12
        bias[self.agent_types == "站内换乘"] = 0.06
        self._type_bias = bias

        walls = np.asarray(self.wall_blocks, dtype=np.float32)
        self._wall_xy = walls[:, :2]
        self._wall_wh = walls[:, 2:]

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

    # ---------------------------------------------------------------- 向量化力场

    def _pairwise(self):
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff, optimize=True))
        return diff, dist

    def _local_density(self):
        _, dist = self._pairwise()
        return np.sum((dist < 5.2) & (dist > 1e-6), axis=1).astype(np.float32)

    def _social_forces(self):
        diff, dist = self._pairwise()
        mask = (dist > 1e-6) & (dist < 4.8)
        inv = np.where(mask, 1.0 / (dist + 1e-4), 0.0).astype(np.float32)
        rep = np.einsum("ijk,ij->ik", diff, inv, optimize=True)
        return (rep * 0.08).astype(np.float32)

    def _wall_repulse_all(self, pts):
        """pts: (m,2) → (m,2) 墙体斥力（对 11 块墙体广播）。"""
        wx, wy = self._wall_xy[:, 0], self._wall_xy[:, 1]
        ww, wh = self._wall_wh[:, 0], self._wall_wh[:, 1]
        cx = np.clip(pts[:, None, 0], wx[None, :], (wx + ww)[None, :])
        cy = np.clip(pts[:, None, 1], wy[None, :], (wy + wh)[None, :])
        vec_x = pts[:, None, 0] - cx
        vec_y = pts[:, None, 1] - cy
        dist = np.sqrt(vec_x ** 2 + vec_y ** 2)
        strength = np.where(dist < 6.5, (6.5 - dist) / (dist + 1e-4) * 0.26, 0.0).astype(np.float32)
        return np.stack([np.sum(vec_x * strength, axis=1), np.sum(vec_y * strength, axis=1)], axis=1)

    def _inside_wall_any(self, pts):
        wx, wy = self._wall_xy[:, 0], self._wall_xy[:, 1]
        ww, wh = self._wall_wh[:, 0], self._wall_wh[:, 1]
        inside = (
            (pts[:, None, 0] >= wx[None, :]) & (pts[:, None, 0] <= (wx + ww)[None, :])
            & (pts[:, None, 1] >= wy[None, :]) & (pts[:, None, 1] <= (wy + wh)[None, :])
        )
        return np.any(inside, axis=1)

    # ---------------------------------------------------------------- 主循环

    def step(self, dt=1.25):
        self.frame_count += 1
        active_idx = np.where((~self.reached) & (self.release_frames <= self.frame_count))[0]
        if active_idx.size == 0:
            return

        # 无剩余阶段的 agent 直接标记完成（历史语义）
        route_lengths = np.array([len(r) for r in self.routes])
        done_mask = self.current_stage[active_idx] >= route_lengths[active_idx]
        if done_mask.any():
            self.reached[active_idx[done_mask]] = True
            active_idx = active_idx[~done_mask]
            if active_idx.size == 0:
                return

        diff, dist = self._pairwise()
        density = np.sum((dist < 5.2) & (dist > 1e-6), axis=1).astype(np.float32)

        mask = (dist > 1e-6) & (dist < 4.8)
        inv = np.where(mask, 1.0 / (dist + 1e-4), 0.0).astype(np.float32)
        social = (np.einsum("ijk,ij->ik", diff, inv, optimize=True) * 0.08).astype(np.float32)

        idx = active_idx
        m = idx.size
        p = self.positions[idx]
        stage = np.clip(self.current_stage[idx], 0, self._max_stage - 1)
        target = self._stage_targets[idx, stage]

        to_goal = target - p
        goal_dir = to_goal / (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-6)

        repulse = self._wall_repulse_all(p)
        noise = self.rng.normal(0, 0.06, size=(m, 2))

        direction = goal_dir + repulse + social[idx] + noise
        direction = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6)

        density_penalty = np.minimum(0.45, 0.04 * density[idx])
        speed = np.maximum(0.38, 1.45 + self._type_bias[idx] + self.rng.uniform(0.15, 0.75, size=m) - density_penalty)
        v = direction * speed[:, None]

        new_p = p + v * dt
        new_p[:, 0] = np.clip(new_p[:, 0], 1.5, self.width - 1.5)
        new_p[:, 1] = np.clip(new_p[:, 1], 1.5, self.height - 1.5)

        inside = self._inside_wall_any(new_p)
        if inside.any():  # 罕见：逐个回退到原始投影逻辑
            for k in np.where(inside)[0]:
                jitter = self.rng.normal(0, 1.0, size=2)
                new_p[k] = self._project_to_free_space(p[k] - 0.22 * v[k] + 0.55 * jitter)

        self.positions[idx] = new_p
        self.velocities[idx] = v
        self.stage_wait[idx] += 1

        arrived = np.linalg.norm(new_p - target, axis=1) < 4.5
        if arrived.any():
            agents = idx[arrived]
            self.current_stage[agents] += 1
            self.stage_wait[agents] = 0
            finished = self.current_stage[agents] >= route_lengths[agents]
            self.reached[agents[finished]] = True

        # 长时间卡滞时，触发引导系统介入，直接引导到当前阶段目标附近
        stuck = (~self.reached[idx]) & (self.stage_wait[idx] > 18)
        if stuck.any():
            for i in idx[stuck]:
                route = self.routes[i]
                target_i = self.targets[route[min(self.current_stage[i], len(route) - 1)]]
                guided_pos = self._project_to_free_space(target_i + self.rng.normal(0, 0.8, size=2))
                self.positions[i] = guided_pos
                self.current_stage[i] += 1
                self.stage_wait[i] = 0
                self.guidance_interventions += 1
                if self.current_stage[i] >= len(route):
                    self.reached[i] = True

    def congestion_index(self):
        _, dist = self._pairwise()
        released = self.release_frames <= self.frame_count
        nearby = np.sum((dist < 4.5) & (dist > 1e-6) & released[None, :], axis=1)
        dense_count = int(np.sum((nearby >= 4) & released))
        active_count = max(1, int(np.sum(released)))
        return dense_count / active_count


def _draw_station_layout(ax, env: HubTransferAnimator) -> None:
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


def make_animation(output_path: Path, frames=100, fps=20, n_agents=44, seed=42):
    """生成换乘环境动图 GIF。帧数下限 60，行人数下限 8（与历史 main 行为一致）。"""
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

    anim = FuncAnimation(fig, update, frames=max(60, frames), interval=int(1000 / max(5, fps)), blit=False)

    output_path = ensure_parent(output_path)
    plt.subplots_adjust(bottom=0.16)
    writer = PillowWriter(fps=max(5, fps))
    anim.save(output_path, writer=writer)
    plt.close(fig)
    return output_path
