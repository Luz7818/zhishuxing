"""性能优化基线：与历史 animate_transfer_env.py 逐行一致的原始 HubTransferAnimator。

仅保留 step()/congestion_index() 及其依赖，供 code_optimization/benchmark_animation.py
作为基线对照。优化后的实现位于 src/zhishuxing/core/animation.py。
"""

from __future__ import annotations

import numpy as np


class BaselineHubTransferAnimator:
    def __init__(self, width=140, height=85, n_agents=44, seed=42):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        self.frame_count = 0

        self.wall_blocks = [
            (0, 0, 140, 2), (0, 83, 140, 2), (0, 0, 2, 85), (138, 0, 2, 85),
            (22, 26, 10, 12), (22, 50, 10, 16), (56, 18, 8, 16), (56, 42, 8, 10),
            (56, 60, 8, 18), (100, 22, 14, 18), (100, 52, 14, 14),
        ]
        self.station_zones = {
            "东广场进站口": (4, 58, 16, 20),
            "西广场进站口": (4, 8, 16, 20),
            "安检闸机群": (32, 34, 16, 18),
            "安检闸机群(中部)": (72, 34, 22, 16),
            "地铁4号线": (116, 58, 18, 20),
            "地铁5号线": (116, 8, 18, 20),
        }
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
        base_release = {"东广场": (0, 10), "西广场": (8, 18), "地铁4号线": (16, 30), "地铁5号线": (24, 38), "安检区内": (4, 14)}
        release = np.zeros(self.n_agents, dtype=np.int32)
        for i in range(self.n_agents):
            lo, hi = base_release.get(self.start_groups[i], (0, 12))
            release[i] = int(self.rng.integers(lo, hi + 1))
        return release

    def _build_routes(self):
        routes = []
        for t in self.agent_types:
            if t == "进站换乘":
                route = ["left_pass_top", "security", "security_mid", "metro4"] if self.rng.random() < 0.5 else ["left_pass_bottom", "security", "security_mid", "metro5"]
            elif t == "地铁出站":
                route = ["right_pass_top", "security_mid", "security", "exit_east"] if self.rng.random() < 0.5 else ["right_pass_bottom", "security_mid", "security", "exit_west"]
            else:
                route = ["security", "security_mid", "right_pass_top", "metro4"] if self.rng.random() < 0.5 else ["security", "security_mid", "right_pass_bottom", "metro5"]
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
