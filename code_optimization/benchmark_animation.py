"""行人仿真 step() 性能基准：基线 vs v1（成对距离矩阵向量化） vs v2（v1+活跃集压缩）。

运行：python code_optimization/benchmark_animation.py
输出：控制台对比 + code_optimization/benchmark_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_animation import BaselineHubTransferAnimator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from zhishuxing.core.animation import HubTransferAnimator as OptimizedCoreAnimator  # noqa: E402


class V1Animator(BaselineHubTransferAnimator):
    """v1：每帧一次性计算成对距离/社会力矩阵（替代 n 次 O(n) 循环）。"""

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

    def step(self, dt=1.25):
        self.frame_count += 1
        local_density = self._local_density()
        social_forces = self._social_forces()

        for i in range(self.n_agents):
            if self.reached[i] or self.frame_count < self.release_frames[i]:
                continue

            p = self.positions[i]
            target = self._current_target(i)
            if target is None:
                self.reached[i] = True
                continue

            to_goal = target - p
            goal_dir = to_goal / (np.linalg.norm(to_goal) + 1e-6)
            repulse = self._wall_repulse(p)
            noise = self.rng.normal(0, 0.06, size=2)

            direction = goal_dir + repulse + social_forces[i] + noise
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            density_penalty = min(0.45, 0.04 * local_density[i])
            type_bias = 0.12 if self.agent_types[i] == "地铁出站" else (0.06 if self.agent_types[i] == "站内换乘" else 0.0)

            speed = max(0.38, 1.45 + type_bias + self.rng.uniform(0.15, 0.75) - density_penalty)
            v = direction * speed

            new_p = self._project_to_free_space(p + v * dt)
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
        _, dist = self._pairwise()
        released = self.release_frames <= self.frame_count
        nearby = np.sum((dist < 4.5) & (dist > 1e-6) & released[None, :], axis=1)
        dense_count = int(np.sum((nearby >= 4) & released))
        active_count = max(1, int(np.sum(released)))
        return dense_count / active_count


class V2Animator(V1Animator):
    """v2：全 step 向量化 + 活跃集压缩（墙体斥力广播、批量噪声/速度、阶段目标查表）。

    与正式实现（src/zhishuxing/core/animation.py）逻辑一致；
    正式实现的独立测量见 main() 中的 optimized_core。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 阶段目标查表 (n, max_stage, 2)
        self._max_stage = max(len(r) for r in self.routes)
        stage_targets = np.zeros((self.n_agents, self._max_stage, 2), dtype=np.float32)
        for i, route in enumerate(self.routes):
            for s, key in enumerate(route):
                stage_targets[i, s] = self.targets[key]
        self._stage_targets = stage_targets
        # 类型加成查表
        bias = np.zeros(self.n_agents, dtype=np.float32)
        bias[self.agent_types == "地铁出站"] = 0.12
        bias[self.agent_types == "站内换乘"] = 0.06
        self._type_bias = bias
        walls = np.asarray(self.wall_blocks, dtype=np.float32)
        self._wall_xy = walls[:, :2]
        self._wall_wh = walls[:, 2:]

    def _wall_repulse_all(self, pts):
        # pts: (m,2) -> (m,2) 斥力
        cx = np.clip(pts[:, None, 0], self._wall_xy[None, :, 0], self._wall_xy[None, :, 0] + self._wall_wh[None, :, 0])
        cy = np.clip(pts[:, None, 1], self._wall_xy[None, :, 1], self._wall_xy[None, :, 1] + self._wall_wh[None, :, 1])
        vec_x = pts[:, None, 0] - cx
        vec_y = pts[:, None, 1] - cy
        dist = np.sqrt(vec_x ** 2 + vec_y ** 2)
        mask = dist < 6.5
        strength = np.where(mask, (6.5 - dist) / (dist + 1e-4) * 0.26, 0.0).astype(np.float32)
        return np.stack(
            [np.sum(vec_x * strength, axis=1), np.sum(vec_y * strength, axis=1)], axis=1
        )

    def _inside_wall_any(self, pts):
        cx = np.clip(pts[:, None, 0], self._wall_xy[None, :, 0], self._wall_xy[None, :, 0] + self._wall_wh[None, :, 0])
        cy = np.clip(pts[:, None, 1], self._wall_xy[None, :, 1], self._wall_xy[None, :, 1] + self._wall_wh[None, :, 1])
        inside = (pts[:, None, 0] >= self._wall_xy[None, :, 0]) & (pts[:, None, 0] <= self._wall_xy[None, :, 0] + self._wall_wh[None, :, 0]) \
            & (pts[:, None, 1] >= self._wall_xy[None, :, 1]) & (pts[:, None, 1] <= self._wall_xy[None, :, 1] + self._wall_wh[None, :, 1])
        return np.any(inside, axis=1)

    def step(self, dt=1.25):
        self.frame_count += 1
        active_idx = np.where((~self.reached) & (self.release_frames <= self.frame_count))[0]
        if active_idx.size == 0:
            return

        # 无目标的 agent 直接标记完成（历史语义：stage 越界即 reached）
        done_mask = self.current_stage[active_idx] >= np.array([len(r) for r in self.routes])[active_idx]
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

        m = active_idx.size
        idx = active_idx
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
                i = idx[k]
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
            finished = self.current_stage[agents] >= np.array([len(r) for r in self.routes])[agents]
            self.reached[agents[finished]] = True

        stuck = (~self.reached[idx]) & (self.stage_wait[idx] > 18)
        if stuck.any():
            for k in idx[stuck]:
                target_k = self.targets[self.routes[k][min(self.current_stage[k], len(self.routes[k]) - 1)]]
                guided_pos = self._project_to_free_space(target_k + self.rng.normal(0, 0.8, size=2))
                self.positions[k] = guided_pos
                self.current_stage[k] += 1
                self.stage_wait[k] = 0
                self.guidance_interventions += 1
                if self.current_stage[k] >= len(self.routes[k]):
                    self.reached[k] = True


def run_steps(animator, steps):
    for _ in range(steps):
        animator.step()


def measure(cls, n_agents, steps, seed=42, repeats=3):
    times = []
    stats = None
    for _ in range(repeats):
        anim = cls(n_agents=n_agents, seed=seed)
        started = time.perf_counter()
        run_steps(anim, steps)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        stats = {
            "reached": int(np.sum(anim.reached)),
            "interventions": int(anim.guidance_interventions),
            "congestion": round(float(anim.congestion_index()), 4),
        }
    return min(times), stats


def main():
    cases = [
        {"n_agents": 44, "steps": 1000, "label": "默认规模 (44人×1000帧)"},
        {"n_agents": 500, "steps": 300, "label": "大规模 (500人×300帧)"},
    ]
    results = {"cases": []}

    for case in cases:
        entry = {"label": case["label"], "n_agents": case["n_agents"], "steps": case["steps"], "impls": {}}
        baseline_time, baseline_stats = measure(BaselineHubTransferAnimator, case["n_agents"], case["steps"])
        v1_time, v1_stats = measure(V1Animator, case["n_agents"], case["steps"])
        v2_time, v2_stats = measure(V2Animator, case["n_agents"], case["steps"])
        core_time, core_stats = measure(OptimizedCoreAnimator, case["n_agents"], case["steps"])
        entry["impls"]["baseline"] = {"sec": round(baseline_time, 3), **baseline_stats}
        entry["impls"]["v1"] = {"sec": round(v1_time, 3), **v1_stats}
        entry["impls"]["v2"] = {"sec": round(v2_time, 3), **v2_stats}
        entry["impls"]["optimized_core"] = {"sec": round(core_time, 3), **core_stats}
        entry["speedup_v1"] = round(baseline_time / v1_time, 2)
        entry["speedup_v2"] = round(baseline_time / v2_time, 2)
        entry["speedup_core"] = round(baseline_time / core_time, 2)

        # 统计等价性校验（社会力改为帧首快照，轨迹允许微小差异）
        entry["equivalence"] = {
            "reached_equal": baseline_stats["reached"] == core_stats["reached"],
            "interventions_delta": abs(baseline_stats["interventions"] - core_stats["interventions"]),
            "congestion_delta": round(abs(baseline_stats["congestion"] - core_stats["congestion"]), 4),
        }
        results["cases"].append(entry)
        print(
            f"{case['label']}: baseline={baseline_time:.2f}s v1={v1_time:.2f}s ({entry['speedup_v1']}x) "
            f"v2={v2_time:.2f}s ({entry['speedup_v2']}x) core={core_time:.2f}s ({entry['speedup_core']}x) | {entry['equivalence']}"
        )

    results_file = Path(__file__).resolve().parent / "benchmark_results.json"
    results_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {results_file}")


if __name__ == "__main__":
    main()
