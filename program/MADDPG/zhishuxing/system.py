from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import sys

import numpy as np

from .adapters import LLMAdapter, MockLLMAdapter
from .navigation import NavigationAdapter, NavigationMap, Point
from .visualization import HubVisualizer


@dataclass
class PassengerGroup:
    name: str
    start: Point
    goal: Point
    via_landmarks: List[str]
    release_time: int
    passengers: int


class ZhiShuXingSystem:
    def __init__(self) -> None:
        self.navigation = NavigationAdapter()
        self.visualizer = HubVisualizer()
        self.llm: LLMAdapter = MockLLMAdapter()
        self.nav_map: Optional[NavigationMap] = None

    def load_navigation(self, file_path: str) -> NavigationMap:
        self.nav_map = self.navigation.load_navigation(file_path)
        return self.nav_map

    def attach_llm(self, model_id: str, model_path: Optional[str] = None) -> Dict:
        return self.llm.load_model(model_id=model_id, model_path=model_path)

    def fine_tune_llm(self, dataset_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict:
        return self.llm.fine_tune(dataset_path=dataset_path, output_dir=output_dir, config=config)

    def plan_guidance(self, groups: List[PassengerGroup]) -> Dict[str, List[Point]]:
        routes: Dict[str, List[Point]] = {}
        for group in groups:
            route = self.navigation.plan_landmark_path(
                start=group.start,
                via=group.via_landmarks,
                goal=group.goal,
            )
            routes[group.name] = route
        return routes

    def generate_dynamic_flow(self, groups: List[PassengerGroup], steps: int = 60, seed: int = 42) -> np.ndarray:
        if self.nav_map is None:
            raise RuntimeError("请先加载导航图。")

        rng = np.random.default_rng(seed)
        flow = np.zeros((self.nav_map.height, self.nav_map.width), dtype=float)
        routes = self.plan_guidance(groups)

        for step in range(steps):
            for group in groups:
                if step < group.release_time:
                    continue
                route = routes[group.name]
                if not route:
                    continue
                idx = min(step - group.release_time, len(route) - 1)
                x, y = route[idx]
                flow[y, x] += group.passengers * (0.9 + 0.25 * rng.random())

                for _ in range(2):
                    nx = np.clip(x + int(rng.integers(-1, 2)), 0, self.nav_map.width - 1)
                    ny = np.clip(y + int(rng.integers(-1, 2)), 0, self.nav_map.height - 1)
                    flow[ny, nx] += group.passengers * 0.08 * rng.random()

        flow = flow / max(flow.max(), 1.0)
        return flow

    def generate_guidance_text(self, flow_grid: np.ndarray) -> str:
        congestion = float(flow_grid.mean() + flow_grid.max()) / 2.0
        queue_level = "高" if congestion > 0.75 else "中" if congestion > 0.45 else "低"
        prompt = "请根据实时客流，生成枢纽换乘引导策略。"
        return self.llm.infer(prompt=prompt, context={"queue_level": queue_level, "congestion": congestion})

    def render_dashboard(
        self,
        groups: List[PassengerGroup],
        output_png: str,
        title: str = "智枢星：动态客流下综合交通枢纽智慧换乘引导",
    ) -> Dict:
        if self.nav_map is None:
            raise RuntimeError("请先加载导航图。")

        routes = self.plan_guidance(groups)
        flow = self.generate_dynamic_flow(groups)
        image_path = self.visualizer.render_snapshot(
            flow_grid=flow,
            blocked=self.nav_map.blocked,
            routes=routes,
            output_file=output_png,
            title=title,
        )
        guidance_text = self.generate_guidance_text(flow)
        return {
            "image": image_path,
            "guidance": guidance_text,
            "flow_mean": float(flow.mean()),
            "flow_peak": float(flow.max()),
        }

    def run_existing_features(self, workspace_root: str, output_dir: str) -> List[str]:
        scripts = [
            "plot_results.py",
            "plot_congestion_heatmap.py",
            "plot_transfer_time_distribution.py",
            "animate_transfer_env.py",
        ]
        root = Path(workspace_root)
        generated: List[str] = []

        for script in scripts:
            script_path = root / "MADDPG" / script
            if not script_path.exists():
                continue
            try:
                subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(root),
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                pass

        output_path = Path(output_dir)
        if output_path.exists():
            for file in output_path.glob("*.png"):
                generated.append(str(file))
            for file in output_path.glob("*.gif"):
                generated.append(str(file))
        return sorted(set(generated))
