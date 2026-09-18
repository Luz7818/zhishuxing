"""智枢星系统编排：导航、客流、引导文案、可视化面板与既有分析报告联动。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .. import config as cfg
from ..analysis.plotting import HubVisualizer
from ..llm.adapters import LLMAdapter, MockLLMAdapter
from .flow import generate_dynamic_flow, plan_group_routes
from .navigation import NavigationAdapter, NavigationMap
from .scenarios import PassengerGroup


class ZhiShuXingSystem:
    def __init__(self) -> None:
        self.navigation = NavigationAdapter()
        self.visualizer = HubVisualizer()
        self.llm: LLMAdapter = MockLLMAdapter()
        self.nav_map: Optional[NavigationMap] = None

    def load_navigation(self, file_path: str | Path) -> NavigationMap:
        self.nav_map = self.navigation.load_navigation(str(file_path))
        return self.nav_map

    def attach_llm(self, model_id: str, model_path: Optional[str] = None) -> Dict:
        return self.llm.load_model(model_id=model_id, model_path=model_path)

    def fine_tune_llm(self, dataset_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict:
        return self.llm.fine_tune(dataset_path=dataset_path, output_dir=output_dir, config=config)

    def plan_guidance(self, groups: List[PassengerGroup]) -> Dict[str, List]:
        return plan_group_routes(self.navigation, groups)

    def generate_dynamic_flow(self, groups: List[PassengerGroup], steps: int = 60, seed: int = 42) -> np.ndarray:
        if self.nav_map is None:
            raise RuntimeError("请先加载导航图。")
        routes = self.plan_guidance(groups)
        return generate_dynamic_flow(self.nav_map, groups, routes, steps=steps, seed=seed)

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

    def run_reports(self) -> Dict[str, Dict[str, Any]]:
        """在进程内依次运行既有分析报告，返回逐报告结构化结果（不再经 subprocess、不再吞异常）。"""
        from ..analysis import reports

        results: Dict[str, Dict[str, Any]] = {}
        runners = {
            "reward_curve": reports.run_reward_curve_report,
            "congestion_heatmap": reports.run_congestion_report,
            "transfer_time_distribution": reports.run_transfer_time_report,
            "security_queue_comparison": reports.run_security_queue_report,
            "transfer_efficiency_scenarios": reports.run_efficiency_report,
            "finetune_metrics": reports.run_finetune_metrics_report,
            "transfer_env_animation": reports.run_animation_report,
        }
        for name, runner in runners.items():
            try:
                results[name] = runner()
            except Exception as exc:  # 单个报告失败不阻断其余报告
                results[name] = {"ok": False, "error": str(exc), "files": []}
        return results
