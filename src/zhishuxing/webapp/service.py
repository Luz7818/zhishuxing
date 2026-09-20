"""Web 服务层：编排 智枢星系统 + MADDPG 运行时 + 真实规划 + 对话式换乘助手。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as cfg
from ..analysis import reports
from ..core.navigation import (
    NavigationMap,
    TAG_LABELS,
    landmark_label,
    resolve_hub_landmark,
)
from ..core.scenarios import PassengerGroup, load_scenarios, resolve_groups
from ..core.system import ZhiShuXingSystem
from ..llm.adapters import SiliconFlowLLMAdapter
from ..llm.assistant import TransferAssistant
from ..llm.kb import TransferKB
from ..llm.profile import profile_from_api_prefs
from ..planning.amap import plan_route as amap_plan_route
from ..rl.runtime import MADDPGRuntime


def parse_groups(groups_payload: List[Dict[str, Any]]) -> List[PassengerGroup]:
    """把 API payload（[x,y] 坐标）解析为 PassengerGroup。"""
    return [
        PassengerGroup(
            name=item["name"],
            start=(int(item["start"][0]), int(item["start"][1])),
            goal=(int(item["goal"][0]), int(item["goal"][1])),
            via_landmarks=item.get("via_landmarks", []),
            release_time=int(item.get("release_time", 0)),
            passengers=int(item.get("passengers", 10)),
        )
        for item in groups_payload
    ]


class ZhiShuXingWebService:
    def __init__(self) -> None:
        cfg.paths.ensure_runtime_dirs()
        self.output_dir = cfg.paths.outputs
        self.system = ZhiShuXingSystem()
        self.rl = MADDPGRuntime()
        self.default_nav = cfg.paths.navigation_config
        self.default_dataset = cfg.paths.configs / "sample_instruction_data.jsonl"
        self.loaded_navigation = ""

        self._ensure_ready()
        self.kb = TransferKB.load_default()
        self.assistant = TransferAssistant(self, kb=self.kb)

    def _ensure_ready(self) -> None:
        if not self.loaded_navigation:
            self.system.load_navigation(self.default_nav)
            self.loaded_navigation = str(self.default_nav)
            self.system.attach_llm(model_id="Qwen2.5-7B-Instruct")
            # 启动时即尝试加载 MADDPG 策略权重；无权重时自动回退启发式并在状态中注明
            self.rl.load_policy()

    # ------------------------------------------------------------ 导航

    def load_navigation(self, file_path: str) -> Dict[str, Any]:
        nav = self.system.load_navigation(file_path)
        self.loaded_navigation = file_path
        return {
            "width": nav.width,
            "height": nav.height,
            "blocked_count": len(nav.blocked),
            "landmarks": {k: list(v) for k, v in nav.landmarks.items()},
            "file": file_path,
        }

    def plan_path(self, start: List[int], goal: List[int], via: Optional[List[str]] = None) -> Dict[str, Any]:
        route = self.system.navigation.plan_landmark_path(
            start=(int(start[0]), int(start[1])),
            via=via or [],
            goal=(int(goal[0]), int(goal[1])),
        )
        return {"length": len(route), "route": [list(p) for p in route]}

    def grid(self) -> Dict[str, Any]:
        """完整导航网格（供前端 Canvas 渲染），含设施语义层。"""
        nav_map = self.system.nav_map
        if nav_map is None:
            raise RuntimeError("请先加载导航图。")
        return {
            "width": nav_map.width,
            "height": nav_map.height,
            "blocked": [list(p) for p in nav_map.blocked],
            "landmarks": {k: list(v) for k, v in nav_map.landmarks.items()},
            "landmark_labels": {k: landmark_label(k) for k in nav_map.landmarks},
            "cell_tags": {tag: [list(p) for p in cells] for tag, cells in nav_map.cell_tags.items()},
            "cell_size_m": nav_map.cell_size_m,
            "file": self.loaded_navigation,
        }

    # ------------------------------------------------------------ LLM

    def load_llm(self, model_id: str, model_path: Optional[str] = None, prefer_real: bool = False) -> Dict[str, Any]:
        if prefer_real:
            try:
                adapter = SiliconFlowLLMAdapter()
                result = adapter.load_model(model_id=model_id, model_path=model_path)
                self.system.llm = adapter
                return result
            except RuntimeError as exc:
                # 无密钥/未装 openai：保留 Mock 并在结果中注明
                result = self.system.attach_llm(model_id=model_id, model_path=model_path)
                result["real_adapter_error"] = str(exc)
                return result
        return self.system.attach_llm(model_id=model_id, model_path=model_path)

    def fine_tune(self, dataset_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        dataset = dataset_path or str(self.default_dataset)
        return self.system.fine_tune_llm(dataset_path=dataset, output_dir=str(self.output_dir), config=config or {})

    def simulate_finetune_metrics(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = config or {}
        result = reports.run_finetune_metrics_report(
            epochs=max(5, int(params.get("epochs", 30))),
            seed=int(params.get("seed", 42)),
        )
        data = reports.generate_finetune_metrics(
            epochs=max(5, int(params.get("epochs", 30))),
            seed=int(params.get("seed", 42)),
            monotonic=True,
        )
        return {
            "model_id": "DeepSeek-R1-Distill-Qwen-7B",
            "dataset": "爬取的小红书换乘语料（模拟）",
            "epochs": int(params.get("epochs", 30)),
            "seed": int(params.get("seed", 42)),
            "final": result["final"],
            "csv_url": f"/outputs/{Path(result['files'][1]).name}",
            "image_url": f"/outputs/{Path(result['files'][0]).name}",
            "series": {k: [round(float(v), 4) for v in data[k]] for k in ("epoch", "loss", "bleu4", "rouge1", "rougeL")},
        }

    # ------------------------------------------------------------ 可视化

    def run_dashboard(self, groups_payload: List[Dict[str, Any]], title: Optional[str] = None) -> Dict[str, Any]:
        groups = parse_groups(groups_payload)
        image_file = self.output_dir / "zhishuxing_web_dashboard.png"
        result = self.system.render_dashboard(groups=groups, output_png=str(image_file), title=title or "智枢星网页控制台")
        return {
            **result,
            "image_url": f"/outputs/{image_file.name}",
        }

    def run_existing_features(self) -> Dict[str, Any]:
        results = self.system.run_reports()
        files: List[str] = []
        for name, item in results.items():
            if item.get("ok"):
                files.extend(item.get("files", []))
        return {"count": len(files), "files": files, "reports": results}

    # ------------------------------------------------------------ MADDPG RL

    def rl_status(self) -> Dict[str, Any]:
        return self.rl.get_status()

    def rl_load_policy(self, checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        return self.rl.load_policy(checkpoint_dir)

    def rl_act(self, observations: List[List[float]]) -> Dict[str, Any]:
        if not observations:
            raise ValueError("缺少 observations")
        return self.rl.act(observations)

    def rl_rewards(self) -> Dict[str, Any]:
        rendered = self.rl.render_reward_curve()
        # 图表需要完整 x/y 序列；render_reward_curve 只返回摘要，需从 reward_series 取全量
        full_series = self.rl.reward_series()["series"]
        result: Dict[str, Any] = {
            "series_count": rendered["series_count"],
            "series": full_series,
        }
        if rendered.get("error"):
            result["error"] = rendered["error"]
        if rendered.get("image"):
            result["image_url"] = f"/outputs/{Path(rendered['image']).name}"
        return result

    def rl_simulate(
        self,
        groups_payload: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = groups_payload if groups_payload else self.default_groups()
        groups = parse_groups(payload)
        params = config or {}

        image_file = self.output_dir / "rl_guided_simulation.png"
        result = self.rl.run_guided_simulation(
            navigation=self.system.navigation,
            groups=groups,
            output_png=str(image_file),
            title=params.get("title", "智枢星 MADDPG 引导仿真"),
            seed=int(params.get("seed", 42)),
            max_steps=int(params.get("max_steps", 240)),
            agents_per_group=int(params.get("agents_per_group", 6)),
        )
        return {
            **result,
            "image_url": f"/outputs/{image_file.name}",
        }

    # ------------------------------------------------------------ 真实路线规划

    def plan_route(self, question: str, engine: str = "amap", prefs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """engine=amap：高德真实路线；engine=hub：枢纽内偏好感知 A* + RL 引导仿真。

        两引擎都会解析需求档案(prefs.note 自由文本 + 结构化开关),档案随结果返回供前端展示。
        """
        profile = profile_from_api_prefs(prefs, self.system.llm)
        if engine == "amap":
            result = amap_plan_route(question, prefs=prefs)
            result["profile"] = profile.to_payload()
            if not profile.is_empty():
                result.setdefault("tips", []).append(
                    f"已按需求「{profile.summary()}」优化建议;市内段以高德策略为准,站内设施请跟随站内指引。"
                )
            return result

        # hub 引擎:解析"从X到Y"中的地标并做偏好感知规划
        from ..planning.amap import extract_od_locally

        od = extract_od_locally(question)
        nav_map: Optional[NavigationMap] = self.system.nav_map
        if nav_map is None:
            raise RuntimeError("请先加载导航图。")

        start = resolve_hub_landmark(od.get("origin_text", ""), nav_map)
        goal = resolve_hub_landmark(od.get("destination_text", ""), nav_map)
        if start is None or goal is None:
            raise ValueError("无法在枢纽导航图中识别起终点地标,可用地标: " + ", ".join(nav_map.landmarks))

        # 明确提及安检 → 硬必经;其余需求走加权代价与软必经(卫生间等)
        via = ["security_backup"] if "备用安检" in question else (["security"] if "安检" in question else [])
        cost_spec = profile.to_cost_spec()
        if via:
            route_points = self.system.navigation.plan_landmark_path(start=start, via=via, goal=goal, cost_spec=cost_spec)
            plan = {
                "route": [list(p) for p in route_points],
                "length": len(route_points),
                "meters": int(round((len(route_points) - 1) * nav_map.cell_size_m)),
                "notes": [],
                "tags_on_path": {},
                "soft_via": {},
                "degraded": False,
            }
        else:
            plan = self.system.navigation.plan_with_preferences(start, goal, cost_spec)

        sim = self.rl_simulate(
            groups_payload=[
                {
                    "name": f"{od.get('origin_text', '')}->{od.get('destination_text', '')}",
                    "start": list(start),
                    "goal": list(goal),
                    "via_landmarks": via,
                    "release_time": 0,
                    "passengers": 10,
                }
            ],
            config={"agents_per_group": 4, "max_steps": 200},
        )

        details = [f"枢纽内路径:约 {plan['meters']} 米({plan['length']} 格)"]
        if not profile.is_empty():
            details.append(f"需求理解:{profile.summary()}")
        details.extend(plan.get("notes", []))
        facility_bits = [
            f"{TAG_LABELS.get(tag, tag)}×{count}" for tag, count in plan.get("tags_on_path", {}).items()
        ]
        if facility_bits:
            details.append("路径经过设施:" + "、".join(facility_bits))
        details.append(f"途经点:{'、'.join(landmark_label(v) for v in via) if via else '无'}")
        details.append(f"RL 策略:{sim.get('policy_source')}")

        tips = ["请跟随站内引导标识与电子屏指引前往目标检票口。"]
        if plan.get("degraded"):
            tips.append("您的通行约束较严格,已自动放宽规划,请注意脚下设施类型。")

        return {
            "engine": "hub",
            "od_source": "landmark",
            "origin_text": od.get("origin_text", ""),
            "destination_text": od.get("destination_text", ""),
            "route": plan["route"],
            "length": plan["length"],
            "meters": plan["meters"],
            "simulation": sim,
            "profile": profile.to_payload(),
            "details": details,
            "tips": tips,
        }

    # ------------------------------------------------------------ 对话式换乘助手

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        prefs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assistant.handle(message, session_id=session_id, prefs=prefs)

    def chat_reset(self, session_id: str) -> Dict[str, Any]:
        return self.assistant.reset(session_id)

    # ------------------------------------------------------------ 场景

    def default_groups(self) -> List[Dict[str, Any]]:
        """configs/scenarios.json 的三组演示场景（坐标已解析）。"""
        nav_map = self.system.nav_map
        resolved = resolve_groups(load_scenarios(), nav_map)
        return [group.to_payload() for group in resolved]
