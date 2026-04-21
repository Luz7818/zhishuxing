from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import traceback
import csv

from flask import Flask, jsonify, render_template, request, send_from_directory
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

CURRENT_DIR = Path(__file__).resolve().parent
MADDPG_DIR = CURRENT_DIR.parent
if str(MADDPG_DIR) not in sys.path:
    sys.path.insert(0, str(MADDPG_DIR))

from zhishuxing import PassengerGroup, ZhiShuXingSystem


class ZhiShuXingWebService:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.output_dir = workspace_root / "data_train"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system = ZhiShuXingSystem()

        self.default_nav = MADDPG_DIR / "zhishuxing" / "sample_navigation.json"
        self.default_dataset = MADDPG_DIR / "zhishuxing" / "sample_instruction_data.jsonl"
        self.loaded_navigation = ""

        self._ensure_ready()

    def _ensure_ready(self) -> None:
        if not self.loaded_navigation:
            self.system.load_navigation(str(self.default_nav))
            self.loaded_navigation = str(self.default_nav)
            self.system.attach_llm(model_id="Qwen2.5-7B-Instruct")

    def load_navigation(self, file_path: str) -> Dict[str, Any]:
        nav = self.system.load_navigation(file_path)
        self.loaded_navigation = file_path
        return {
            "width": nav.width,
            "height": nav.height,
            "blocked_count": len(nav.blocked),
            "landmarks": nav.landmarks,
            "file": file_path,
        }

    def load_llm(self, model_id: str, model_path: str | None = None) -> Dict[str, Any]:
        return self.system.attach_llm(model_id=model_id, model_path=model_path)

    def fine_tune(self, dataset_path: str | None = None, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        dataset = dataset_path or str(self.default_dataset)
        return self.system.fine_tune_llm(dataset_path=dataset, output_dir=str(self.output_dir), config=config or {})

    def simulate_finetune_metrics(self, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cfg = config or {}
        epochs = max(5, int(cfg.get("epochs", 30)))
        seed = int(cfg.get("seed", 42))
        loss_noise = float(cfg.get("loss_noise", 0.04))
        metric_noise = float(cfg.get("metric_noise", 0.012))

        start_loss = float(cfg.get("start_loss", 2.8))
        end_loss = float(cfg.get("end_loss", 0.42))

        rng = np.random.default_rng(seed)
        x = np.arange(1, epochs + 1)

        decay = np.exp(-np.linspace(0, 3.4, epochs))
        loss = end_loss + (start_loss - end_loss) * decay + rng.normal(0.0, loss_noise, epochs)
        loss = np.clip(loss, 0.05, None)

        def rising_curve(max_value: float, base_value: float, speed: float) -> np.ndarray:
            curve = base_value + (max_value - base_value) * (1 - np.exp(-np.linspace(0, speed, epochs)))
            curve = curve + rng.normal(0.0, metric_noise, epochs)
            curve = np.clip(curve, 0.0, 1.0)
            return np.maximum.accumulate(curve)

        bleu4 = rising_curve(max_value=0.48, base_value=0.08, speed=2.6)
        rouge1 = rising_curve(max_value=0.63, base_value=0.2, speed=2.3)
        rouge_l = rising_curve(max_value=0.58, base_value=0.18, speed=2.1)

        csv_file = self.output_dir / "finetune_metrics_simulated.csv"
        image_file = self.output_dir / "finetune_metrics_simulated.png"

        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "bleu4", "rouge1", "rougeL"])
            for i in range(epochs):
                writer.writerow(
                    [
                        int(x[i]),
                        round(float(loss[i]), 6),
                        round(float(bleu4[i]), 6),
                        round(float(rouge1[i]), 6),
                        round(float(rouge_l[i]), 6),
                    ]
                )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)

        axes[0].plot(x, loss, color="#ef4444", linewidth=2.2, marker="o", markersize=3)
        axes[0].set_title("Fine-tuning Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.28, linestyle="--")

        axes[1].plot(x, bleu4, label="BLEU-4", color="#2563eb", linewidth=2.2)
        axes[1].plot(x, rouge1, label="ROUGE-1", color="#16a34a", linewidth=2.2)
        axes[1].plot(x, rouge_l, label="ROUGE-L", color="#f59e0b", linewidth=2.2)
        axes[1].set_title("Text Quality Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(alpha=0.28, linestyle="--")
        axes[1].legend(loc="lower right")

        fig.suptitle("DeepSeek-R1-Distill-Qwen-7B 微调模拟指标", fontsize=12)
        fig.tight_layout()
        fig.savefig(image_file)
        plt.close(fig)

        return {
            "model_id": "DeepSeek-R1-Distill-Qwen-7B",
            "dataset": "爬取的小红书换乘语料（模拟）",
            "epochs": epochs,
            "seed": seed,
            "final": {
                "loss": round(float(loss[-1]), 4),
                "bleu4": round(float(bleu4[-1]), 4),
                "rouge1": round(float(rouge1[-1]), 4),
                "rougeL": round(float(rouge_l[-1]), 4),
            },
            "csv_url": f"/outputs/{csv_file.name}",
            "image_url": f"/outputs/{image_file.name}",
        }

    def plan_path(self, start: List[int], goal: List[int], via: List[str] | None = None) -> Dict[str, Any]:
        route = self.system.navigation.plan_landmark_path(
            start=(int(start[0]), int(start[1])),
            via=via or [],
            goal=(int(goal[0]), int(goal[1])),
        )
        return {
            "length": len(route),
            "route": route,
        }

    def run_dashboard(self, groups_payload: List[Dict[str, Any]], title: str | None = None) -> Dict[str, Any]:
        groups = [
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

        image_file = self.output_dir / "zhishuxing_web_dashboard.png"
        result = self.system.render_dashboard(groups=groups, output_png=str(image_file), title=title or "智枢星网页控制台")
        return {
            **result,
            "image_url": f"/outputs/{image_file.name}",
        }

    def run_existing_features(self) -> Dict[str, Any]:
        files = self.system.run_existing_features(
            workspace_root=str(self.workspace_root),
            output_dir=str(self.output_dir),
        )
        return {
            "count": len(files),
            "files": files,
        }


def create_app() -> Flask:
    workspace_root = MADDPG_DIR.parent
    app = Flask(
        __name__,
        template_folder=str(CURRENT_DIR / "templates"),
        static_folder=str(CURRENT_DIR / "static"),
    )
    service = ZhiShuXingWebService(workspace_root=workspace_root)

    @app.get("/")
    def home():
        default_groups = [
            {
                "name": "A口进站->地铁",
                "start": [1, 2],
                "goal": [28, 12],
                "via_landmarks": ["security"],
                "release_time": 0,
                "passengers": 15,
            },
            {
                "name": "B口进站->高铁",
                "start": [1, 13],
                "goal": [28, 3],
                "via_landmarks": ["security_backup"],
                "release_time": 8,
                "passengers": 18,
            },
        ]
        return render_template(
            "index.html",
            default_navigation=service.loaded_navigation,
            default_groups=json.dumps(default_groups, ensure_ascii=False, indent=2),
            default_dataset=str(service.default_dataset),
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "navigation": service.loaded_navigation})

    @app.get("/outputs/<path:filename>")
    def outputs(filename: str):
        return send_from_directory(str(service.output_dir), filename)

    @app.post("/api/navigation/load")
    def api_load_navigation():
        try:
            payload = request.get_json(force=True)
            file_path = payload.get("file_path", "")
            if not file_path:
                return jsonify({"error": "缺少 file_path"}), 400
            data = service.load_navigation(file_path)
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500

    @app.post("/api/llm/load")
    def api_load_llm():
        try:
            payload = request.get_json(force=True)
            model_id = payload.get("model_id", "")
            if not model_id:
                return jsonify({"error": "缺少 model_id"}), 400
            model_path = payload.get("model_path")
            data = service.load_llm(model_id=model_id, model_path=model_path)
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500

    @app.post("/api/llm/fine_tune")
    def api_fine_tune():
        try:
            payload = request.get_json(force=True)
            data = service.fine_tune(
                dataset_path=payload.get("dataset_path"),
                config=payload.get("config", {}),
            )
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500

    @app.post("/api/llm/simulate_metrics")
    def api_simulate_metrics():
        try:
            payload = request.get_json(force=True)
            data = service.simulate_finetune_metrics(config=payload.get("config", {}))
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500

    @app.post("/api/navigation/plan")
    def api_plan_navigation():
        try:
            payload = request.get_json(force=True)
            data = service.plan_path(
                start=payload.get("start"),
                goal=payload.get("goal"),
                via=payload.get("via", []),
            )
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500

    @app.post("/api/dashboard/run")
    def api_run_dashboard():
        try:
            payload = request.get_json(force=True)
            groups = payload.get("groups", [])
            if not groups:
                return jsonify({"error": "缺少 groups"}), 400
            title = payload.get("title")
            data = service.run_dashboard(groups_payload=groups, title=title)
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500

    @app.post("/api/features/run_existing")
    def api_run_existing():
        try:
            data = service.run_existing_features()
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500

    return app


app = create_app()
