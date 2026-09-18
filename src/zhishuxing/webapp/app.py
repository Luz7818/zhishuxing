"""Flask 应用工厂与全部 API 路由。

端点契约与历史版本一致（含 RL 端点），新增：
- POST /api/plan：真实路线规划（engine=amap 高德 / engine=hub 枢纽内 A*+RL）
- GET  /mobile 与 /mobile_static/<path>：同源服务移动端 PWA（免 CORS、便于 SW 注册）
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from .. import config as cfg
from .service import ZhiShuXingWebService


def create_app(service: ZhiShuXingWebService | None = None) -> Flask:
    service = service or ZhiShuXingWebService()
    webapp_dir = Path(__file__).resolve().parent

    app = Flask(
        __name__,
        template_folder=str(webapp_dir / "templates"),
        static_folder=str(webapp_dir / "static"),
    )

    def ok(data):
        return jsonify({"ok": True, "data": data})

    def fail(exc: Exception, status: int = 500):
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), status

    # ------------------------------------------------------------ 页面

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            default_navigation=service.loaded_navigation,
            default_groups=json.dumps(service.default_groups(), ensure_ascii=False, indent=2),
            default_dataset=str(service.default_dataset),
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "navigation": service.loaded_navigation})

    @app.get("/outputs/<path:filename>")
    def outputs(filename: str):
        return send_from_directory(str(service.output_dir), filename)

    @app.get("/mobile")
    def mobile():
        return send_from_directory(str(cfg.paths.mobile_dir), "mobile_app.html")

    @app.get("/mobile/<path:filename>")
    def mobile_static(filename: str):
        # sw.js / manifest / 图标 等资源与页面同前缀，保证 ServiceWorker scope 覆盖 /mobile/
        return send_from_directory(str(cfg.paths.mobile_dir), filename)

    # ------------------------------------------------------------ 导航

    @app.post("/api/navigation/load")
    def api_load_navigation():
        try:
            payload = request.get_json(force=True)
            file_path = payload.get("file_path", "")
            if not file_path:
                return jsonify({"error": "缺少 file_path"}), 400
            return ok(service.load_navigation(file_path))
        except Exception as exc:
            return fail(exc)

    @app.post("/api/navigation/plan")
    def api_plan_navigation():
        try:
            payload = request.get_json(force=True)
            data = service.plan_path(
                start=payload.get("start"),
                goal=payload.get("goal"),
                via=payload.get("via", []),
            )
            return ok(data)
        except Exception as exc:
            return fail(exc)

    # ------------------------------------------------------------ LLM

    @app.post("/api/llm/load")
    def api_load_llm():
        try:
            payload = request.get_json(force=True)
            model_id = payload.get("model_id", "")
            if not model_id:
                return jsonify({"error": "缺少 model_id"}), 400
            data = service.load_llm(
                model_id=model_id,
                model_path=payload.get("model_path"),
                prefer_real=bool(payload.get("prefer_real", False)),
            )
            return ok(data)
        except Exception as exc:
            return fail(exc)

    @app.post("/api/llm/fine_tune")
    def api_fine_tune():
        try:
            payload = request.get_json(force=True)
            data = service.fine_tune(
                dataset_path=payload.get("dataset_path"),
                config=payload.get("config", {}),
            )
            return ok(data)
        except Exception as exc:
            return fail(exc)

    @app.post("/api/llm/simulate_metrics")
    def api_simulate_metrics():
        try:
            payload = request.get_json(force=True)
            data = service.simulate_finetune_metrics(config=payload.get("config", {}))
            return ok(data)
        except Exception as exc:
            return fail(exc)

    # ------------------------------------------------------------ MADDPG RL

    @app.get("/api/rl/status")
    def api_rl_status():
        try:
            return ok(service.rl_status())
        except Exception as exc:
            return fail(exc)

    @app.post("/api/rl/load_policy")
    def api_rl_load_policy():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            return ok(service.rl_load_policy(payload.get("checkpoint_dir")))
        except Exception as exc:
            return fail(exc)

    @app.post("/api/rl/act")
    def api_rl_act():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            observations = payload.get("observations")
            if not isinstance(observations, list):
                return jsonify({"error": "缺少 observations（二维数值数组）"}), 400
            return ok(service.rl_act(observations))
        except Exception as exc:
            return fail(exc)

    @app.get("/api/rl/rewards")
    def api_rl_rewards():
        try:
            return ok(service.rl_rewards())
        except Exception as exc:
            return fail(exc)

    @app.post("/api/rl/simulate")
    def api_rl_simulate():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            data = service.rl_simulate(
                groups_payload=payload.get("groups"),
                config=payload.get("config", {}),
            )
            return ok(data)
        except Exception as exc:
            return fail(exc)

    # ------------------------------------------------------------ 面板与报告

    @app.post("/api/dashboard/run")
    def api_run_dashboard():
        try:
            payload = request.get_json(force=True)
            groups = payload.get("groups", [])
            if not groups:
                return jsonify({"error": "缺少 groups"}), 400
            data = service.run_dashboard(groups_payload=groups, title=payload.get("title"))
            return ok(data)
        except Exception as exc:
            return fail(exc)

    @app.post("/api/features/run_existing")
    def api_run_existing():
        try:
            return ok(service.run_existing_features())
        except Exception as exc:
            return fail(exc)

    # ------------------------------------------------------------ 真实路线规划

    @app.post("/api/plan")
    def api_plan():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            question = payload.get("question", "")
            engine = payload.get("engine", "amap")
            data = service.plan_route(
                question=question,
                engine=engine,
                prefs=payload.get("prefs", {}),
            )
            return ok(data)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:
            return fail(exc)

    return app


app = create_app()
