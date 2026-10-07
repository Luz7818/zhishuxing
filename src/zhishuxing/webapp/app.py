"""Flask 应用工厂与全部 API 路由。

端点契约与历史版本一致（含 RL 端点），新增：
- POST /api/plan：真实路线规划（engine=amap 高德 / engine=hub 枢纽内 A*+RL）
- GET  /mobile 与 /mobile/<path>：同源服务移动端 PWA（免 CORS、便于 SW 注册）
- GET/POST /api/settings：查看与保存本地 .env 密钥配置（写接口仅限本机）
- GET  /admin/login、POST /api/admin/login|logout：管理端会话（设置 ADMIN_PASSWORD 后启用，
  只锁管理动作端点；乘客端点与 /mobile 不受影响）
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, current_app, jsonify, make_response, render_template, request, send_from_directory

from .. import config as cfg
from .. import settings as settings_store
from . import auth as admin_auth
from .service import ZhiShuXingWebService

# 管理动作端点（全部为 POST）：启用鉴权后需要管理员会话。其余端点属公开域
# （乘客对话/规划、/mobile PWA、演示只读），保持历史行为不变。
ADMIN_POST_PATHS = frozenset(
    {
        "/api/navigation/load",
        "/api/llm/load",
        "/api/llm/fine_tune",
        "/api/llm/simulate_metrics",
        "/api/rl/load_policy",
        "/api/rl/simulate",
        "/api/dashboard/run",
        "/api/features/run_existing",
        "/api/settings",  # 叠加在既有 loopback / settings_writable 双闸之上
    }
)


def create_app(
    service: ZhiShuXingWebService | None = None,
    settings_writable: bool = True,
) -> Flask:
    """settings_writable=False 用于「生产模式且监听非 loopback」：禁用写接口，只留只读状态。"""
    service = service or ZhiShuXingWebService()
    webapp_dir = Path(__file__).resolve().parent

    app = Flask(
        __name__,
        template_folder=str(webapp_dir / "templates"),
        static_folder=str(webapp_dir / "static"),
    )
    app.extensions["service"] = service   # 供 cmd_serve 做 LLM 自动挂载等启动后置步骤使用

    def ok(data):
        return jsonify({"ok": True, "data": data})

    def fail(exc: Exception, status: int = 500):
        # 堆栈只进服务端日志：响应体给客户端回 traceback 会泄露源码路径与依赖版本
        current_app.logger.exception("接口处理失败")
        return jsonify({"ok": False, "error": str(exc)}), status

    def deny(message: str, detail: str, status: int = 403):
        return jsonify({"ok": False, "error": message, "detail": detail}), status

    # ------------------------------------------------------------ 管理端鉴权（ADMIN_PASSWORD 设置后启用）

    admin_guard = admin_auth.new_guard()
    app.extensions["admin_auth"] = admin_guard   # None = 未设置口令,鉴权关闭(历史行为)

    @app.before_request
    def admin_gate():
        """只拦管理动作端点;公开域(乘客/演示/移动端)与未启用鉴权时一律直通。"""
        if admin_guard is None or request.method != "POST":
            return None
        if request.path not in ADMIN_POST_PATHS:
            return None
        header = request.headers.get("Authorization", "")
        token = header[7:].strip() if header.startswith("Bearer ") else request.cookies.get(admin_auth.COOKIE_NAME, "")
        if admin_auth.parse_token(admin_guard, token):
            return None
        return deny(
            "管理操作需要登录",
            "该接口属于管理端点：请先在 /admin/login 页面登录，或携带登录返回的 "
            "Authorization: Bearer <令牌>。乘客端点不受影响。",
            401,
        )

    # ------------------------------------------------------------ 页面

    @app.get("/")
    def home():
        amap_conf = cfg.amap_config()
        resp = make_response(
            render_template(
                "index.html",
                amap_js_key=amap_conf["js_key"] or "",
                amap_security_code=amap_conf["security_code"] or "",
                settings_writable=settings_writable,
            )
        )
        # 首页禁用启发式缓存:静态资源带版本号,页面本身必须每次取最新
        resp.headers["Cache-Control"] = "no-cache"
        return resp

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

    @app.get("/api/navigation/grid")
    def api_navigation_grid():
        try:
            return ok(service.grid())
        except Exception as exc:
            return fail(exc)

    @app.get("/api/scenarios")
    def api_scenarios():
        try:
            return ok(service.default_groups())
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

    # ------------------------------------------------------------ 对话式换乘助手

    @app.post("/api/chat")
    def api_chat():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            message = (payload.get("message") or "").strip()
            if not message:
                return jsonify({"error": "缺少 message"}), 400
            data = service.chat(
                message=message,
                session_id=payload.get("session_id"),
                prefs=payload.get("prefs", {}),
            )
            return ok(data)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:
            return fail(exc)

    @app.post("/api/chat/reset")
    def api_chat_reset():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            session_id = payload.get("session_id", "")
            if not session_id:
                return jsonify({"error": "缺少 session_id"}), 400
            return ok(service.chat_reset(session_id))
        except Exception as exc:
            return fail(exc)

    # ------------------------------------------------------------ 密钥配置（本地 .env）

    @app.get("/api/settings")
    def api_get_settings():
        """只读状态：所有值一律掩码，故对局域网内的移动端 PWA 也开放。"""
        try:
            return ok(settings_store.read_state())
        except Exception as exc:
            return fail(exc)

    @app.post("/api/settings")
    def api_save_settings():
        """写入 workspace 根 .env 并热重载；仅接受本机（loopback）请求。"""
        if not settings_store.is_loopback(request.remote_addr):
            return deny(
                "密钥配置仅允许本机修改",
                f"请求来源 {request.remote_addr or '未知'} 不是 loopback；"
                "请在本机浏览器操作，或直接在 .env 中填写后重启服务。",
            )
        if not settings_writable:
            return deny(
                "生产模式下已禁用密钥写入",
                "服务以 waitress + 非 loopback 地址监听，写入接口默认关闭；"
                "请改用环境变量或 workspace 根目录 .env 配置后重启服务。",
            )
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "请求体应为 {配置项: 值} 形式的 JSON 对象"}), 400
        try:
            return ok(settings_store.save_settings(payload))
        except settings_store.SettingError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:
            return fail(exc)

    # ------------------------------------------------------------ 管理端会话

    @app.post("/api/admin/login")
    def api_admin_login():
        if admin_guard is None:
            return ok({"enabled": False, "message": "未设置 ADMIN_PASSWORD，管理端鉴权未启用，无需登录。"})
        payload = request.get_json(force=True, silent=True) or {}
        try:
            token = admin_auth.authenticate(admin_guard, str(payload.get("password") or ""))
        except admin_auth.AuthError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 401
        resp = jsonify(
            {"ok": True, "data": {"token": token, "expires_at": admin_auth.token_expires_at(token)}}
        )
        # Cookie 与 body 里的 token 是同一枚:浏览器用 Cookie,脚本/CI 用 Bearer
        resp.set_cookie(
            admin_auth.COOKIE_NAME,
            token,
            max_age=admin_auth.SESSION_TTL,
            httponly=True,
            samesite="Lax",
            path="/",
        )
        return resp

    @app.post("/api/admin/logout")
    def api_admin_logout():
        resp = jsonify({"ok": True, "data": {"logged_out": True}})
        resp.delete_cookie(admin_auth.COOKIE_NAME, path="/")
        return resp

    @app.get("/admin/login")
    def admin_login_page():
        resp = send_from_directory(str(webapp_dir / "static"), "admin_login.html")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    return app


_APP_SINGLETON: "Flask | None" = None


def __getattr__(name: str) -> "Flask":
    """PEP 562 惰性模块属性:`webapp.app` 的 `app` 在首次访问时才构建。

    构建应用要加载导航图、扫描模型目录、建 BM25 索引 —— 模块级直接
    `app = create_app()` 会让任何 `import zhishuxing.webapp.app`(哪怕只为
    拿一个工具函数)都付出全系统加载代价。现在只有真正以 `webapp.app:app`
    作 WSGI 入口或显式访问 `app` 属性时才构建一次并缓存;`create_app()` /
    `cmd_smoke` 等工厂用法不受影响,各自拿到独立实例。
    """
    global _APP_SINGLETON
    if name == "app":
        if _APP_SINGLETON is None:
            _APP_SINGLETON = create_app()
        return _APP_SINGLETON
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
