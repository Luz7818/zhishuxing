"""统一命令行入口：demo / train / analyze / simulate / animate / serve / smoke / doctor。

用法示例：
    zhishuxing demo --reports
    zhishuxing analyze --report all
    zhishuxing simulate --max_steps 200
    zhishuxing train --algorithm MADDPG --max_train_steps 500000
    zhishuxing serve --host 0.0.0.0 --port 7860 --production
    zhishuxing doctor          # 检查密钥配置与当前降级项（不输出明文）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import config as cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zhishuxing", description="智枢星：综合交通枢纽智慧换乘引导系统")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="一键演示：导航→客流面板→模拟微调→汇总")
    demo.add_argument("--output_dir", type=Path, default=None, help="输出目录（默认 data/outputs）")
    demo.add_argument("--model_id", type=str, default="Qwen2.5-7B-Instruct")
    demo.add_argument("--reports", action="store_true", help="同时运行全部分析报告")

    train = sub.add_parser("train", help="连接 Unity 运行 MADDPG/MATD3 训练")
    train.add_argument("--config", type=Path, default=None, help="训练配置 JSON（默认 configs/training.json）")
    train.add_argument("--mlagents_file", type=str, default=None, help="Unity 可执行文件；留空连接 Editor")
    train.add_argument("--behavior_name", type=str, default=None)
    train.add_argument("--algorithm", type=str, default=None, choices=["MADDPG", "MATD3"])
    train.add_argument("--max_train_steps", type=int, default=None)
    train.add_argument("--episode_limit", type=int, default=None)
    train.add_argument("--evaluate_freq", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--number", type=int, default=None)

    analyze = sub.add_parser("analyze", help="运行分析报告")
    analyze.add_argument(
        "--report",
        type=str,
        default="all",
        choices=["reward", "heatmap", "transfer", "queue", "efficiency", "finetune", "animation", "all"],
    )

    simulate = sub.add_parser("simulate", help="运行 MADDPG（或启发式）引导的多智能体仿真")
    simulate.add_argument("--groups", type=Path, default=None, help="群组 JSON 路径；默认用 configs/scenarios.json")
    simulate.add_argument("--max_steps", type=int, default=240)
    simulate.add_argument("--agents_per_group", type=int, default=6)
    simulate.add_argument("--seed", type=int, default=42)

    animate = sub.add_parser("animate", help="生成枢纽换乘环境动图")
    animate.add_argument("--frames", type=int, default=60)
    animate.add_argument("--fps", type=int, default=20)
    animate.add_argument("--n_agents", type=int, default=44)
    animate.add_argument("--seed", type=int, default=42)

    serve = sub.add_parser("serve", help="启动 Web 控制台（Flask/waitress）")
    serve.add_argument("--host", type=str, default="0.0.0.0")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--debug", action="store_true")
    serve.add_argument("--production", action="store_true", help="使用 waitress 生产托管")
    serve.add_argument(
        "--allow-remote-settings",
        action="store_true",
        help="生产模式监听非本机地址时仍开放密钥写入接口（默认关闭）",
    )

    sub.add_parser("smoke", help="运行 Web API 冒烟检查")
    doctor = sub.add_parser("doctor", help="检查密钥配置项状态与当前降级行为（退出码 0=就绪 / 1=有缺失）")
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="把可选项（LLM 端点/模型/超时）留空也算作缺失",
    )

    kb_ingest = sub.add_parser("kb-ingest", help="把换乘经验文档(txt/md/html)入库为 JSONL 语料")
    kb_ingest.add_argument("--src", type=Path, default=None, help="源文档目录（默认 data/transfer_kb/shenzhen_north）")
    kb_ingest.add_argument("--out", type=Path, default=None, help="输出语料（默认 data/transfer_kb/corpus.jsonl）")
    kb_ingest.add_argument("--hub", type=str, default="shenzhen_north", help="枢纽标识（写入每条语料的 hub 字段）")
    kb_ingest.add_argument("--query", type=str, default=None, help="入库后用该查询自检检索效果")

    return parser


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_demo(args) -> int:
    from .core.navigation import NavigationAdapter
    from .core.scenarios import load_scenarios, resolve_groups
    from .core.system import ZhiShuXingSystem

    output_dir = Path(args.output_dir) if args.output_dir else cfg.paths.outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    system = ZhiShuXingSystem()
    navigation = NavigationAdapter()
    navigation.load_navigation(str(cfg.paths.navigation_config))
    system.nav_map = navigation.map
    system.navigation = navigation

    model_info = system.attach_llm(model_id=args.model_id)

    groups = resolve_groups(load_scenarios(), navigation.map)
    dashboard_file = output_dir / "zhishuxing_dashboard.png"
    dashboard = system.render_dashboard(groups=groups, output_png=str(dashboard_file))

    fine_tune = system.fine_tune_llm(
        dataset_path=str(cfg.paths.configs / "sample_instruction_data.jsonl"),
        output_dir=str(output_dir),
        config={"method": "LoRA", "epochs": 2, "lr": 0.0002, "rank": 16},
    )

    existing_outputs: List[str] = []
    if args.reports:
        for report in system.run_reports().values():
            if report.get("ok"):
                existing_outputs.extend(report.get("files", []))

    summary = {
        "model": model_info,
        "dashboard": {
            "image": dashboard["image"],
            "guidance": dashboard["guidance"],
            "flow_mean": dashboard["flow_mean"],
            "flow_peak": dashboard["flow_peak"],
        },
        "fine_tune": fine_tune,
        "existing_outputs": existing_outputs,
    }
    _write_json(output_dir / "zhishuxing_summary.json", summary)
    print(f"面板已生成: {dashboard['image']}")
    print(f"引导文案: {dashboard['guidance']}")
    print(f"汇总已写入: {output_dir / 'zhishuxing_summary.json'}")
    return 0


def cmd_train(args) -> int:
    from .rl.runner import run_training

    config = cfg.load_training_config(args.config)
    overrides = {
        "mlagents_file": args.mlagents_file,
        "behavior_name": args.behavior_name,
        "algorithm": args.algorithm,
        "max_train_steps": args.max_train_steps,
        "episode_limit": args.episode_limit,
        "evaluate_freq": args.evaluate_freq,
        "seed": args.seed,
        "number": args.number,
    }
    summary = run_training(config, overrides=overrides)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_analyze(args) -> int:
    from .analysis import reports

    runners = {
        "reward": reports.run_reward_curve_report,
        "heatmap": reports.run_congestion_report,
        "transfer": reports.run_transfer_time_report,
        "queue": reports.run_security_queue_report,
        "efficiency": reports.run_efficiency_report,
        "finetune": reports.run_finetune_metrics_report,
        "animation": reports.run_animation_report,
    }
    if args.report == "all":
        selected = runners
    else:
        selected = {args.report: runners[args.report]}

    failures = 0
    for name, runner in selected.items():
        try:
            result = runner()
            print(f"[{name}] OK -> {result.get('files')}")
        except Exception as exc:
            failures += 1
            print(f"[{name}] FAILED: {exc}")
    return 1 if failures else 0


def cmd_simulate(args) -> int:
    from .core.navigation import NavigationAdapter
    from .core.scenarios import resolve_groups
    from .core.simulation import run_guided_simulation
    from .rl.runtime import MADDPGRuntime

    navigation = NavigationAdapter()
    navigation.load_navigation(str(cfg.paths.navigation_config))

    if args.groups:
        payload = json.loads(Path(args.groups).read_text(encoding="utf-8"))
        payload = payload.get("groups", payload)
    else:
        payload = json.loads(cfg.paths.scenarios_config.read_text(encoding="utf-8"))["groups"]
    groups = resolve_groups(payload, navigation.map)

    runtime = MADDPGRuntime()
    runtime.load_policy()
    result = run_guided_simulation(
        navigation=navigation,
        groups=groups,
        output_png=str(cfg.paths.outputs / "rl_guided_simulation_cli.png"),
        policy=runtime,
        seed=args.seed,
        max_steps=args.max_steps,
        agents_per_group=args.agents_per_group,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "agents"}, ensure_ascii=False, indent=2))
    return 0


def cmd_animate(args) -> int:
    from .core.animation import make_animation

    output = make_animation(
        output_path=cfg.paths.outputs / "transfer_env_demo.gif",
        frames=args.frames,
        fps=args.fps,
        n_agents=args.n_agents,
        seed=args.seed,
    )
    print(f"Saved GIF: {output}")
    return 0


def cmd_serve(args) -> int:
    from . import settings as settings_store
    from .webapp.app import create_app

    # 生产模式且监听非本机地址时，默认关闭密钥写入接口（仍可只读查看状态）
    remote_bind = not settings_store.is_loopback(args.host)
    settings_writable = args.allow_remote_settings or not (args.production and remote_bind)
    app = create_app(settings_writable=settings_writable)
    if args.production:
        from waitress import serve

        print(f"生产模式启动: http://{args.host}:{args.port}")
        if not settings_writable:
            print("密钥写入接口已禁用（监听非本机地址）：请改用环境变量或 .env 配置后重启服务")
        serve(app, host=args.host, port=args.port)
    else:
        print(f"开发模式启动: http://127.0.0.1:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def _map_fallback_label(caps: Dict[str, bool]) -> str:
    """底图为什么没起来：JS Key 与安全密钥必须成对，缺一即回退 Canvas。"""
    if caps["amap_map_blocked_by_security_code"]:
        return "缺 JS 安全密钥，回退 Canvas 折线"
    return "未配置 JS Key，回退 Canvas 折线"


def cmd_doctor(args) -> int:
    """逐项体检密钥配置：未配置的说明去哪申请、填哪个文件、当前降级成什么。

    只输出掩码值；退出码给脚本用（0=就绪，1=有缺失）。
    """
    from . import settings as settings_store

    state = settings_store.read_state()
    caps = state["capabilities"]

    print("智枢星 · 配置体检（密钥一律掩码，不输出明文）")
    print(f"  工作区   : {cfg.WORKSPACE_ROOT}")
    env_note = "已存在" if state["env_file_exists"] else f"不存在，请复制 {state['template_file']} 为 .env"
    print(f"  配置文件 : {state['env_file']}（{env_note}）")
    print()

    missing_optional: List[str] = []
    for entry in state["items"]:
        status = "已配置" if entry["configured"] else "未配置"
        mark = "" if entry["required"] else "  （可选）"
        shown = entry["value_masked"] or entry["default"] or "留空即按默认值/降级运行"
        reused = f" [复用 {entry['reused_from']}]" if entry["reused_from"] else ""
        print(f"  [{status}] {entry['key']:<24} {shown}{reused}{mark}")
        if entry["configured"]:
            continue
        if not entry["required"]:
            missing_optional.append(entry["key"])
        print(f"             用途    : {entry['purpose']}")
        print(f"             申请    : {entry['apply_entry']}")
        print(f"             填写    : {state['env_file']} 中的 {entry['key']}=（或控制台「设置」视图保存）")
        print(f"             当前降级: {entry['degrades_to']}")

    plan_label = "可用" if caps["amap_plan"] else "降级为内置枢纽引擎"
    map_label = "可用" if caps["amap_map"] else _map_fallback_label(caps)
    llm_label = "可用" if caps["llm_real"] else "降级为 Mock 确定性模板"
    print()
    print(f"  在线能力 : 真实路线规划 {plan_label} / 高德底图 {map_label} / 真实 LLM {llm_label}")
    print("  离线演示 : 始终可用（Mock 对话 + Canvas 折线 + 内置枢纽引擎），与是否填写密钥无关")

    missing_required = state["missing"]
    blocking = missing_required + (missing_optional if args.strict else [])
    if blocking:
        kinds = "必需项" if missing_required else "可选项"
        print(
            f"  结论     : {kinds}缺失 {len(blocking)} 项 → {', '.join(blocking)}"
            "（离线演示不受影响，仅对应在线能力受限）"
        )
        return 1
    if missing_optional:
        print(f"  结论     : 必需项已全部配置；{len(missing_optional)} 项可选项留空 → {', '.join(missing_optional)}")
    else:
        print("  结论     : 全部配置项均已填写")
    return 0


def cmd_kb_ingest(args) -> int:
    from .llm.kb import TransferKB, ingest_directory

    src = Path(args.src) if args.src else cfg.paths.kb_sources_dir
    out = Path(args.out) if args.out else cfg.paths.kb_corpus
    result = ingest_directory(src_dir=src, out_path=out, hub=args.hub)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.query:
        kb = TransferKB.load(out)
        print(f"语料 {len(kb)} 条,检索自检「{args.query}」:")
        for hit in kb.search(args.query, hub=args.hub, top_k=3):
            print(f"  [{hit['score']:.3f}] {hit['doc']['title']} <- {hit['doc'].get('source', '')}")
    return 0


def cmd_smoke(_args) -> int:
    from .core.navigation import NavigationAdapter
    from .core.scenarios import resolve_groups
    from .webapp.app import create_app

    app = create_app()
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200, health.data

    nav_path = str(cfg.paths.navigation_config)
    load_nav = client.post("/api/navigation/load", json={"file_path": nav_path})
    assert load_nav.status_code == 200, load_nav.data

    plan = client.post("/api/navigation/plan", json={"start": [1, 2], "goal": [28, 12], "via": ["security"]})
    assert plan.status_code == 200, plan.data

    scenario_payload = json.loads(cfg.paths.scenarios_config.read_text(encoding="utf-8"))["groups"]
    nav_map = NavigationAdapter()
    nav_map.load_navigation(nav_path)
    groups = [g.to_payload() for g in resolve_groups(scenario_payload, nav_map.map)]

    run_dash = client.post("/api/dashboard/run", json={"groups": groups, "title": "smoke-test"})
    assert run_dash.status_code == 200, run_dash.data
    image_url = json.loads(run_dash.data.decode("utf-8"))["data"]["image_url"]
    assert client.get(image_url).status_code == 200

    rl_status = client.get("/api/rl/status")
    assert rl_status.status_code == 200, rl_status.data

    rl_act = client.post(
        "/api/rl/act",
        json={"observations": [[0.5, 0.25, 1.0, 0.0, 0.2, 0.1, -0.3, 0.4, 0.0, -0.2]]},
    )
    assert rl_act.status_code == 200, rl_act.data

    rl_sim = client.post("/api/rl/simulate", json={"groups": groups, "config": {"max_steps": 120, "agents_per_group": 3}})
    assert rl_sim.status_code == 200, rl_sim.data
    sim_image_url = json.loads(rl_sim.data.decode("utf-8"))["data"]["image_url"]
    assert client.get(sim_image_url).status_code == 200

    print("Web smoke test passed.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "demo": cmd_demo,
        "train": cmd_train,
        "analyze": cmd_analyze,
        "simulate": cmd_simulate,
        "animate": cmd_animate,
        "serve": cmd_serve,
        "smoke": cmd_smoke,
        "doctor": cmd_doctor,
        "kb-ingest": cmd_kb_ingest,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
