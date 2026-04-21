from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from zhishuxing import PassengerGroup, ZhiShuXingSystem


def main() -> None:
    parser = argparse.ArgumentParser("智枢星演示入口")
    parser.add_argument(
        "--navigation",
        type=str,
        default=str(Path(__file__).resolve().parent / "zhishuxing" / "sample_navigation.json"),
        help="导航图JSON文件路径",
    )
    parser.add_argument("--output_dir", type=str, default="data_train", help="输出目录")
    parser.add_argument("--model_id", type=str, default="Qwen2.5-7B-Instruct", help="大模型标识")
    parser.add_argument("--run_existing", action="store_true", help="是否触发已有可视化脚本")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system = ZhiShuXingSystem()
    nav = system.load_navigation(args.navigation)
    model_info = system.attach_llm(model_id=args.model_id)

    groups = [
        PassengerGroup(
            name="A口进站->地铁",
            start=nav.landmarks["entry_a"],
            goal=nav.landmarks["metro_gate"],
            via_landmarks=["security"],
            release_time=0,
            passengers=15,
        ),
        PassengerGroup(
            name="B口进站->高铁",
            start=nav.landmarks["entry_b"],
            goal=nav.landmarks["rail_gate"],
            via_landmarks=["security_backup"],
            release_time=8,
            passengers=18,
        ),
        PassengerGroup(
            name="A口进站->公交",
            start=nav.landmarks["entry_a"],
            goal=nav.landmarks["bus_gate"],
            via_landmarks=["security"],
            release_time=14,
            passengers=9,
        ),
    ]

    dashboard = system.render_dashboard(
        groups=groups,
        output_png=str(output_dir / "zhishuxing_dashboard.png"),
    )

    tune_info = system.fine_tune_llm(
        dataset_path=str(Path(__file__).resolve().parent / "zhishuxing" / "sample_instruction_data.jsonl"),
        output_dir=str(output_dir),
        config={"method": "LoRA", "epochs": 2, "lr": 2e-4, "rank": 16},
    )

    existing_outputs = []
    if args.run_existing:
        workspace_root = str(Path(__file__).resolve().parent.parent)
        existing_outputs = system.run_existing_features(
            workspace_root=workspace_root,
            output_dir=str(output_dir),
        )

    summary = {
        "model": model_info,
        "dashboard": dashboard,
        "fine_tune": tune_info,
        "existing_outputs": existing_outputs,
    }

    summary_path = output_dir / "zhishuxing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("智枢星系统演示完成。")
    print(f"- 导航图: {args.navigation}")
    print(f"- 可视化输出: {dashboard['image']}")
    print(f"- 策略建议: {dashboard['guidance']}")
    print(f"- 汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
