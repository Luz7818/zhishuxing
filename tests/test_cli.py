from __future__ import annotations

import json

from zhishuxing import config as cfg
from zhishuxing.cli import main as cli_main


def test_cli_analyze_finetune(tmp_path, monkeypatch, capsys):
    result = cli_main(["analyze", "--report", "finetune"])
    assert result == 0
    output = capsys.readouterr().out
    assert "[finetune] OK" in output
    assert (cfg.paths.outputs / "finetune_metrics_simulated.png").exists()


def test_cli_simulate_quick(capsys):
    result = cli_main(["simulate", "--max_steps", "80", "--agents_per_group", "2"])
    assert result == 0
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{"):])
    assert payload["agents_total"] == 6  # 3 组 × agents_per_group=2
    assert payload["agents_arrived"] > 0


def test_cli_demo(tmp_path):
    result = cli_main(["demo", "--output_dir", str(tmp_path)])
    assert result == 0
    summary = json.loads((tmp_path / "zhishuxing_summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"model", "dashboard", "fine_tune", "existing_outputs"}
    assert (tmp_path / "zhishuxing_dashboard.png").exists()
