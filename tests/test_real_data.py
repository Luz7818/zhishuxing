"""data/real 真实数据通路:约定目录发现、来源标注、缺失回退合成。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from zhishuxing.analysis import reports


def _use_real_dir(tmp_path, monkeypatch):
    from zhishuxing import config as cfg

    monkeypatch.setattr(cfg.paths, "data", tmp_path)
    return tmp_path / "real"


def test_resolver_falls_back_to_synthetic_when_missing(tmp_path, monkeypatch):
    real_dir = _use_real_dir(tmp_path, monkeypatch)
    real_dir.mkdir(parents=True)

    resolved = reports.resolve_real_inputs("heatmap")

    assert resolved["kwargs"] == {}
    assert "合成" in resolved["source"]


def test_resolver_picks_up_real_files(tmp_path, monkeypatch):
    real_dir = _use_real_dir(tmp_path, monkeypatch)
    real_dir.mkdir(parents=True)
    (real_dir / "congestion_before.csv").write_text("区域,07:00\nA,1\n", encoding="utf-8")
    (real_dir / "congestion_after.csv").write_text("区域,07:00\nA,0.8\n", encoding="utf-8")
    (real_dir / "transfer_summary.csv").write_text("iteration,p50,p90,max\n1,400,500,650\n", encoding="utf-8")

    heatmap = reports.resolve_real_inputs("heatmap")
    transfer = reports.resolve_real_inputs("transfer")
    queue = reports.resolve_real_inputs("queue")          # 无真实数据约定的报告

    assert heatmap["kwargs"]["before_csv"] == real_dir / "congestion_before.csv"
    assert "真实数据" in heatmap["source"]
    assert transfer["kwargs"]["input_csv"] == real_dir / "transfer_summary.csv"
    assert queue["kwargs"] == {} and "合成" in queue["source"]


def test_congestion_report_runs_on_real_csv(tmp_path, monkeypatch):
    """端到端:真实矩阵 CSV 直接产出三联图,且图题带来源标注。"""
    real_dir = _use_real_dir(tmp_path, monkeypatch)
    real_dir.mkdir(parents=True)
    zones = "区域,07:00,08:00,18:00\n东广场,96,110,102\n地铁通道,112,130,118"
    (real_dir / "congestion_before.csv").write_text(zones + "\n", encoding="utf-8")
    (real_dir / "congestion_after.csv").write_text(zones + "\n", encoding="utf-8")

    result = reports.run_congestion_report(
        before_csv=real_dir / "congestion_before.csv",
        after_csv=real_dir / "congestion_after.csv",
        output=tmp_path / "hm.png",
        rank_output=tmp_path / "rank.png",
        peak_output=tmp_path / "peak.png",
        source_note="真实数据:congestion_before.csv",
    )

    assert result["ok"] is True
    for f in result["files"]:
        assert Path(f).is_file()


def test_transfer_report_labels_real_source(tmp_path):
    rows = "\n".join(f"{i},{500 - i},{600 - i},{700 - i}" for i in range(1, 41))
    csv_path = tmp_path / "transfer_summary.csv"
    csv_path.write_text("iteration,p50,p90,max\n" + rows + "\n", encoding="utf-8")

    result = reports.run_transfer_time_report(
        input_csv=csv_path, output=tmp_path / "tt.png", source_note="真实数据:transfer_summary.csv",
    )

    assert result["ok"] is True and result["points"] == 40
    assert Path(result["files"][0]).is_file()
