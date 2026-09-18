"""CSV 读写工具：全仓库唯一的 CSV IO 实现（历史上 7 处手写重复）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .plotting import ensure_parent


def read_summary_csv(csv_path: Path, min_rows: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取换乘时间分布 CSV（列: iteration,p50,p90,max）。"""
    iterations: List[int] = []
    p50: List[float] = []
    p90: List[float] = []
    max_values: List[float] = []

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"iteration", "p50", "p90", "max"}
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV 需要包含列: iteration,p50,p90,max")

        for row in reader:
            iterations.append(int(row["iteration"]))
            p50.append(float(row["p50"]))
            p90.append(float(row["p90"]))
            max_values.append(float(row["max"]))

    if len(iterations) < max(1, min_rows):
        raise ValueError(f"CSV 样本点过少（{len(iterations)} < {min_rows}），无法稳定分析")

    return (
        np.asarray(iterations, dtype=np.int32),
        np.asarray(p50, dtype=np.float32),
        np.asarray(p90, dtype=np.float32),
        np.asarray(max_values, dtype=np.float32),
    )


def write_summary_csv(csv_path: Path, iterations, p50, p90, max_values) -> None:
    write_csv_rows(
        csv_path,
        fieldnames=["iteration", "p50", "p90", "max"],
        rows=[
            {"iteration": int(i), "p50": float(a), "p90": float(b), "max": float(c)}
            for i, a, b, c in zip(iterations, p50, p90, max_values)
        ],
        fmt={
            "iteration": "{:d}",
            "p50": "{:.4f}",
            "p90": "{:.4f}",
            "max": "{:.4f}",
        },
    )


def read_matrix_csv(csv_path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """读取拥堵矩阵 CSV：首列区域名，首行时段标签。"""
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"CSV 格式错误: {csv_path}")

    header = rows[0]
    time_slots = header[1:]
    zones: List[str] = []
    values: List[List[float]] = []

    for row in rows[1:]:
        if not row:
            continue
        zones.append(row[0])
        values.append([float(x) for x in row[1:]])

    matrix = np.asarray(values, dtype=np.float32)
    return zones, time_slots, matrix


def write_csv_rows(
    csv_path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict],
    fmt: Dict[str, str] | None = None,
) -> Path:
    """通用 DictWriter 写出（utf-8-sig，便于 Excel 打开中文）。fmt 控制每列数字格式。"""
    csv_path = ensure_parent(csv_path)
    fmt = fmt or {}
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (fmt[key].format(value) if key in fmt else value)
                    for key, value in row.items()
                }
            )
    return csv_path
