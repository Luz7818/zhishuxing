from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


Point = Tuple[int, int]


@dataclass
class VisualizationConfig:
    figsize: Tuple[float, float] = (11.5, 6.5)
    cmap: str = "YlOrRd"


class HubVisualizer:
    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()

    def render_snapshot(
        self,
        flow_grid: np.ndarray,
        blocked: List[Point],
        routes: Dict[str, List[Point]],
        output_file: str,
        title: str,
    ) -> str:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = flow_grid.shape
        fig, (ax_heat, ax_route) = plt.subplots(1, 2, figsize=self.config.figsize)

        im = ax_heat.imshow(flow_grid, cmap=self.config.cmap, origin="lower")
        ax_heat.set_title("动态客流热力图")
        ax_heat.set_xlabel("X")
        ax_heat.set_ylabel("Y")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        ax_route.set_xlim(-0.5, width - 0.5)
        ax_route.set_ylim(-0.5, height - 0.5)
        ax_route.set_title("智慧换乘引导路径")
        ax_route.set_xlabel("X")
        ax_route.set_ylabel("Y")
        ax_route.set_aspect("equal")
        ax_route.grid(alpha=0.2)

        if blocked:
            blocked_x = [p[0] for p in blocked]
            blocked_y = [p[1] for p in blocked]
            ax_route.scatter(blocked_x, blocked_y, marker="s", color="#4A90E2", s=55, label="阻挡区域")

        colors = ["#D64541", "#16A085", "#8E44AD", "#F39C12", "#2C3E50"]
        for index, (group_name, path) in enumerate(routes.items()):
            if not path:
                continue
            px = [point[0] for point in path]
            py = [point[1] for point in path]
            color = colors[index % len(colors)]
            ax_route.plot(px, py, color=color, linewidth=2.2, label=group_name)
            ax_route.scatter(px[0], py[0], color=color, marker="o", s=35)
            ax_route.scatter(px[-1], py[-1], color=color, marker="*", s=95)

        ax_route.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True, framealpha=0.92)
        fig.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return str(output_path)
