from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol
import json
import time


class LLMAdapter(Protocol):
    def load_model(self, model_id: str, model_path: Optional[str] = None) -> Dict:
        ...

    def fine_tune(self, dataset_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict:
        ...

    def infer(self, prompt: str, context: Optional[Dict] = None) -> str:
        ...


@dataclass
class MockLLMAdapter:
    model_id: str = ""
    model_path: Optional[str] = None
    tuned_artifact: Optional[str] = None
    tuned_config: Dict = field(default_factory=dict)

    def load_model(self, model_id: str, model_path: Optional[str] = None) -> Dict:
        self.model_id = model_id
        self.model_path = model_path
        return {
            "status": "loaded",
            "model_id": model_id,
            "model_path": model_path,
        }

    def fine_tune(self, dataset_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict:
        config = config or {}
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "mock_llm_finetune_metadata.json"
        payload = {
            "base_model": self.model_id or "mock-base-model",
            "dataset_path": dataset_path,
            "config": config,
            "timestamp": int(time.time()),
            "note": "这是接口占位实现，可替换为 LoRA/QLoRA/PEFT 实际流程。",
        }
        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.tuned_artifact = str(artifact)
        self.tuned_config = config
        return {
            "status": "fine_tuned",
            "artifact": str(artifact),
            "config": config,
        }

    def infer(self, prompt: str, context: Optional[Dict] = None) -> str:
        context = context or {}
        queue_level = context.get("queue_level", "中")
        congestion = context.get("congestion", 0.5)
        suggestion = "建议分流至备用安检闸机并提示乘客错峰通行。"
        if congestion < 0.35:
            suggestion = "建议维持当前引导策略，保持主通道通行。"
        elif congestion > 0.75:
            suggestion = "建议立即启动高拥堵预案，开启临时引导栏并限制入口流量。"
        return (
            f"[模型:{self.model_id or 'mock-base-model'}] 已读取请求：{prompt}。"
            f"当前排队等级={queue_level}，拥堵指数={congestion:.2f}。{suggestion}"
        )


class ExistingFeatureBridge(Protocol):
    def run_existing_visualizations(self, output_dir: str) -> List[str]:
        ...
