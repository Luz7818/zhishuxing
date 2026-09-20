"""大模型适配层：统一接口 + Mock 占位 + SiliconFlow（OpenAI 兼容）真实实现。

密钥一律从环境变量读取（见 config.siliconflow_config），代码中不存在真实密钥。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from .. import config as cfg


class LLMAdapter(Protocol):
    def load_model(self, model_id: str, model_path: Optional[str] = None) -> Dict:
        ...

    def fine_tune(self, dataset_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict:
        ...

    def infer(self, prompt: str, context: Optional[Dict] = None) -> str:
        ...

    def chat(
        self,
        messages: list,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """多轮对话补全:messages 为 [{role, content}] 列表。

        json_mode=True 时要求模型输出 JSON 对象(OpenAI response_format)。
        Mock 实现返回确定性占位文本,保证离线链路可运行。
        """
        ...


@dataclass
class MockLLMAdapter:
    """接口占位实现，可替换为 LoRA/QLoRA/PEFT 实际微调流程。

    mock=True 供上层判断:依赖模型智能的环节(需求解析/对话合成)在 Mock 下
    走确定性模板,保证无密钥时全链路离线可演示。
    """

    mock = True

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

    def chat(
        self,
        messages: list,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """确定性占位对话:回显最后一条用户消息要点,保证离线链路可运行。"""
        last_user = ""
        for message in reversed(messages or []):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break
        suffix = "（要求 JSON 输出）" if json_mode else ""
        return f"[Mock模型] 已收到消息{suffix}：{last_user[:120]}"


@dataclass
class SiliconFlowLLMAdapter:
    """SiliconFlow OpenAI 兼容接口适配器（也适用于任何 OpenAI 兼容服务）。

    未配置 API Key 时 load_model 会抛出 RuntimeError，由调用方决定降级到 Mock。
    """

    mock = False

    model_id: str = ""
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    base_url: str = "https://api.siliconflow.cn/v1"
    temperature: float = 0.7
    max_tokens: int = 512
    _client: Any = field(default=None, repr=False)

    def load_model(self, model_id: str, model_path: Optional[str] = None) -> Dict:
        conf = cfg.siliconflow_config()
        self.api_key = self.api_key or conf["api_key"]
        self.base_url = conf["base_url"]
        self.model_id = model_id or conf["model"] or ""
        self.model_path = model_path
        if not self.api_key:
            raise RuntimeError(
                "未配置 SILICONFLOW_API_KEY 环境变量；请配置密钥或使用 MockLLMAdapter。"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("使用 SiliconFlowLLMAdapter 需要安装 openai 包（pip install .[llm]）。") from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return {
            "status": "loaded",
            "model_id": self.model_id,
            "base_url": self.base_url,
            "model_path": model_path,
        }

    def fine_tune(self, dataset_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict:
        """云端微调不在本适配器范围内：记录任务元数据，提示用平台控制台或替换为 PEFT 流程。"""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "llm_finetune_metadata.json"
        payload = {
            "base_model": self.model_id,
            "dataset_path": dataset_path,
            "config": config or {},
            "provider": "siliconflow",
            "timestamp": int(time.time()),
            "note": "云端微调请在服务商控制台执行，或参考 adapters 接口实现本地 LoRA/PEFT 流程。",
        }
        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "metadata_recorded", "artifact": str(artifact), "config": config or {}}

    def infer(self, prompt: str, context: Optional[Dict] = None) -> str:
        if self._client is None:
            raise RuntimeError("模型未加载，请先调用 load_model。")
        context = context or {}
        queue_level = context.get("queue_level", "中")
        congestion = context.get("congestion", 0.5)
        system_prompt = (
            "你是综合交通枢纽的智慧换乘引导助手，根据实时客流生成简明中文引导策略，"
            "包含分流建议、安检安排与乘客提示。"
        )
        user_prompt = f"{prompt}\n当前排队等级：{queue_level}；拥堵指数：{congestion:.2f}。"
        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def chat(
        self,
        messages: list,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._client is None:
            raise RuntimeError("模型未加载，请先调用 load_model。")
        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def create_llm_adapter(prefer_real: bool = False) -> LLMAdapter:
    """工厂：prefer_real 且已配置密钥时返回真实适配器，否则返回 Mock。"""
    if prefer_real and cfg.siliconflow_config()["api_key"]:
        return SiliconFlowLLMAdapter()
    return MockLLMAdapter()
