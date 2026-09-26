"""智枢星：动态客流下的综合性交通枢纽智慧换乘引导系统。"""

from .core import (
    HubTransferAnimator,
    NavigationAdapter,
    NavigationMap,
    PassengerGroup,
    ZhiShuXingSystem,
    generate_dynamic_flow,
    make_animation,
    resolve_groups,
    run_guided_simulation,
)
from .llm import LLMAdapter, MockLLMAdapter, SiliconFlowLLMAdapter
from .rl import MADDPGRuntime

__version__ = "2.1.0"

__all__ = [
    "NavigationAdapter",
    "NavigationMap",
    "PassengerGroup",
    "resolve_groups",
    "generate_dynamic_flow",
    "ZhiShuXingSystem",
    "run_guided_simulation",
    "HubTransferAnimator",
    "make_animation",
    "LLMAdapter",
    "MockLLMAdapter",
    "SiliconFlowLLMAdapter",
    "MADDPGRuntime",
    "__version__",
]
