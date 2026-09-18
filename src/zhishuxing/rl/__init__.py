"""rl 子包：训练依赖（torch / mlagents_envs）为可选，全部公开符号惰性导入。"""

_LAZY = {
    "MADDPG": (".agents", "MADDPG"),
    "MATD3": (".agents", "MATD3"),
    "ReplayBuffer": (".buffer", "ReplayBuffer"),
    "Env": (".envs", "Env"),
    "Space": (".envs", "Space"),
    "Actor": (".networks", "Actor"),
    "Critic_MADDPG": (".networks", "Critic_MADDPG"),
    "Critic_MATD3": (".networks", "Critic_MATD3"),
    "Runner": (".runner", "Runner"),
    "MADDPGRuntime": (".runtime", "MADDPGRuntime"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr = _LAZY[name]
    return getattr(import_module(module_name, __name__), attr)
