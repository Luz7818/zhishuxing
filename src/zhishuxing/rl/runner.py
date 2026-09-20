"""MADDPG/MATD3 训练主循环（原 MADDPG_main.py 的 Runner，路径锚定 workspace）。

保持历史数值行为：训练/评估双环境、deepcopy 动作下发、逐 agent 训练、
噪声线性衰减、按 evaluate_freq 无噪声评估并保存 npy/权重（文件命名格式不变）。
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np
import torch

from .. import config as cfg
from .agents import MADDPG, MATD3
from .buffer import ReplayBuffer
from .envs import Env


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        cfg.paths.ensure_runtime_dirs()

        # Create env
        self.env = Env(
            env_name,
            discrete=False,
            behavior_name=args.behavior_name,
            file_name=args.mlagents_file,
            base_port=args.base_port,
            seed=self.seed,
            no_graphics=args.no_graphics,
            worker_id=0,
        )
        self.env_evaluate = Env(
            env_name,
            discrete=False,
            behavior_name=args.behavior_name,
            file_name=args.mlagents_file,
            base_port=args.base_port,
            seed=self.seed + 1000,
            no_graphics=args.no_graphics,
            worker_id=1,
        )
        args.N = self.env.n  # The number of agents
        args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(args.N)]  # obs dimensions of N agents
        args.action_dim_n = [self.env.action_space[i].shape[0] for i in range(args.N)]  # actions dimensions of N agents
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(args.action_dim_n))

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        if args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
        elif args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
        else:
            raise ValueError(f"未知算法: {args.algorithm}（支持 MADDPG / MATD3）")

        self.replay_buffer = ReplayBuffer(args)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = cfg.paths.runs_dir / "{}_env_{}_number_{}_seed_{}".format(args.algorithm, env_name, number, seed)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            print("未安装 tensorboard，跳过训练曲线记录")
            self.writer = None

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.noise_std = args.noise_std_init  # Initialize noise_std
        self.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    def run(self):
        self.evaluate_policy()

        while self.total_steps < self.args.max_train_steps:
            obs_n = self.env.reset()
            for _ in range(self.args.episode_limit):
                # Each agent selects actions based on its own local observations(add noise for exploration)
                a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                # --------------------------!!!注意！！！这里一定要deepcopy，MPE环境会把a_n乘5-------------------------------------------
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))
                # Store the transition
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)
                obs_n = obs_next_n
                self.total_steps += 1

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = (
                        self.noise_std - self.noise_std_decay
                        if self.noise_std - self.noise_std_decay > self.args.noise_std_min
                        else self.args.noise_std_min
                    )

                if self.replay_buffer.current_size > self.args.batch_size:
                    # Train each agent individually
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy()

                if all(done_n):
                    break

        self.env.close()
        self.env_evaluate.close()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(int(self.args.evaluate_times)):
            obs_n = self.env_evaluate.reset()
            episode_reward = 0
            for _ in range(self.args.episode_limit):
                a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # We do not add noise when evaluating
                obs_next_n, r_n, done_n, _ = self.env_evaluate.step(copy.deepcopy(a_n))
                episode_reward += r_n[0]
                obs_n = obs_next_n
                if all(done_n):
                    break
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t noise_std:{}".format(self.total_steps, evaluate_reward, self.noise_std))
        if self.writer is not None:
            self.writer.add_scalar("evaluate_step_rewards_{}".format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models（文件命名格式与历史一致）
        reward_file = cfg.paths.outputs / "{}_env_{}_number_{}_seed_{}.npy".format(
            self.args.algorithm, self.env_name, self.number, self.seed
        )
        np.save(reward_file, np.array(self.evaluate_rewards))
        for agent_id in range(self.args.N):
            self.agent_n[agent_id].save_model(
                cfg.paths.model_dir, self.env_name, self.args.algorithm, self.number, self.total_steps, agent_id
            )


def build_train_args(config: dict, overrides: Optional[dict] = None):
    """由 configs/training.json + CLI 覆盖项构建训练命名空间。"""
    matd3 = config.get("matd3", {})
    merged = {k: v for k, v in config.items() if k != "matd3"}
    merged.update({k: v for k, v in matd3.items() if k not in merged or v is not None})
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})

    from types import SimpleNamespace

    args = SimpleNamespace(**merged)
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    return args


def run_training(config: dict, overrides: Optional[dict] = None, max_steps_override: Optional[int] = None) -> dict:
    """训练入口：连接 Unity（mlagents_file=None 时连 Editor）。返回训练摘要。"""
    if max_steps_override is not None:
        overrides = dict(overrides or {})
        overrides["max_train_steps"] = max_steps_override
    args = build_train_args(config, overrides)
    runner = Runner(args, env_name=config["env_name"], number=args.number, seed=args.seed)
    started = time.time()
    runner.run()
    return {
        "algorithm": args.algorithm,
        "total_steps": runner.total_steps,
        "evaluations": len(runner.evaluate_rewards),
        "last_evaluate_reward": runner.evaluate_rewards[-1] if runner.evaluate_rewards else None,
        "elapsed_sec": round(time.time() - started, 1),
    }
