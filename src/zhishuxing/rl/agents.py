"""多智能体 Off-Policy 算法：MADDPG 与 MATD3（原两个文件合并，TD3 三技巧作为独立路径）。

数值行为与历史实现一致：相同的更新顺序、损失构造与软更新；
仅修复 MATD3.save_model 缺失目录创建、梯度裁剪阈值可配置两点。
"""

from __future__ import annotations

import copy
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, Critic_MADDPG, Critic_MATD3

DEFAULT_GRAD_CLIP = 10.0


def _grad_clip_value(args) -> float:
    return float(getattr(args, "grad_clip", DEFAULT_GRAD_CLIP) or DEFAULT_GRAD_CLIP)


class _OffPolicyAgentBase:
    """公共骨架：网络构建、动作选择、软更新、模型保存。"""

    td3_tricks = False

    def __init__(self, args, agent_id):
        self.N = args.N
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.grad_clip = _grad_clip_value(args)

        if self.td3_tricks:
            self.policy_noise = args.policy_noise
            self.noise_clip = args.noise_clip
            self.policy_update_freq = args.policy_update_freq
            self.actor_pointer = 0

        # Create an individual actor and critic for each agent according to the 'agent_id'
        self.actor = Actor(args, agent_id)
        self.critic = Critic_MATD3(args) if self.td3_tricks else Critic_MADDPG(args)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    # Each agent selects actions based on its own local observations(add noise for exploration)
    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def _soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _backward(self, loss: torch.Tensor, module: torch.nn.Module, optimizer) -> None:
        optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)
        optimizer.step()

    def save_model(self, model_dir, env_name, algorithm, number, total_steps, agent_id):
        env_dir = os.path.join(str(model_dir), env_name)
        os.makedirs(env_dir, exist_ok=True)
        path = os.path.join(env_dir, "{}_actor_number_{}_step_{}k_agent_{}.pth".format(algorithm, number, int(total_steps / 1000), agent_id))
        torch.save(self.actor.state_dict(), path)
        return path


class MADDPG(_OffPolicyAgentBase):
    """标准 MADDPG：单 Q 集中式 Critic，critic/actor 每次训练都更新。"""

    td3_tricks = False

    def train(self, replay_buffer, agent_n: List["MADDPG"]):
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Select next actions according to the actor_target
            batch_a_next_n = [agent.actor_target(batch_obs_next) for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)]
            Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next  # shape:(batch_size,1)

        current_Q = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(target_Q, current_Q)
        self._backward(critic_loss, self.critic, self.critic_optimizer)

        # Reselect the actions of the agent corresponding to 'agent_id'，the actions of other agents remain unchanged
        batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
        actor_loss = -self.critic(batch_obs_n, batch_a_n).mean()
        self._backward(actor_loss, self.actor, self.actor_optimizer)

        # Softly update the target networks
        self._soft_update()


class MATD3(_OffPolicyAgentBase):
    """MATD3：target policy smoothing + clipped double Q + delayed policy updates。"""

    td3_tricks = True

    def train(self, replay_buffer, agent_n: List["MATD3"]):
        self.actor_pointer += 1
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1: target policy smoothing
            batch_a_next_n = []
            for i in range(self.N):
                batch_a_next = agent_n[i].actor_target(batch_obs_next_n[i])
                noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)
                batch_a_next_n.append(batch_a_next)

            # Trick 2: clipped double Q-learning
            Q1_next, Q2_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * torch.min(Q1_next, Q2_next)  # shape:(batch_size,1)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self._backward(critic_loss, self.critic, self.critic_optimizer)

        # Trick 3: delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -self.critic.Q1(batch_obs_n, batch_a_n).mean()  # Only use Q1
            self._backward(actor_loss, self.actor, self.actor_optimizer)

            # Softly update the target networks
            self._soft_update()
