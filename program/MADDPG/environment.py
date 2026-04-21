import numpy as np
from dataclasses import dataclass
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment


@dataclass
class Space:
	shape: tuple


class Env:
	def __init__(
		self,
		env_name,
		discrete=False,
		behavior_name=None,
		file_name=None,
		base_port=5005,
		seed=1,
		no_graphics=True,
		timeout_wait=60,
		worker_id=0,
	):
		self.env_name = env_name
		self.discrete = discrete
		self.behavior_name = behavior_name
		self.file_name = file_name
		self.base_port = base_port
		self.seed = seed
		self.no_graphics = no_graphics
		self.timeout_wait = timeout_wait
		self.worker_id = worker_id

		self.unity_env = UnityEnvironment(
			file_name=self.file_name,
			seed=self.seed,
			base_port=self.base_port,
			no_graphics=self.no_graphics,
			timeout_wait=self.timeout_wait,
			worker_id=self.worker_id,
		)
		self.unity_env.reset()

		if self.behavior_name is None:
			behavior_names = list(self.unity_env.behavior_specs.keys())
			if len(behavior_names) == 0:
				raise RuntimeError("No behavior spec found in Unity environment.")
			self.behavior_name = behavior_names[0]

		self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
		if self.behavior_spec.action_spec.discrete_size > 0:
			raise ValueError("MADDPG当前实现仅支持连续动作，请在Unity中使用连续动作空间。")

		decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
		initial_agent_ids = list(decision_steps.agent_id)
		if len(initial_agent_ids) == 0:
			initial_agent_ids = list(terminal_steps.agent_id)
		if len(initial_agent_ids) == 0:
			raise RuntimeError("No active agents found after environment reset.")

		self.agent_ids = sorted(int(agent_id) for agent_id in initial_agent_ids)
		self.n = len(self.agent_ids)
		self.id_to_index = {agent_id: i for i, agent_id in enumerate(self.agent_ids)}

		obs_dims = self._get_obs_dims(self.behavior_spec)
		action_dim = int(self.behavior_spec.action_spec.continuous_size)
		self.observation_space = [Space(shape=(obs_dims,)) for _ in range(self.n)]
		self.action_space = [Space(shape=(action_dim,)) for _ in range(self.n)]

		self.last_obs_n = self._build_obs(decision_steps, terminal_steps)
		self.pending_action_agent_ids = [int(agent_id) for agent_id in decision_steps.agent_id]

	def _get_obs_dims(self, behavior_spec):
		total_obs_dim = 0
		for obs_spec in behavior_spec.observation_specs:
			obs_dim = int(np.prod(obs_spec.shape))
			total_obs_dim += obs_dim
		return total_obs_dim

	def _flatten_obs(self, steps, agent_id):
		obs_parts = []
		for obs in steps.obs:
			obs_parts.append(np.asarray(obs[agent_id]).reshape(-1))
		return np.concatenate(obs_parts, axis=0).astype(np.float32)

	def _build_obs(self, decision_steps, terminal_steps):
		obs_n = [None] * self.n
		for agent_id in self.agent_ids:
			if agent_id in decision_steps:
				obs_n[self.id_to_index[agent_id]] = self._flatten_obs(decision_steps, agent_id)
			elif agent_id in terminal_steps:
				obs_n[self.id_to_index[agent_id]] = self._flatten_obs(terminal_steps, agent_id)
			elif self.last_obs_n is not None:
				obs_n[self.id_to_index[agent_id]] = self.last_obs_n[self.id_to_index[agent_id]]
			else:
				obs_n[self.id_to_index[agent_id]] = np.zeros(self.observation_space[0].shape[0], dtype=np.float32)
		return obs_n

	def reset(self):
		self.unity_env.reset()
		decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
		self.pending_action_agent_ids = [int(agent_id) for agent_id in decision_steps.agent_id]
		self.last_obs_n = self._build_obs(decision_steps, terminal_steps)
		return self.last_obs_n

	def step(self, action_n):
		if len(self.pending_action_agent_ids) > 0:
			action_array = []
			for agent_id in self.pending_action_agent_ids:
				index = self.id_to_index[int(agent_id)]
				action_array.append(np.asarray(action_n[index], dtype=np.float32))
			action_array = np.asarray(action_array, dtype=np.float32)
			self.unity_env.set_actions(
				self.behavior_name,
				ActionTuple(continuous=action_array),
			)

		self.unity_env.step()
		decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

		obs_next_n = self._build_obs(decision_steps, terminal_steps)
		reward_n = [0.0] * self.n
		done_n = [False] * self.n

		for agent_id in self.agent_ids:
			index = self.id_to_index[agent_id]
			if agent_id in terminal_steps:
				reward_n[index] = float(terminal_steps[agent_id].reward)
				done_n[index] = True
			elif agent_id in decision_steps:
				reward_n[index] = float(decision_steps[agent_id].reward)

		self.pending_action_agent_ids = [int(agent_id) for agent_id in decision_steps.agent_id]
		self.last_obs_n = obs_next_n
		return obs_next_n, reward_n, done_n, {}

	def close(self):
		self.unity_env.close()
