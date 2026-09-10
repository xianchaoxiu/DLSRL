import os
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import sys
import gym
import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper
import json
from scipy.spatial.transform import Rotation

from dppo.env.gym_utils.wrapper import wrapper_dict
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def make_robomimic_env(render=False, env='square', normalization_path=None, low_dim_keys=None, dppo_path=None, abs_action=False):
	wrappers = OmegaConf.create({
		'robomimic_lowdim': {
			'normalization_path': normalization_path,
			'low_dim_keys': low_dim_keys,
		},
	})
	obs_modality_dict = {
		"low_dim": (
			wrappers.robomimic_image.low_dim_keys
			if "robomimic_image" in wrappers
			else wrappers.robomimic_lowdim.low_dim_keys
		),
		"rgb": (
			wrappers.robomimic_image.image_keys
			if "robomimic_image" in wrappers
			else None
		),
	}
	if obs_modality_dict["rgb"] is None:
		obs_modality_dict.pop("rgb")
	ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
	robomimic_env_cfg_path = f'{dppo_path}/cfg/robomimic/env_meta/{env}.json'
	with open(robomimic_env_cfg_path, "r") as f:
		env_meta = json.load(f)
	env_meta["reward_shaping"] = False
	if abs_action:
		controller_configs = env_meta["env_kwargs"].get("controller_configs")
		if isinstance(controller_configs, list):
			for controller_config in controller_configs:
				controller_config["control_delta"] = False
		elif isinstance(controller_configs, dict):
			controller_configs["control_delta"] = False
	env = EnvUtils.create_env_from_metadata(
		env_meta=env_meta,
		render=False,
		render_offscreen=render,
		use_image_obs=False,
	)
	env.env.hard_reset = False
	for wrapper, args in wrappers.items():
		env = wrapper_dict[wrapper](env, **args)
	return env


def rotation_6d_to_axis_angle(rotation_6d):
	rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
	a1 = rotation_6d[..., :3]
	a2 = rotation_6d[..., 3:6]
	b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
	b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
	b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
	b3 = np.cross(b1, b2, axis=-1)
	mat = np.stack((b1, b2, b3), axis=-2)
	rotvec = Rotation.from_matrix(mat.reshape(-1, 3, 3)).as_rotvec()
	return rotvec.reshape(rotation_6d.shape[:-1] + (3,)).astype(np.float32)


class OfficialRobomimicAbsActionWrapper(gym.Env):
	def __init__(self, env):
		self.env = env
		self.action_space = spaces.Box(
			low=-np.ones(10, dtype=np.float32),
			high=np.ones(10, dtype=np.float32),
			dtype=np.float32,
		)
		self.observation_space = env.observation_space

	def seed(self, seed=None):
		return self.env.seed(seed=seed)

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

	def step(self, action):
		action = np.asarray(action, dtype=np.float32)
		if action.shape[-1] != 10:
			raise ValueError(f"Official transformer action must be 10-D, got shape {action.shape}")
		pos = action[..., :3]
		rot = rotation_6d_to_axis_angle(action[..., 3:9])
		gripper = action[..., 9:10]
		raw_action = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
		return self.env.step(raw_action)

	def render(self, **kwargs):
		return self.env.render(**kwargs)

	def close(self):
		if hasattr(self.env, "close"):
			return self.env.close()


class ObservationHistoryWrapper(gym.Env):
	def __init__(self, env, n_obs_steps, single_obs_dim=None):
		self.env = env
		self.n_obs_steps = int(n_obs_steps)
		self.single_obs_dim = single_obs_dim
		self.action_space = env.action_space
		low = getattr(env.observation_space, "low", None)
		high = getattr(env.observation_space, "high", None)
		if low is None or high is None:
			if single_obs_dim is None:
				raise ValueError("single_obs_dim is required when the wrapped env has no Box observation space")
			low = -np.inf * np.ones(single_obs_dim, dtype=np.float32)
			high = np.inf * np.ones(single_obs_dim, dtype=np.float32)
		else:
			low = np.asarray(low, dtype=np.float32).reshape(-1)
			high = np.asarray(high, dtype=np.float32).reshape(-1)
			if single_obs_dim is None:
				single_obs_dim = low.shape[0]
		self.single_obs_dim = int(single_obs_dim)
		self.observation_space = spaces.Box(
			low=np.tile(low, self.n_obs_steps),
			high=np.tile(high, self.n_obs_steps),
			dtype=np.float32,
		)
		self.history = None

	def _format_obs(self, obs):
		obs = np.asarray(obs, dtype=np.float32).reshape(-1)
		if obs.shape[0] != self.single_obs_dim:
			raise ValueError(f"Expected obs dim {self.single_obs_dim}, got {obs.shape[0]}")
		return obs

	def _flatten_history(self):
		return np.concatenate(self.history, axis=0).astype(np.float32)

	def seed(self, seed=None):
		return self.env.seed(seed=seed)

	def reset(self, **kwargs):
		obs = self._format_obs(self.env.reset(**kwargs))
		self.history = [obs.copy() for _ in range(self.n_obs_steps)]
		return self._flatten_history()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		obs = self._format_obs(obs)
		self.history = self.history[1:] + [obs]
		return self._flatten_history(), reward, done, info

	def render(self, **kwargs):
		return self.env.render(**kwargs)

	def close(self):
		if hasattr(self.env, "close"):
			return self.env.close()


class ObservationWrapperRobomimic(gym.Env):
	def __init__(
		self,
		env,
		reward_offset=1,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_offset = reward_offset

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = raw_obs['state'].flatten()
		return obs

	def step(self, action):
		raw_obs, reward, done, info = self.env.step(action)
		info = dict(info)
		success = None
		if hasattr(self.env, "is_success"):
			success = self.env.is_success()
		elif hasattr(self.env, "env") and hasattr(self.env.env, "is_success"):
			success = self.env.env.is_success()
		if isinstance(success, dict):
			info["is_success"] = bool(success.get("task", False))
		elif success is not None:
			info["is_success"] = bool(success)
		reward = (reward - self.reward_offset)
		obs = raw_obs['state'].flatten()
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	

class ObservationWrapperGym(gym.Env):
	def __init__(
		self,
		env,
		normalization_path,
	):
		self.env = env
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		normalization = np.load(normalization_path)
		self.obs_min = normalization["obs_min"]
		self.obs_max = normalization["obs_max"]
		self.action_min = normalization["action_min"]
		self.action_max = normalization["action_max"]

	def seed(self, seed=None):
		if seed is not None:
			np.random.seed(seed=seed)
		else:
			np.random.seed()

	def reset(self, **kwargs):
		options = kwargs.get("options", {})
		new_seed = options.get("seed", None)
		if new_seed is not None:
			self.seed(seed=new_seed)
		raw_obs = self.env.reset()
		obs = self.normalize_obs(raw_obs)
		return obs

	def step(self, action):
		raw_action = self.unnormalize_action(action)
		raw_obs, reward, done, info = self.env.step(raw_action)
		obs = self.normalize_obs(raw_obs)
		return obs, reward, done, info

	def render(self, **kwargs):
		return self.env.render()
	
	def normalize_obs(self, obs):
		return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

	def unnormalize_action(self, action):
		action = (action + 1) / 2
		return action * (self.action_max - self.action_min) + self.action_min
	

class ActionChunkWrapper(gymnasium.Env):
	def __init__(self, env, cfg, max_episode_steps=300):
		self.max_episode_steps = max_episode_steps
		self.env = env
		self.act_steps = cfg.act_steps
		self.action_space = spaces.Box(
			low=np.tile(env.action_space.low, cfg.act_steps),
			high=np.tile(env.action_space.high, cfg.act_steps),
			dtype=np.float32
		)
		self.observation_space = spaces.Box(
			low=-np.ones(cfg.obs_dim),
			high=np.ones(cfg.obs_dim),
			dtype=np.float32
		)
		self.count = 0

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)
		self.count = 0
		return obs, {}
	
	def step(self, action):
		if len(action.shape) == 1:
			action = action.reshape(self.act_steps, -1)
		obs_ = []
		reward_ = []
		done_ = []
		info_ = []
		done_i = False
		for i in range(action.shape[0]):
			self.count += 1
			obs_i, reward_i, done_i, info_i = self.env.step(action[i])
			obs_.append(obs_i)
			reward_.append(reward_i)
			done_.append(done_i)
			info_.append(info_i)
		obs = obs_[-1]
		reward = sum(reward_)
		done = np.max(done_)
		info = dict(info_[-1])
		if any("is_success" in info_i for info_i in info_):
			info["is_success"] = any(bool(info_i.get("is_success", False)) for info_i in info_)
		if self.count >= self.max_episode_steps:
			done = True
		if done:
			info['terminal_observation'] = obs
		return obs, reward, done, False, info

	def render(self):
		return self.env.render()
	
	def close(self):
		return
	

class DiffusionPolicyEnvWrapper(VecEnvWrapper):
	def __init__(self, env, cfg, base_policy):
		super().__init__(env)
		self.action_horizon = cfg.act_steps
		self.action_dim = cfg.action_dim
		self.adapter_enabled = bool(OmegaConf.select(cfg, "adapter.enabled", default=False))
		self.adapter_actor = None
		self.adapter_scale = float(OmegaConf.select(cfg, "adapter.injection_scale", default=0.0))
		if self.adapter_enabled:
			self.adapter_latent_steps = int(OmegaConf.select(cfg, "adapter.latent_steps", default=1))
			self.adapter_noise_steps = int(OmegaConf.select(cfg, "adapter.noise_steps", default=self.action_horizon))
			self.adapter_noise_magnitude = float(
				OmegaConf.select(cfg, "adapter.noise_magnitude", default=cfg.train.action_magnitude)
			)
			self.adapter_control_dim = int(OmegaConf.select(cfg, "adapter.control_dim", default=16))
			self.adapter_gate_dim = int(OmegaConf.select(cfg, "adapter.gate_dim", default=1))
			self.adapter_control_action_dim = self.adapter_control_dim + self.adapter_gate_dim
			noise_low = -self.adapter_noise_magnitude * np.ones(
				self.adapter_noise_steps * self.action_dim,
				dtype=np.float32,
			)
			noise_high = self.adapter_noise_magnitude * np.ones(
				self.adapter_noise_steps * self.action_dim,
				dtype=np.float32,
			)
			control_low = -np.ones(
				self.adapter_latent_steps * self.adapter_control_action_dim,
				dtype=np.float32,
			)
			control_high = np.ones(
				self.adapter_latent_steps * self.adapter_control_action_dim,
				dtype=np.float32,
			)
			self.action_space = spaces.Box(
				low=np.concatenate([noise_low, control_low]),
				high=np.concatenate([noise_high, control_high]),
				dtype=np.float32,
			)
		else:
			self.adapter_latent_steps = self.action_horizon
			self.adapter_noise_steps = self.action_horizon
			self.adapter_control_action_dim = self.action_dim
			self.action_space = spaces.Box(
				low=-cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
				high=cfg.train.action_magnitude*np.ones(self.action_dim*self.action_horizon),
				dtype=np.float32
			)
		self.obs_dim = cfg.obs_dim
		self.observation_space = spaces.Box(
			low=-np.ones(self.obs_dim),
			high=np.ones(self.obs_dim),
			dtype=np.float32
		)
		self.env = env
		self.device = cfg.model.device
		self.base_policy = base_policy
		self.obs = None

	def set_adapter_actor(self, actor):
		self.adapter_actor = actor

	def step_async(self, actions):
		actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
		if self.adapter_enabled:
			if self.adapter_actor is None:
				raise RuntimeError("Adapter conditioning is enabled, but no adapter actor was attached to the env wrapper.")
			actor_device = next(self.adapter_actor.parameters()).device
			latent_actions = actions.to(actor_device).view(actions.shape[0], -1)
			with torch.no_grad():
				noise, adapter_feature, adapter_gate = self.adapter_actor.pack_adapter_action(latent_actions)
				if noise.shape[1] == 1:
					noise = noise.repeat(1, self.action_horizon, 1)
				elif noise.shape[1] != self.action_horizon:
					raise ValueError(
						f"Adapter latent steps must be 1 or {self.action_horizon}, got {noise.shape[1]}"
					)
				if adapter_feature.shape[1] == 1:
					adapter_feature = adapter_feature.repeat(1, self.action_horizon, 1)
					adapter_gate = adapter_gate.repeat(1, self.action_horizon, 1)
				elif adapter_feature.shape[1] != self.action_horizon:
					raise ValueError(
						f"Adapter control steps must be 1 or {self.action_horizon}, got {adapter_feature.shape[1]}"
					)
				diffused_actions = self.base_policy(
					self.obs,
					noise.to(self.device),
					adapter_feature=adapter_feature.to(self.device),
					adapter_gate=adapter_gate.to(self.device),
					adapter_scale=self.adapter_scale,
				)
		else:
			actions = actions.view(-1, self.action_horizon, self.action_dim)
			diffused_actions = self.base_policy(self.obs, actions)
		self.venv.step_async(diffused_actions)

	def step_wait(self):
		obs, rewards, dones, infos = self.venv.step_wait()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy(), rewards, dones, infos

	def reset(self):
		obs = self.venv.reset()
		self.obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
		obs_out = self.obs
		return obs_out.detach().cpu().numpy()
	
