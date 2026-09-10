import os
import torch
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import hydra
import time
from omegaconf import OmegaConf


class DPPOBasePolicyWrapper:
	def __init__(self, base_policy):
		self.base_policy = base_policy
		
	def __call__(
		self,
		obs,
		initial_noise,
		return_numpy=True,
		adapter_feature=None,
		adapter_gate=None,
		adapter_scale=0.0,
	):
		cond = {
			"state": obs,
			"noise_action": initial_noise,
		}
		if adapter_feature is not None:
			cond["adapter_feature"] = adapter_feature
			cond["adapter_scale"] = adapter_scale
		if adapter_gate is not None:
			cond["adapter_gate"] = adapter_gate
		with torch.no_grad():
			samples = self.base_policy(cond=cond, deterministic=True)
		diffused_actions = (samples.trajectories.detach())
		if return_numpy:
			diffused_actions = diffused_actions.cpu().numpy()
		return diffused_actions	


def load_base_policy(cfg):
	base_policy = hydra.utils.instantiate(cfg.model)
	base_policy = base_policy.eval()
	return DPPOBasePolicyWrapper(base_policy)


class LoggingCallback(BaseCallback):
	def __init__(self, 
		action_chunk=4, 
		log_freq=1000,
		use_wandb=True, 
		eval_env=None, 
		eval_env_fn=None,
		eval_freq=70, 
		eval_episodes=2, 
		verbose=0, 
		rew_offset=0, 
		num_train_env=1,
		num_eval_env=1,
		algorithm='dsrl_sac',
		max_steps=-1,
		deterministic_eval=False,
		best_model_save_path=None,
	):
		super().__init__(verbose)
		self.action_chunk = action_chunk
		self.log_freq = log_freq
		self.episode_rewards = []
		self.episode_lengths = []
		self.use_wandb = use_wandb
		self.eval_env = eval_env
		self.eval_env_fn = eval_env_fn
		self.eval_episodes = eval_episodes
		self.eval_freq = eval_freq
		self.log_count = 0
		self.total_reward = 0
		self.rew_offset = rew_offset
		self.total_timesteps = 0
		self.num_train_env = num_train_env
		self.num_eval_env = num_eval_env
		self.episode_success = np.zeros(self.num_train_env)
		self.episode_completed = np.zeros(self.num_train_env)
		self.algorithm = algorithm
		self.max_steps = max_steps
		self.deterministic_eval = deterministic_eval
		self.best_model_save_path = best_model_save_path
		self.best_eval_success = -np.inf
		self.best_eval_reward = -np.inf

	def _on_step(self):
		for info in self.locals['infos']:
			if 'episode' in info:
				self.episode_rewards.append(info['episode']['r'])
				self.episode_lengths.append(info['episode']['l'])
		rew = self.locals['rewards']
		self.total_reward += np.mean(rew)
		for env_idx, info in enumerate(self.locals['infos']):
			if 'is_success' in info:
				if info['is_success']:
					self.episode_success[env_idx] = 1
			elif rew[env_idx] > -self.rew_offset:
				self.episode_success[env_idx] = 1
		self.episode_completed[self.locals['dones']] = 1
		self.total_timesteps += self.action_chunk * self.model.n_envs
		if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
			if len(self.episode_rewards) > 0:
				if self.use_wandb:
					self.log_count += 1
					logger_values = self.locals['self'].logger.name_to_value
					log_data = {
						"train/ep_len_mean": np.mean(self.episode_lengths),
						"train/ep_rew_mean": np.mean(self.episode_rewards),
						"train/rew_mean": self.total_reward,
						"train/timesteps": self.total_timesteps,
					}
					completed = np.sum(self.episode_completed)
					if completed > 0:
						log_data["train/success_rate"] = np.sum(self.episode_success) / completed
					for key in [
						"train/ent_coef",
						"train/actor_loss",
						"train/critic_loss",
						"train/critic_loss_per_q",
						"train/action_l2",
						"train/qf_pi",
						"train/ent_coef_loss",
						"train/noise_critic_loss",
						"train/adapter_l2",
						"train/adapter_gate_l1",
						"train/adapter_feature_norm",
						"train/adapter_gate_mean",
						"train/adapter_effective_delta_norm",
						"train/noise_l2",
						"train/control_code_l2",
						"train/gate_code_l2",
						"train/noise_saturation_rate",
						"train/control_code_saturation_rate",
						"train/gate_code_mean",
						"train/gate_code_saturation_rate",
						"train/adapter_runtime_scale",
						"train/noise_log_prob",
						"train/adapter_log_prob",
					]:
						if key in logger_values:
							log_data[key] = logger_values[key]
					wandb.log(log_data, step=self.log_count)
				self.episode_rewards = []
				self.episode_lengths = []
				self.total_reward = 0
				self.episode_success = np.zeros(self.num_train_env)
				self.episode_completed = np.zeros(self.num_train_env)

		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			self.evaluate(self.locals['self'], deterministic=False)
			if self.deterministic_eval:
				self.evaluate(self.locals['self'], deterministic=True)
		return True

	def _get_eval_env(self):
		if self.eval_env is None and self.eval_env_fn is not None:
			self.eval_env = self.eval_env_fn()
		return self.eval_env
		
	def evaluate(self, agent, deterministic=False):
		if self.eval_episodes <= 0:
			return
		env = self._get_eval_env()
		if env is None:
			return
		with torch.no_grad():
			eval_start = time.time()
			print(
				f"[eval] start: episodes={self.eval_episodes}, "
				f"n_envs={self.num_eval_env}, max_steps={self.max_steps}, "
				f"algorithm={self.algorithm}, deterministic={deterministic}"
			)
			success = []
			adapter_diagnostics = []
			rew_total, total_ep = 0, 0
			rew_ep = np.zeros(self.num_eval_env)
			for i in range(self.eval_episodes):
				episode_start = time.time()
				obs = env.reset()
				if hasattr(agent, "adapter_action_diagnostics"):
					diagnostics = agent.adapter_action_diagnostics(obs, deterministic=deterministic)
					if diagnostics is not None:
						adapter_diagnostics.append(diagnostics)
				success_i = np.zeros(obs.shape[0])
				r = []
				for step_idx in range(self.max_steps):
					if self.algorithm == 'dsrl_sac':
						action, _ = agent.predict(obs, deterministic=deterministic)
					elif self.algorithm == 'dsrl_na':
						action, _ = agent.predict_diffused(obs, deterministic=deterministic)
					next_obs, reward, done, info = env.step(action)
					obs = next_obs
					rew_ep += reward
					rew_total += sum(rew_ep[done])
					rew_ep[done] = 0 
					total_ep += np.sum(done)
					for env_idx, info_i in enumerate(info):
						if 'is_success' in info_i:
							if info_i['is_success']:
								success_i[env_idx] = 1
						elif reward[env_idx] > -self.rew_offset:
							success_i[env_idx] = 1
					r.append(reward)
					if (step_idx + 1) % 10 == 0 or (step_idx + 1) == self.max_steps:
						print(
							f"[eval] episode {i}: step {step_idx + 1}/{self.max_steps}, "
							f"elapsed={time.time() - episode_start:.1f}s"
						)
				success.append(success_i.mean())
				print(
					f"eval episode {i} at timestep {self.total_timesteps}; "
					f"episode_time={time.time() - episode_start:.1f}s"
				)
			success_rate = np.mean(success)
			if total_ep > 0:
				avg_rew = rew_total / total_ep
			else:
				avg_rew = 0
			diagnostic_log = {}
			if adapter_diagnostics:
				for key in adapter_diagnostics[0]:
					diagnostic_log[f"eval/{key}"] = float(np.mean([item[key] for item in adapter_diagnostics]))
			if self.use_wandb:
				name = 'eval'
				if deterministic:
					wandb.log({
						f"{name}/success_rate_deterministic": success_rate,
						f"{name}/reward_deterministic": avg_rew,
						**diagnostic_log,
					}, step=self.log_count)
				else:
					wandb.log({
						f"{name}/success_rate": success_rate,
						f"{name}/reward": avg_rew,
						f"{name}/timesteps": self.total_timesteps,
						**diagnostic_log,
					}, step=self.log_count)
			if not deterministic and self.best_model_save_path is not None:
				is_better = (
					success_rate > self.best_eval_success
					or (
						np.isclose(success_rate, self.best_eval_success)
						and avg_rew > self.best_eval_reward
					)
				)
				if is_better:
					self.best_eval_success = success_rate
					self.best_eval_reward = avg_rew
					os.makedirs(os.path.dirname(self.best_model_save_path), exist_ok=True)
					agent.save(self.best_model_save_path)
					print(
						f"[eval] saved best model: success_rate={success_rate:.3f}, "
						f"reward={avg_rew:.3f}, path={self.best_model_save_path}.zip"
					)
			print(f"[eval] done: total_time={time.time() - eval_start:.1f}s, success_rate={success_rate:.3f}, reward={avg_rew:.3f}")

	def close_eval_env(self):
		if self.eval_env is not None:
			self.eval_env.close()
			self.eval_env = None

	def set_timesteps(self, timesteps):
		self.total_timesteps = timesteps



def collect_rollouts(model, env, num_steps, base_policy, cfg):
	obs = env.reset()
	rollout_start = time.time()
	progress_interval = max(1, min(10, int(num_steps / 20) if num_steps > 0 else 1))
	print(
		f"[rollout] start: chunks={num_steps}, n_envs={cfg.env.n_envs}, "
		f"act_steps={cfg.act_steps}, action_dim={cfg.action_dim}"
	)
	for i in range(num_steps):
		iter_start = time.time()
		if i == 0:
			print("[rollout] running first diffusion-policy chunk; this includes CUDA/model warmup")
		policy_start = time.time()
		if cfg.algorithm == 'dsrl_sac':
			if bool(OmegaConf.select(cfg, "adapter.enabled", default=False)):
				latent_steps = int(OmegaConf.select(cfg, "adapter.latent_steps", default=1))
				noise_steps = int(OmegaConf.select(cfg, "adapter.noise_steps", default=cfg.act_steps))
				noise_magnitude = float(OmegaConf.select(cfg, "adapter.noise_magnitude", default=cfg.train.action_magnitude))
				control_dim = int(OmegaConf.select(cfg, "adapter.control_dim", default=16))
				gate_dim = int(OmegaConf.select(cfg, "adapter.gate_dim", default=1))
				noise = torch.randn(cfg.env.n_envs, noise_steps, cfg.action_dim, device=cfg.device)
				noise = noise.clamp(-noise_magnitude, noise_magnitude)
				control = torch.zeros(cfg.env.n_envs, latent_steps, control_dim, device=cfg.device)
				gate_code = -torch.ones(cfg.env.n_envs, latent_steps, gate_dim, device=cfg.device)
				latent_action = torch.cat([
					noise.reshape(cfg.env.n_envs, -1),
					control.reshape(cfg.env.n_envs, -1),
					gate_code.reshape(cfg.env.n_envs, -1),
				], dim=-1)
				action = latent_action.detach().cpu().numpy()
			else:
				noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
				noise = noise.clamp(-cfg.train.action_magnitude, cfg.train.action_magnitude)
				action = noise.reshape(cfg.env.n_envs, cfg.act_steps * cfg.action_dim).detach().cpu().numpy()
		else:
			noise = torch.randn(cfg.env.n_envs, cfg.act_steps, cfg.action_dim).to(device=cfg.device)
			# Match the original DSRL-NA initial replay distribution. Adapter
			# noise bounds apply to the learned actor, not this baseline rollout.
			noise = noise.clamp(-cfg.train.action_magnitude, cfg.train.action_magnitude)
			action = base_policy(torch.tensor(obs, device=cfg.device, dtype=torch.float32), noise)
		policy_time = time.time() - policy_start
		env_start = time.time()
		next_obs, reward, done, info = env.step(action)
		env_time = time.time() - env_start
		if cfg.algorithm == 'dsrl_na':
			action_store = action
		elif cfg.algorithm == 'dsrl_sac':
			action_store = model.policy.scale_action(action)
			if getattr(model.actor, "use_adapter_conditioning", False):
				with torch.no_grad():
					action_tensor = torch.as_tensor(action_store, device=model.device, dtype=torch.float32)
					action_store = model.actor.critic_adapter_action(action_tensor).cpu().numpy()
		if cfg.algorithm == 'dsrl_na':
			action_store = action_store.reshape(-1, action_store.shape[1] * action_store.shape[2])
		model.replay_buffer.add(
				obs=obs,
				next_obs=next_obs,
				action=action_store,
				reward=reward,
				done=done,
				infos=info,
		)
		obs = next_obs
		if i < 5 or (i + 1) % progress_interval == 0 or (i + 1) == num_steps:
			print(
				f"[rollout] {i + 1}/{num_steps} chunks, "
				f"env_steps={(i + 1) * cfg.env.n_envs * cfg.act_steps}, "
				f"policy={policy_time:.2f}s, env={env_time:.2f}s, "
				f"iter={time.time() - iter_start:.2f}s, elapsed={time.time() - rollout_start:.1f}s"
			)
	model.replay_buffer.final_offline_step()
	


def load_offline_data(model, offline_data_path, n_env):
	# this function should only be applied with dsrl_na
	offline_data = np.load(offline_data_path)
	obs = offline_data['states']
	next_obs = offline_data['states_next']
	actions = offline_data['actions']
	rewards = offline_data['rewards']
	terminals = offline_data['terminals']
	for i in range(int(obs.shape[0]/n_env)):
		model.replay_buffer.add(
					obs=obs[n_env*i:n_env*i+n_env],
					next_obs=next_obs[n_env*i:n_env*i+n_env],
					action=actions[n_env*i:n_env*i+n_env],
					reward=rewards[n_env*i:n_env*i+n_env],
					done=terminals[n_env*i:n_env*i+n_env],
					infos=[{}] * n_env,
				)
	model.replay_buffer.final_offline_step()
