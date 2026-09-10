import os
import time

def _prepend_env_path(name, paths):
	current = os.environ.get(name, "")
	parts = [p for p in current.split(os.pathsep) if p]
	for path in reversed(paths):
		if os.path.exists(path) and path not in parts:
			parts.insert(0, path)
	os.environ[name] = os.pathsep.join(parts)

mujoco_path = os.path.expanduser("~/.mujoco/mujoco210")
if os.path.exists(mujoco_path):
	os.environ.setdefault("MUJOCO_PATH", mujoco_path)
	os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", mujoco_path)
	_prepend_env_path("LD_LIBRARY_PATH", [
		os.path.join(mujoco_path, "bin"),
		"/usr/lib/nvidia",
		"/usr/lib/x86_64-linux-gnu",
		os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib"),
	])
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np
import hydra
from omegaconf import OmegaConf
import gym
try:
	import d4rl
	import d4rl.gym_mujoco
except ModuleNotFoundError:
	d4rl = None
import sys
sys.path.append('./dppo')
 
from stable_baselines3 import SAC, DSRL
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env, OfficialRobomimicAbsActionWrapper, ObservationHistoryWrapper
from utils import load_base_policy, load_offline_data, collect_rollouts, LoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))

	


@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"), config_name="dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
	OmegaConf.resolve(cfg)
	run_initial_eval = OmegaConf.select(cfg, "run_initial_eval", default=True)
	total_timesteps = OmegaConf.select(cfg, "total_timesteps", default=20000000)
	start_time = time.time()
	print(
		f"[setup] device={cfg.device}, algorithm={cfg.algorithm}, "
		f"n_envs={cfg.env.n_envs}, act_steps={cfg.act_steps}"
	)

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	if cfg.use_wandb:
		wandb.init(
			project=cfg.wandb.project,
			name=cfg.name,
			group=cfg.wandb.group,
			monitor_gym=True,
			save_code=True,
			config=OmegaConf.to_container(cfg, resolve=True),
		)

	MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)
	official_abs_action = bool(OmegaConf.select(cfg, "official_diffusion_policy.abs_action", default=False))
	obs_history_steps = int(OmegaConf.select(cfg, "official_diffusion_policy.obs_history_steps", default=1))
	single_obs_dim = int(OmegaConf.select(cfg, "official_diffusion_policy.single_obs_dim", default=cfg.obs_dim))

	num_env = cfg.env.n_envs
	def make_env():
		if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
			if d4rl is None:
				raise ModuleNotFoundError(
					"d4rl is required for D4RL locomotion tasks. "
					"Install d4rl before using halfcheetah/hopper/walker2d configs."
				)
			env = gym.make(cfg.env_name)
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
			normalization_path = None if official_abs_action else cfg.normalization_path
			env = make_robomimic_env(env=cfg.env_name, normalization_path=normalization_path, low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys, dppo_path=cfg.dppo_path, abs_action=official_abs_action)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
			if official_abs_action:
				env = OfficialRobomimicAbsActionWrapper(env)
			if obs_history_steps > 1:
				env = ObservationHistoryWrapper(env, obs_history_steps, single_obs_dim=single_obs_dim)
		env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
		return env

	phase_time = time.time()
	base_policy = load_base_policy(cfg)
	print(f"[timing] load_base_policy: {time.time() - phase_time:.1f}s")
	phase_time = time.time()
	env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
	if cfg.algorithm == 'dsrl_sac':
		env = DiffusionPolicyEnvWrapper(env, cfg, base_policy)
	env.seed(cfg.seed + 1)
	print(f"[timing] train env setup: {time.time() - phase_time:.1f}s")
	post_linear_modules = None
	if cfg.train.use_layer_norm:
		post_linear_modules = [torch.nn.LayerNorm]

	net_arch = []
	for _ in range(cfg.train.num_layers):
		net_arch.append(cfg.train.layer_size)
	use_adapter = bool(OmegaConf.select(cfg, "adapter.enabled", default=False))
	log_std_init = (
		float(OmegaConf.select(cfg, "adapter.log_std_init", default=0.0))
		if use_adapter
		else 0.0
	)
	policy_kwargs = dict(
		net_arch=dict(pi=net_arch, qf=net_arch),
		activation_fn=torch.nn.Tanh,
		log_std_init=log_std_init,
		post_linear_modules=post_linear_modules,
		n_critics=cfg.train.n_critics,
	)
	adapter_kwargs = None
	if use_adapter:
		adapter_feature_dim = int(
			OmegaConf.select(
				cfg,
				"adapter.feature_dim",
				default=getattr(base_policy.base_policy, "hidden_dim", 256),
			)
		)
		adapter_kwargs = dict(
			enabled=True,
			latent_steps=int(OmegaConf.select(cfg, "adapter.latent_steps", default=1)),
			noise_steps=int(OmegaConf.select(cfg, "adapter.noise_steps", default=cfg.act_steps)),
			noise_dim=int(cfg.action_dim),
			noise_magnitude=float(OmegaConf.select(cfg, "adapter.noise_magnitude", default=cfg.train.action_magnitude)),
			control_dim=int(OmegaConf.select(cfg, "adapter.control_dim", default=16)),
			gate_dim=int(OmegaConf.select(cfg, "adapter.gate_dim", default=1)),
			feature_dim=adapter_feature_dim,
			hidden_dim=int(OmegaConf.select(cfg, "adapter.hidden_dim", default=128)),
			feature_init_std=float(OmegaConf.select(cfg, "adapter.feature_init_std", default=1e-3)),
			feature_scale=float(OmegaConf.select(cfg, "adapter.feature_scale", default=0.25)),
			gate_max=float(OmegaConf.select(cfg, "adapter.gate_max", default=0.2)),
			gate_logit_scale=float(OmegaConf.select(cfg, "adapter.gate_logit_scale", default=5.0)),
			gate_logit_bias=float(OmegaConf.select(cfg, "adapter.gate_logit_bias", default=-2.5)),
			injection_scale=float(OmegaConf.select(cfg, "adapter.injection_scale", default=1.0)),
			critic_effect_scale=float(OmegaConf.select(cfg, "adapter.critic_effect_scale", default=1.0)),
			control_log_std_init=float(OmegaConf.select(cfg, "adapter.control_log_std_init", default=-3.0)),
			entropy_noise_only=bool(OmegaConf.select(cfg, "adapter.entropy_noise_only", default=True)),
			warmup_steps=int(OmegaConf.select(cfg, "adapter.warmup_steps", default=0)),
			ramp_steps=int(OmegaConf.select(cfg, "adapter.ramp_steps", default=0)),
			gradient_scale=float(OmegaConf.select(cfg, "adapter.gradient_scale", default=1.0)),
		)
		if cfg.algorithm == 'dsrl_sac':
			policy_kwargs["adapter_kwargs"] = adapter_kwargs
	phase_time = time.time()
	if cfg.algorithm == 'dsrl_sac':
		model = SAC(
			"MlpPolicy",
			env,
			device=cfg.device,
			learning_rate=cfg.train.actor_lr,
			actor_lr=float(OmegaConf.select(cfg, "train.actor_lr", default=cfg.train.actor_lr)),
			critic_lr=float(OmegaConf.select(cfg, "train.critic_lr", default=cfg.train.actor_lr)),
			ent_coef_lr=float(OmegaConf.select(cfg, "train.temp_lr", default=cfg.train.actor_lr)),
			buffer_size=20000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			action_norm_regularization=float(OmegaConf.select(cfg, "train.action_norm_regularization", default=0.0)),
			adapter_l2_coef=float(OmegaConf.select(cfg, "adapter.l2_coef", default=0.0)),
			adapter_gate_l1_coef=float(OmegaConf.select(cfg, "adapter.gate_l1_coef", default=0.0)),
			critic_reduction=str(OmegaConf.select(cfg, "train.critic_reduction", default="min")),
		)
	elif cfg.algorithm == 'dsrl_na':
		model = DSRL(
			"MlpPolicy",
			env,
			device=cfg.device,
			learning_rate=cfg.train.actor_lr,
			buffer_size=10000000,      # Replay buffer size
			learning_starts=1,    # How many steps before learning starts (total steps for all env combined)
			batch_size=cfg.train.batch_size,
			tau=cfg.train.tau,                # Target network update rate
			gamma=cfg.train.discount,               # Discount factor
			train_freq=cfg.train.train_freq,             # Update the model every train_freq steps
			gradient_steps=cfg.train.utd,         # How many gradient steps to do at each update
			action_noise=None,        # No additional action noise
			optimize_memory_usage=False,
			ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,          # Automatic entropy tuning
			target_update_interval=1, # Update target network every interval
			target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,    # Automatic target entropy
			use_sde=False,
			sde_sample_freq=-1,
			tensorboard_log=cfg.logdir,
			verbose=1,
			policy_kwargs=policy_kwargs,
			diffusion_policy=base_policy,
			diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
			noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
			critic_backup_combine_type=cfg.train.critic_backup_combine_type,
			actor_gradient_steps=int(OmegaConf.select(cfg, "train.actor_gradient_steps", default=-1)),
			adapter_kwargs=adapter_kwargs,
			action_norm_regularization=float(OmegaConf.select(cfg, "train.action_norm_regularization", default=0.0)),
			adapter_l2_coef=float(OmegaConf.select(cfg, "adapter.l2_coef", default=0.0)),
			adapter_gate_l1_coef=float(OmegaConf.select(cfg, "adapter.gate_l1_coef", default=0.0)),
			adapter_control_l2_coef=float(OmegaConf.select(cfg, "adapter.control_l2_coef", default=0.0)),
			adapter_gate_code_l2_coef=float(OmegaConf.select(cfg, "adapter.gate_code_l2_coef", default=0.0)),
		)
	if cfg.algorithm == 'dsrl_sac' and hasattr(env, "set_adapter_actor"):
		env.set_adapter_actor(model.actor)
	if use_adapter:
		latent_dim = (
			adapter_kwargs["noise_steps"] * adapter_kwargs["noise_dim"]
			+ adapter_kwargs["latent_steps"]
			* (adapter_kwargs["control_dim"] + adapter_kwargs["gate_dim"])
		)
		critic_latent_dim = (
			adapter_kwargs["noise_steps"] * adapter_kwargs["noise_dim"]
			+ adapter_kwargs["latent_steps"]
			* (adapter_kwargs["control_dim"] + adapter_kwargs["gate_dim"])
		)
		print(
			f"[setup] adapter backend={cfg.algorithm}, latent_dim={latent_dim} "
			f"(noise={adapter_kwargs['noise_steps']}x{adapter_kwargs['noise_dim']}, "
			f"control={adapter_kwargs['latent_steps']}x{adapter_kwargs['control_dim']}, "
			f"gate={adapter_kwargs['latent_steps']}x{adapter_kwargs['gate_dim']}), "
			f"critic_latent_dim={critic_latent_dim}, feature_dim={adapter_kwargs['feature_dim']}, "
			f"warmup={adapter_kwargs['warmup_steps']}, ramp={adapter_kwargs['ramp_steps']}, "
			f"critic_effect_scale={adapter_kwargs['critic_effect_scale']}, "
			f"adapter_gradient_scale={adapter_kwargs['gradient_scale']}"
		)
	print(f"[timing] RL model setup: {time.time() - phase_time:.1f}s")

	checkpoint_callback = CheckpointCallback(
		save_freq=cfg.save_model_interval, 
		save_path=cfg.logdir+'/checkpoint/',
		name_prefix='ft_policy',
		save_replay_buffer=cfg.save_replay_buffer, 
		save_vecnormalize=True,
	)

	num_env_eval = int(cfg.env.n_eval_envs)
	eval_episodes = 0
	eval_env_fn = None
	if cfg.num_evals > 0:
		num_env_eval = max(1, min(num_env_eval, int(cfg.num_evals)))
		eval_episodes = int(math.ceil(cfg.num_evals / num_env_eval))

		def make_eval_env():
			print(
				f"[setup] creating eval env lazily: "
				f"n_eval_envs={num_env_eval}, num_evals={cfg.num_evals}"
			)
			phase_time = time.time()
			eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
			if cfg.algorithm == 'dsrl_sac':
				eval_env = DiffusionPolicyEnvWrapper(eval_env, cfg, base_policy)
				if hasattr(eval_env, "set_adapter_actor"):
					eval_env.set_adapter_actor(model.actor)
			eval_env.seed(cfg.seed + num_env + 1)
			print(f"[timing] eval env setup: {time.time() - phase_time:.1f}s")
			return eval_env

		eval_env_fn = make_eval_env

	logging_callback = LoggingCallback(
		action_chunk = cfg.act_steps, 
		eval_episodes = eval_episodes, 
		log_freq=MAX_STEPS, 
		use_wandb=cfg.use_wandb, 
		eval_env=None, 
		eval_env_fn=eval_env_fn,
		eval_freq=cfg.eval_interval,
		num_train_env=num_env,
		num_eval_env=num_env_eval,
		rew_offset=cfg.env.reward_offset,
		algorithm=cfg.algorithm,
		max_steps=MAX_STEPS,
		deterministic_eval=cfg.deterministic_eval,
		best_model_save_path=cfg.logdir+"/checkpoint/best_eval",
	)

	if run_initial_eval and eval_episodes > 0:
		logging_callback.evaluate(model, deterministic=False)
		if cfg.deterministic_eval:
			logging_callback.evaluate(model, deterministic=True)
		logging_callback.log_count += 1

	if cfg.load_offline_data:
		phase_time = time.time()
		load_offline_data(model, cfg.offline_data_path, num_env)
		print(f"[timing] load_offline_data: {time.time() - phase_time:.1f}s")
	if cfg.train.init_rollout_steps > 0:
		print(
			f"[setup] collecting initial rollouts: "
			f"steps={cfg.train.init_rollout_steps}, n_envs={num_env}, act_steps={cfg.act_steps}"
		)
		phase_time = time.time()
		collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)	
		logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)
		print(f"[timing] initial rollouts: {time.time() - phase_time:.1f}s")

	callbacks = [checkpoint_callback, logging_callback]
	# Train the agent
	print(
		f"[setup] starting learn: total_timesteps={total_timesteps}, "
		f"batch_size={cfg.train.batch_size}, utd={cfg.train.utd}, "
		f"noise_critic_grad_steps={cfg.train.noise_critic_grad_steps}, "
		f"actor_gradient_steps={OmegaConf.select(cfg, 'train.actor_gradient_steps', default=-1)}"
	)
	model.learn(
		total_timesteps=total_timesteps,
		callback = callbacks
	)
	print(f"[timing] total runtime: {time.time() - start_time:.1f}s")

	# Save the final model
	if len(cfg.name) > 0:
		model.save(cfg.logdir+"/checkpoint/final")

	# Close environment and wandb
	env.close()
	logging_callback.close_eval_env()
	if cfg.use_wandb:
		wandb.finish()


if __name__ == "__main__":
	main()
