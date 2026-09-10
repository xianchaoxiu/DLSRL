import math
from collections import namedtuple
from typing import Optional, Union

import torch
import torch.nn as nn


Sample = namedtuple("Sample", "trajectories chains")


class SinusoidalPosEmb(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		device = x.device
		half_dim = self.dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
		emb = x[:, None] * emb[None, :]
		emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
		return emb


class OfficialTransformerForDiffusion(nn.Module):
	"""Lightweight copy of real-stanford/diffusion_policy's lowdim Transformer."""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		horizon: int,
		n_obs_steps: Optional[int] = None,
		cond_dim: int = 0,
		n_layer: int = 12,
		n_head: int = 12,
		n_emb: int = 768,
		p_drop_emb: float = 0.1,
		p_drop_attn: float = 0.1,
		causal_attn: bool = False,
		time_as_cond: bool = True,
		obs_as_cond: bool = False,
		n_cond_layers: int = 0,
	):
		super().__init__()
		if n_obs_steps is None:
			n_obs_steps = horizon

		T = horizon
		T_cond = 1
		if not time_as_cond:
			T += 1
			T_cond -= 1
		obs_as_cond = cond_dim > 0
		if obs_as_cond:
			if not time_as_cond:
				raise ValueError("obs_as_cond requires time_as_cond")
			T_cond += n_obs_steps

		self.input_emb = nn.Linear(input_dim, n_emb)
		self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
		self.drop = nn.Dropout(p_drop_emb)
		self.time_emb = SinusoidalPosEmb(n_emb)
		self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if obs_as_cond else None

		self.cond_pos_emb = None
		self.encoder = None
		self.decoder = None
		encoder_only = False
		if T_cond > 0:
			self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
			if n_cond_layers > 0:
				encoder_layer = nn.TransformerEncoderLayer(
					d_model=n_emb,
					nhead=n_head,
					dim_feedforward=4 * n_emb,
					dropout=p_drop_attn,
					activation="gelu",
					batch_first=True,
					norm_first=True,
				)
				self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_cond_layers)
			else:
				self.encoder = nn.Sequential(
					nn.Linear(n_emb, 4 * n_emb),
					nn.Mish(),
					nn.Linear(4 * n_emb, n_emb),
				)

			decoder_layer = nn.TransformerDecoderLayer(
				d_model=n_emb,
				nhead=n_head,
				dim_feedforward=4 * n_emb,
				dropout=p_drop_attn,
				activation="gelu",
				batch_first=True,
				norm_first=True,
			)
			self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
		else:
			encoder_only = True
			encoder_layer = nn.TransformerEncoderLayer(
				d_model=n_emb,
				nhead=n_head,
				dim_feedforward=4 * n_emb,
				dropout=p_drop_attn,
				activation="gelu",
				batch_first=True,
				norm_first=True,
			)
			self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

		if causal_attn:
			mask = (torch.triu(torch.ones(T, T)) == 1).transpose(0, 1)
			mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
			self.register_buffer("mask", mask)
			if time_as_cond and obs_as_cond:
				t, s = torch.meshgrid(torch.arange(T), torch.arange(T_cond), indexing="ij")
				memory_mask = t >= (s - 1)
				memory_mask = memory_mask.float().masked_fill(memory_mask == 0, float("-inf")).masked_fill(memory_mask == 1, 0.0)
				self.register_buffer("memory_mask", memory_mask)
			else:
				self.memory_mask = None
		else:
			self.mask = None
			self.memory_mask = None

		self.ln_f = nn.LayerNorm(n_emb)
		self.head = nn.Linear(n_emb, output_dim)
		self.T = T
		self.T_cond = T_cond
		self.horizon = horizon
		self.time_as_cond = time_as_cond
		self.obs_as_cond = obs_as_cond
		self.encoder_only = encoder_only

	def _adapter_delta(
		self,
		x: torch.Tensor,
		adapter_feature: Optional[torch.Tensor],
		adapter_gate: Optional[torch.Tensor],
		adapter_scale: float,
	):
		if adapter_feature is None:
			return None
		if adapter_feature.shape[:2] != x.shape[:2]:
			raise ValueError(
				f"adapter_feature shape {tuple(adapter_feature.shape)} must match "
				f"hidden token shape (*, {x.shape[1]}, {x.shape[2]})"
			)
		if adapter_feature.shape[-1] != x.shape[-1]:
			raise ValueError(
				f"adapter_feature dim {adapter_feature.shape[-1]} does not match "
				f"Transformer hidden dim {x.shape[-1]}"
			)
		if adapter_gate is None:
			adapter_gate = torch.ones((*adapter_feature.shape[:-1], 1), device=x.device, dtype=x.dtype)
		if adapter_gate.shape[:2] != x.shape[:2]:
			raise ValueError(
				f"adapter_gate shape {tuple(adapter_gate.shape)} must match "
				f"hidden token shape (*, {x.shape[1]}, 1)"
			)
		return float(adapter_scale) * adapter_gate.to(dtype=x.dtype) * torch.tanh(adapter_feature.to(dtype=x.dtype))

	def _inject_adapter(
		self,
		x: torch.Tensor,
		adapter_feature: Optional[torch.Tensor],
		adapter_gate: Optional[torch.Tensor],
		adapter_scale: float,
	):
		delta = self._adapter_delta(x, adapter_feature, adapter_gate, adapter_scale)
		if delta is None:
			return x
		return x + delta

	def _run_decoder_with_adapter(
		self,
		tgt: torch.Tensor,
		memory: torch.Tensor,
		adapter_feature: Optional[torch.Tensor],
		adapter_gate: Optional[torch.Tensor],
		adapter_scale: float,
	):
		x = tgt
		for layer in self.decoder.layers:
			x = layer(
				x,
				memory,
				tgt_mask=self.mask,
				memory_mask=self.memory_mask,
			)
			x = self._inject_adapter(x, adapter_feature, adapter_gate, adapter_scale)
		if self.decoder.norm is not None:
			x = self.decoder.norm(x)
		return x

	def _run_encoder_with_adapter(
		self,
		src: torch.Tensor,
		adapter_feature: Optional[torch.Tensor],
		adapter_gate: Optional[torch.Tensor],
		adapter_scale: float,
	):
		x = src
		for layer in self.encoder.layers:
			x = layer(x, src_mask=self.mask)
			action_x = x[:, 1:, :]
			action_x = self._inject_adapter(action_x, adapter_feature, adapter_gate, adapter_scale)
			x = torch.cat([x[:, :1, :], action_x], dim=1)
		if self.encoder.norm is not None:
			x = self.encoder.norm(x)
		return x

	def forward(
		self,
		sample: torch.Tensor,
		timestep: Union[torch.Tensor, float, int],
		cond: Optional[torch.Tensor] = None,
		adapter_feature: Optional[torch.Tensor] = None,
		adapter_gate: Optional[torch.Tensor] = None,
		adapter_scale: float = 0.0,
	):
		if not torch.is_tensor(timestep):
			timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
		elif len(timestep.shape) == 0:
			timestep = timestep[None].to(sample.device)
		timestep = timestep.expand(sample.shape[0])
		time_emb = self.time_emb(timestep).unsqueeze(1)
		input_emb = self.input_emb(sample)

		if self.encoder_only:
			token_embeddings = torch.cat([time_emb, input_emb], dim=1)
			position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]
			x = self.drop(token_embeddings + position_embeddings)
			x = self._run_encoder_with_adapter(
				x,
				adapter_feature=adapter_feature,
				adapter_gate=adapter_gate,
				adapter_scale=adapter_scale,
			)
			x = x[:, 1:, :]
		else:
			cond_embeddings = time_emb
			if self.obs_as_cond:
				cond_obs_emb = self.cond_obs_emb(cond)
				cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
			position_embeddings = self.cond_pos_emb[:, : cond_embeddings.shape[1], :]
			x = self.drop(cond_embeddings + position_embeddings)
			memory = self.encoder(x)

			position_embeddings = self.pos_emb[:, : input_emb.shape[1], :]
			x = self.drop(input_emb + position_embeddings)
			x = self._run_decoder_with_adapter(
				x,
				memory,
				adapter_feature=adapter_feature,
				adapter_gate=adapter_gate,
				adapter_scale=adapter_scale,
			)

		x = self.ln_f(x)
		return self.head(x)


def _load_torch_state(path):
	try:
		return torch.load(path, map_location="cpu", weights_only=True)
	except TypeError:
		return torch.load(path, map_location="cpu")


def squaredcos_cap_v2_betas(num_train_timesteps: int, max_beta: float = 0.999):
	def alpha_bar(time_step):
		return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

	betas = []
	for i in range(num_train_timesteps):
		t1 = i / num_train_timesteps
		t2 = (i + 1) / num_train_timesteps
		betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
	return torch.tensor(betas, dtype=torch.float32)


class OfficialDiffusionPolicyAdapter(nn.Module):
	"""Adapter that exposes official Diffusion Policy Transformer as DSRL base_policy."""

	def __init__(
		self,
		checkpoint_path: str,
		device: str = "cuda:0",
		num_inference_steps: Optional[int] = None,
		sampler: str = "ddim",
		clip_sample: bool = True,
		**unused_kwargs,
	):
		super().__init__()
		payload = _load_torch_state(checkpoint_path)
		cfg = dict(payload["config"])
		if sampler != "ddim":
			raise ValueError("Only deterministic DDIM sampling is supported for DSRL control noise.")

		self.single_obs_dim = int(cfg["obs_dim"])
		self.action_dim = int(cfg["action_dim"])
		self.horizon = int(cfg["horizon"])
		self.n_obs_steps = int(cfg["n_obs_steps"])
		self.n_action_steps = int(cfg["n_action_steps"])
		self.start = self.n_obs_steps - 1
		self.end = self.start + self.n_action_steps
		self.num_train_timesteps = int(cfg["num_train_timesteps"])
		self.num_inference_steps = int(num_inference_steps or self.num_train_timesteps)
		self.clip_sample = bool(clip_sample)
		self.hidden_dim = int(cfg["n_emb"])

		self.model = OfficialTransformerForDiffusion(
			input_dim=int(cfg["input_dim"]),
			output_dim=int(cfg["output_dim"]),
			horizon=self.horizon,
			n_obs_steps=self.n_obs_steps,
			cond_dim=int(cfg["cond_dim"]),
			n_layer=int(cfg["n_layer"]),
			n_head=int(cfg["n_head"]),
			n_emb=int(cfg["n_emb"]),
			p_drop_emb=float(cfg["p_drop_emb"]),
			p_drop_attn=float(cfg["p_drop_attn"]),
			causal_attn=bool(cfg["causal_attn"]),
			time_as_cond=bool(cfg["time_as_cond"]),
			obs_as_cond=bool(cfg["obs_as_cond"]),
			n_cond_layers=int(cfg["n_cond_layers"]),
		)
		self.model.load_state_dict(payload["model_state"], strict=True)

		normalizer = payload["normalizer"]
		self.register_buffer("action_offset", normalizer["action_offset"].float())
		self.register_buffer("action_scale", normalizer["action_scale"].float())
		self.register_buffer("obs_offset", normalizer["obs_offset"].float())
		self.register_buffer("obs_scale", normalizer["obs_scale"].float())

		betas = squaredcos_cap_v2_betas(self.num_train_timesteps)
		alphas = 1.0 - betas
		self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

		requested_device = torch.device(device)
		if requested_device.type == "cuda" and not torch.cuda.is_available():
			requested_device = torch.device("cpu")
		self.to(requested_device)
		self.eval()

	@property
	def device(self):
		return next(self.parameters()).device

	def _format_obs(self, obs):
		if obs.ndim == 1:
			obs = obs.unsqueeze(0)
		if obs.ndim == 3:
			if obs.shape[1:] != (self.n_obs_steps, self.single_obs_dim):
				raise ValueError(f"Expected obs shape (*, {self.n_obs_steps}, {self.single_obs_dim}), got {tuple(obs.shape)}")
			obs_seq = obs
		elif obs.shape[-1] == self.single_obs_dim * self.n_obs_steps:
			obs_seq = obs.reshape(obs.shape[0], self.n_obs_steps, self.single_obs_dim)
		elif obs.shape[-1] == self.single_obs_dim:
			obs_seq = obs.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
		else:
			raise ValueError(
				f"Expected flattened obs dim {self.single_obs_dim * self.n_obs_steps} "
				f"or single obs dim {self.single_obs_dim}, got {obs.shape[-1]}"
			)
		return obs_seq * self.obs_scale + self.obs_offset

	def _format_initial_trajectory(self, noise_action):
		if noise_action.ndim == 2:
			if noise_action.shape[-1] == self.n_action_steps * self.action_dim:
				noise_action = noise_action.reshape(noise_action.shape[0], self.n_action_steps, self.action_dim)
			elif noise_action.shape[-1] == self.horizon * self.action_dim:
				noise_action = noise_action.reshape(noise_action.shape[0], self.horizon, self.action_dim)
			else:
				raise ValueError(f"Unexpected noise action shape {tuple(noise_action.shape)}")
		if noise_action.ndim != 3 or noise_action.shape[-1] != self.action_dim:
			raise ValueError(f"Unexpected noise action shape {tuple(noise_action.shape)}")

		if noise_action.shape[1] == self.horizon:
			return noise_action
		if noise_action.shape[1] != self.n_action_steps:
			raise ValueError(f"Expected {self.n_action_steps} or {self.horizon} action steps, got {noise_action.shape[1]}")

		trajectory = noise_action.new_zeros(noise_action.shape[0], self.horizon, self.action_dim)
		trajectory[:, self.start : self.end] = noise_action
		if self.start > 0:
			trajectory[:, : self.start] = noise_action[:, :1]
		if self.end < self.horizon:
			trajectory[:, self.end :] = noise_action[:, -1:]
		return trajectory

	def _format_adapter_trajectory(self, value, feature_dim):
		if value is None:
			return None
		if value.ndim == 2:
			value = value.unsqueeze(1)
		if value.ndim != 3 or value.shape[-1] != feature_dim:
			raise ValueError(f"Unexpected adapter tensor shape {tuple(value.shape)}")
		if value.shape[1] == self.horizon:
			return value
		if value.shape[1] == 1:
			value = value.repeat(1, self.n_action_steps, 1)
		if value.shape[1] != self.n_action_steps:
			raise ValueError(f"Expected 1, {self.n_action_steps}, or {self.horizon} adapter steps, got {value.shape[1]}")
		trajectory = value.new_zeros(value.shape[0], self.horizon, feature_dim)
		trajectory[:, self.start : self.end] = value
		if self.start > 0:
			trajectory[:, : self.start] = value[:, :1]
		if self.end < self.horizon:
			trajectory[:, self.end :] = value[:, -1:]
		return trajectory

	def _ddim_sample(self, trajectory, obs_cond, adapter_feature=None, adapter_gate=None, adapter_scale: float = 0.0):
		timesteps = torch.linspace(
			self.num_train_timesteps - 1,
			0,
			self.num_inference_steps,
			device=trajectory.device,
			dtype=torch.long,
		)
		for i, t in enumerate(timesteps):
			t_batch = t.expand(trajectory.shape[0])
			model_output = self.model(
				trajectory,
				t_batch,
				obs_cond,
				adapter_feature=adapter_feature,
				adapter_gate=adapter_gate,
				adapter_scale=adapter_scale,
			)
			alpha_prod_t = self.alphas_cumprod[t]
			if i + 1 < len(timesteps):
				alpha_prod_prev = self.alphas_cumprod[timesteps[i + 1]]
			else:
				alpha_prod_prev = torch.ones((), device=trajectory.device, dtype=trajectory.dtype)

			pred_original = (trajectory - (1 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt()
			if self.clip_sample:
				pred_original = pred_original.clamp(-1.0, 1.0)
			trajectory = alpha_prod_prev.sqrt() * pred_original + (1 - alpha_prod_prev).sqrt() * model_output
		return trajectory

	def forward(self, cond, deterministic: bool = True):
		obs = cond["state"].to(device=self.device, dtype=torch.float32)
		noise_action = cond["noise_action"].to(device=self.device, dtype=torch.float32)
		adapter_feature = cond.get("adapter_feature", None)
		adapter_gate = cond.get("adapter_gate", None)
		adapter_scale = float(cond.get("adapter_scale", 0.0))
		if adapter_feature is not None:
			adapter_feature = adapter_feature.to(device=self.device, dtype=torch.float32)
			adapter_feature = self._format_adapter_trajectory(adapter_feature, self.hidden_dim)
		if adapter_gate is not None:
			adapter_gate = adapter_gate.to(device=self.device, dtype=torch.float32)
			adapter_gate = self._format_adapter_trajectory(adapter_gate, 1)
		obs_cond = self._format_obs(obs)
		trajectory = self._format_initial_trajectory(noise_action)
		trajectory = self._ddim_sample(
			trajectory,
			obs_cond,
			adapter_feature=adapter_feature,
			adapter_gate=adapter_gate,
			adapter_scale=adapter_scale,
		)
		normalized_action = trajectory[:, self.start : self.end]
		action = (normalized_action - self.action_offset) / self.action_scale
		return Sample(action, None)
