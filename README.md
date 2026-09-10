# DLSRL

The code in this toolbox implements "Beyond Noise Steering: Dual‑Latent Space Reinforcement Learning for Generative Robot Policy" by <i>P. Zhang, T. Sun, X. Xiu</i>.

## Overview

A pretrained diffusion policy generates action sequences by iteratively denoising conditioned on observations. DLSRL extends DSRL's noise-space steering with hidden latents, providing two complementary ways to control the policy:

- **Noise latents** control the starting point of denoising to guide action generation.
- **Hidden latents** modulate internal Transformer representations to provide additional control over the generation process.

The base diffusion policy remains fixed while the steering policy learns from environment interactions and rewards. The current robotic task configurations use the DSRL-NA training backend by default.

## Framework

![Comparison of DSRL and DLSRL](assets/dlsrl_compare.png)

*Noise-space steering in DSRL and dual-latent-space steering in DLSRL.* [PDF](assets/dlsrl_compare.pdf)

## Features

- Jointly learns noise and hidden latents to expand control over a pretrained diffusion policy.
- Reuses the pretrained policy with warmup and gradual ramp-up schedules for the hidden branch.
- Includes Lift, Can, Square, and Transport configurations, DSRL baselines, and ablation experiments with multiple random seeds.

## Installation

Run the following commands from the repository root.

**1. Create the environment**

```bash
conda create -n dlsrl python=3.9 -y
conda activate dlsrl
```

**2. Install dependencies**

```bash
python -m pip install -e ./dppo
python -m pip install -e ./robosuite_src
python -m pip install 'robomimic==0.3.0' 'cython<3' patchelf
python -m pip install -e ./stable-baselines3
python -m pip install tensorboard 'shimmy>=0.2.1'
```

Use the local installation commands above: the `dppo[robomimic]` extra still references a path on the original development machine.

**3. Install D4RL dependencies (optional)**

Required only for the Hopper, Walker2d, and HalfCheetah baselines:

```bash
python -m pip install -e './dppo[gym]'
```

Also configure a compatible MuJoCo / mujoco-py environment. When using MuJoCo 2.1, the training entry point checks `~/.mujoco/mujoco210`.

## Data and Pretrained Models

The robotic experiments use low-dimensional Robomimic observations. Prepare base policy weights and normalization statistics that match the task, action dimensions, and Transformer architecture.

**1. Prepare resources**

The original DSRL README provides [this resource link](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1?usp=share_link). DLSRL Transformer checkpoints must be prepared separately; MLP checkpoints are not interchangeable with them. Pretraining code is available in [dppo/agent/pretrain](dppo/agent/pretrain). These weights and data are not included in this repository.

**2. Set resource paths**

Update the following fields in the selected task's YAML configuration:

```yaml
base_policy_path: /path/to/checkpoint.pt
normalization_path: /path/to/normalization.npz
```

To load offline data with the `dsrl_na` backend, set `load_offline_data=True` and `offline_data_path`. The NPZ file must contain `states`, `states_next`, `actions`, `rewards`, and `terminals`. By default, initial data is collected through interactions using the base policy.

## Training

### DLSRL

To train on Can:

```bash
python train_dsrl.py \
  --config-path=cfg/robomimic \
  --config-name=dlsrl_can_transformer.yaml \
  use_wandb=False
```

For other tasks, replace `--config-name` with the corresponding configuration:

| Task      | Configuration                                                                      |
| --------- | ---------------------------------------------------------------------------------- |
| Lift      | [dlsrl_lift_transformer.yaml](cfg/robomimic/dlsrl_lift_transformer.yaml)           |
| Can       | [dlsrl_can_transformer.yaml](cfg/robomimic/dlsrl_can_transformer.yaml)             |
| Square    | [dlsrl_square_transformer.yaml](cfg/robomimic/dlsrl_square_transformer.yaml)       |
| Transport | [dlsrl_transport_transformer.yaml](cfg/robomimic/dlsrl_transport_transformer.yaml) |

The Can configuration uses 400,000 online action-chunk transitions, plus 75,050 initial collection transitions (1,501 vectorized calls × 50 environments). It uses 50 training and 25 evaluation environments, with up to 4 action steps per chunk. Other tasks use the settings in their respective YAML files.

Append overrides such as `seed=2 device=cuda:0 env.n_envs=10` to the command. The internal `adapter` parameters control DLSRL's hidden branch; Hydra manages model, data, and training settings.

### DSRL Baselines

To run the baseline with the same pretrained Can Transformer policy:

```bash
python train_dsrl.py \
  --config-path=cfg/robomimic \
  --config-name=dsrl_can_transformer.yaml \
  +total_timesteps=400000 \
  use_wandb=False
```

Example D4RL baseline:

```bash
python train_dsrl.py \
  --config-path=cfg/gym \
  --config-name=dsrl_hopper.yaml \
  use_wandb=False
```

### Ablation Experiments

Compare four hidden-branch ramp-up durations on Can, each with three random seeds:

```bash
python train_dsrl.py \
  --config-path=cfg/robomimic \
  --config-name=dlsrl_can_transformer_ramp.yaml -m \
  seed=1,2,3 \
  adapter.ramp_steps=0,25000,125000,250000 \
  use_wandb=False
```


### Acknowledgement
Please contact P. Zhang for more details.
