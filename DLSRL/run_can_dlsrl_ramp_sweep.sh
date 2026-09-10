#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

python_bin="${DLSRL_PYTHON:-${DSRL_PYTHON:-/home/mlo/miniconda3/envs/dsrl/bin/python}}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

# Run sequentially on one GPU. The dedicated config fixes every non-schedule
# DLSRL setting, so only ramp_steps and seed differ between jobs.
ramps=(0 25000 125000 250000)
seeds=(1 2 3)

for ramp in "${ramps[@]}"; do
  for seed in "${seeds[@]}"; do
    command=(
      "$python_bin" -u train_dsrl.py
      --config-path=cfg/robomimic
      --config-name=dlsrl_can_transformer_ramp.yaml
      "seed=$seed"
      "adapter.ramp_steps=$ramp"
      wandb.group=can-dlsrl-ramp
    )

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'ramp=%s seed=%s: ' "$ramp" "$seed"
      printf '%q ' "${command[@]}"
      printf '\n'
    else
      echo "Starting Can DLSRL ramp experiment: ramp=$ramp seed=$seed"
      "${command[@]}"
    fi
  done
done
