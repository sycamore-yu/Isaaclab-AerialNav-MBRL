#!/usr/bin/env bash
# Full S0 -> S1 curriculum for v4 (Ego-Compensated Kinematic Event Field
# + Zero-Init Residual Dual-Stream + LoS Inv-TTC Aux + Spatio-Temporal BPTT).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMNI_KIT_ACCEPT_EULA=YES

NUM_ENVS="${NUM_ENVS:-1024}"
S0_MAX_UPDATES="${S0_MAX_UPDATES:-500}"
S1_MAX_UPDATES="${S1_MAX_UPDATES:-512}"
ROOT="${ROOT:-logs/depth_event_v4_s0_s1_20261117}"

if [[ -e "$ROOT/driver.log" || -e "$ROOT/s0/config.json" || -e "$ROOT/s1/config.json" ]]; then
  echo "Experiment exists; refusing to overwrite: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT/tensorboard"
export NAVRL_TENSORBOARD_DIR="$PWD/$ROOT/tensorboard"
date -Is > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/parent_commit.txt"
git status --short > "$ROOT/source_status.txt"
printf 'DiffNav v4 (Kinematic Event Field + Zero-Init Dual Stream + LoS Inv-TTC Aux + Temporal Yielding + Active Yaw) | S0 -> S1 | %s envs\n' "$NUM_ENVS" > "$ROOT/protocol.txt"

COMMON_ARGS=(
  --num_envs "$NUM_ENVS"
  --report_episodes "$NUM_ENVS"
  --seed 0
  --terrain_seed 0
  --timesteps 96
  --depth_event_channel 1
  --depth_event_kinematic_field 1
  --depth_event_history_frames 4
  --depth_event_pose_source estimated
  --coef_dyn_aux 0.5
  --dyn_aux_sectors 6
  --aux_from_perception 1
  --aux_inv_ttc 1
  --aux_survival_gate_s 5.0
  --smooth_approach_gate 1
  --temporal_yielding_lambda 1.5
  --temporal_yielding_floor 0.2
  --active_yaw_alignment 1.0
  --geometry_backend curobo_v2
  --timing unified_20hz
  --algorithm bptt
  --variant dcol
  --goal_speed_profile linear
  --velocity_loss_mode full_window
  --velocity_window_s 1.0
  --coef_v 1.2
  --coef_collide 10.0
  --coef_obj_avoidance 2.5
  --grad_decay 0.4
  --position_integration trapezoid
  --loss_timing pre_reset
  --lr 0.001
)

echo "START_S0 at=$(date -Is)" | tee -a "$ROOT/driver.log"
.venv/bin/python -u scripts/train_zhang.py \
  --log_dir "$ROOT/s0" --stage pretrain_s0 \
  --stop_after_acceptance --max_iterations "$S0_MAX_UPDATES" \
  "${COMMON_ARGS[@]}" > "$ROOT/s0.stdout.log" 2>&1
echo "DONE_S0 at=$(date -Is)" | tee -a "$ROOT/driver.log"

s0_ckpt="$ROOT/s0/checkpoint_stage.pt"
if [[ ! -f "$s0_ckpt" ]]; then
  s0_ckpt="$ROOT/s0/checkpoint_best_pretrain_s0.pt"
fi

.venv/bin/python -u scripts/evaluate.py --method zhang \
  --checkpoint "$s0_ckpt" --output "$ROOT/s0-evaluation.json" \
  --stage pretrain_s0 --static_obstacles 240 --dynamic_obstacles 0 \
  --geometry_backend curobo_v2 --episodes 256 --seed 11000 --terrain_seed 11000 \
  > "$ROOT/s0-evaluation.stdout.log" 2>&1
echo "DONE_S0_EVAL at=$(date -Is)" | tee -a "$ROOT/driver.log"

echo "START_S1 from=$s0_ckpt at=$(date -Is)" | tee -a "$ROOT/driver.log"
.venv/bin/python -u scripts/train_zhang.py \
  --log_dir "$ROOT/s1" --stage pretrain_s1 \
  --checkpoint "$s0_ckpt" \
  --stop_after_acceptance --max_iterations "$S1_MAX_UPDATES" \
  "${COMMON_ARGS[@]}" > "$ROOT/s1.stdout.log" 2>&1
echo "DONE_S1 at=$(date -Is)" | tee -a "$ROOT/driver.log"

for tag in best_pretrain_s1 stage final; do
  ckpt="$ROOT/s1/checkpoint_${tag}.pt"
  if [[ -f "$ckpt" ]]; then
    extra_trace=()
    if [[ "$tag" == "best_pretrain_s1" || "$tag" == "stage" ]]; then
      extra_trace=(--trace_output "$ROOT/s1-${tag}-traces.json" --trace_episodes 16)
    fi
    .venv/bin/python -u scripts/evaluate.py --method zhang \
      --checkpoint "$ckpt" --output "$ROOT/s1-${tag}-evaluation.json" \
      --stage pretrain_s1 --static_obstacles 240 --dynamic_obstacles 120 \
      --geometry_backend curobo_v2 --episodes 256 --seed 11000 --terrain_seed 11000 \
      "${extra_trace[@]}" \
      > "$ROOT/s1-${tag}-evaluation.stdout.log" 2>&1
  fi
done
echo "DONE_S1_EVAL at=$(date -Is)" | tee -a "$ROOT/driver.log"
date -Is > "$ROOT/finished_at.txt"
