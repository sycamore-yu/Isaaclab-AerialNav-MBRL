#!/usr/bin/env bash
# S0 -> S1 curriculum for v3 (Suggestion 1 + Method A):
# Dual-channel (appeared, disappeared) 24x32 early spatial conv + static-edge suppression
# + training-only ego-frame dynamic obstacle sector proximity/velocity auxiliary head.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMNI_KIT_ACCEPT_EULA=YES

NUM_ENVS="${NUM_ENVS:-1024}"
S0_MAX_UPDATES="${S0_MAX_UPDATES:-1000}"
S1_MAX_UPDATES="${S1_MAX_UPDATES:-1024}"
COEF_AUX="${COEF_AUX:-0.5}"
ROOT="${ROOT:-logs/depth_event_v3_s0_s1_20261116}"

if [[ -e "$ROOT/driver.log" || -e "$ROOT/s0/config.json" || -e "$ROOT/s1/config.json" ]]; then
  echo "Experiment exists; refusing to overwrite: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT/tensorboard"
export NAVRL_TENSORBOARD_DIR="$PWD/$ROOT/tensorboard"
date -Is > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/parent_commit.txt"
git status --short > "$ROOT/source_status.txt"
printf 'DiffNav CNN-GRU192 v3 (Suggestion 1 + Method A) | S0 -> S1 | %s envs | 20 Hz | wait_before_truncate=1.0s | goals in END_CLEARANCE | 3x24x32 dual events + early conv + edge suppression + dyn_aux(coef=%s, sectors=8) | eval 256 ep seed=11000\n' \
  "$NUM_ENVS" "$COEF_AUX" > "$ROOT/protocol.txt"

COMMON_ARGS=(
  --num_envs "$NUM_ENVS"
  --report_episodes "$NUM_ENVS"
  --seed 0
  --terrain_seed 0
  --timesteps 96
  --depth_event_channel 1
  --depth_event_dual_channel 1
  --depth_event_edge_suppression 1
  --depth_event_history_frames 4
  --depth_event_pose_source estimated
  --coef_dyn_aux "$COEF_AUX"
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
    .venv/bin/python -u scripts/evaluate.py --method zhang \
      --checkpoint "$ckpt" --output "$ROOT/s1-${tag}-evaluation.json" \
      --stage pretrain_s1 --static_obstacles 240 --dynamic_obstacles 120 \
      --geometry_backend curobo_v2 --episodes 256 --seed 11000 --terrain_seed 11000 \
      > "$ROOT/s1-${tag}-evaluation.stdout.log" 2>&1
  fi
done
echo "DONE_S1_EVAL at=$(date -Is)" | tee -a "$ROOT/driver.log"
date -Is > "$ROOT/finished_at.txt"
