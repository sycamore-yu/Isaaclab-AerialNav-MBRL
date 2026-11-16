#!/usr/bin/env bash
# Paired A/B test on pure dynamic obstacle scene (0 static + 120 dynamic):
# depth_only vs 4-frame depth_event (v2, estimated camera pose).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMNI_KIT_ACCEPT_EULA=YES

UPDATES="${UPDATES:-256}"
NUM_ENVS="${NUM_ENVS:-1024}"
ROOT="${ROOT:-logs/pure_dynamic_ab_20261116}"

if [[ -e "$ROOT/driver.log" || -e "$ROOT/depth_only/config.json" || -e "$ROOT/depth_event/config.json" ]]; then
  echo "Experiment exists; refusing to overwrite: $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT/tensorboard"
export NAVRL_TENSORBOARD_DIR="$PWD/$ROOT/tensorboard"
date -Is > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/parent_commit.txt"
git status --short > "$ROOT/source_status.txt"
printf 'Pure dynamic A/B | 0 static + 120 dynamic | %s envs | %s updates | H96 | 20 Hz | wait_before_truncate=1.0s | goals in END_CLEARANCE | depth_only vs depth_event(history=4, pose=estimated) | cuRoboV2 SDF | linear goal | seed=terrain_seed=0 | eval 256 ep seed=11000\n' \
  "$NUM_ENVS" "$UPDATES" > "$ROOT/protocol.txt"

COMMON_TRAIN=(
  --stage pretrain_s1
  --num_static_obstacles 0
  --num_dynamic_obstacles 120
  --num_envs "$NUM_ENVS"
  --report_episodes "$NUM_ENVS"
  --max_iterations "$UPDATES"
  --timesteps 96
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
  --seed 0
  --terrain_seed 0
)

run_arm() {
  local name="$1"
  shift
  echo "START_TRAIN name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"
  .venv/bin/python -u scripts/train_zhang.py \
    --log_dir "$ROOT/$name" "${COMMON_TRAIN[@]}" "$@" \
    > "$ROOT/$name.stdout.log" 2>&1
  echo "DONE_TRAIN name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"

  local ckpt="$ROOT/$name/checkpoint_best_pretrain_s1.pt"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$ROOT/$name/checkpoint_final.pt"
  fi
  echo "START_EVAL name=$name ckpt=$ckpt at=$(date -Is)" | tee -a "$ROOT/driver.log"
  .venv/bin/python -u scripts/evaluate.py \
    --method zhang \
    --checkpoint "$ckpt" \
    --output "$ROOT/${name}-best-evaluation.json" \
    --stage pretrain_s1 \
    --static_obstacles 0 \
    --dynamic_obstacles 120 \
    --geometry_backend curobo_v2 \
    --episodes 256 \
    --seed 11000 \
    --terrain_seed 11000 \
    > "$ROOT/${name}-best-evaluation.stdout.log" 2>&1
  if [[ -f "$ROOT/$name/checkpoint_final.pt" && "$ckpt" != "$ROOT/$name/checkpoint_final.pt" ]]; then
    .venv/bin/python -u scripts/evaluate.py \
      --method zhang \
      --checkpoint "$ROOT/$name/checkpoint_final.pt" \
      --output "$ROOT/${name}-final-evaluation.json" \
      --stage pretrain_s1 \
      --static_obstacles 0 \
      --dynamic_obstacles 120 \
      --geometry_backend curobo_v2 \
      --episodes 256 \
      --seed 11000 \
      --terrain_seed 11000 \
      > "$ROOT/${name}-final-evaluation.stdout.log" 2>&1
  fi
  echo "DONE_EVAL name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"
}

run_arm depth_only --depth_event_channel 0 --depth_event_history_frames 1
run_arm depth_event --depth_event_channel 1 --depth_event_history_frames 4 --depth_event_pose_source estimated

date -Is > "$ROOT/finished_at.txt"
echo "PAIRED_COMPLETED at=$(date -Is)" | tee -a "$ROOT/driver.log"
