#!/usr/bin/env bash
# Pure dynamic obstacle scene (0 static + 120 dynamic) evaluation for v3:
# Suggestion 1 (dual event channels + 24x32 early conv + static-edge suppression)
# + Method A (ego-frame dynamic obstacle sector proximity/velocity auxiliary head).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMNI_KIT_ACCEPT_EULA=YES

UPDATES="${UPDATES:-256}"
NUM_ENVS="${NUM_ENVS:-1024}"
COEF_AUX="${COEF_AUX:-0.5}"
ROOT="${ROOT:-logs/pure_dynamic_v3_20261116}"

if [[ -e "$ROOT/driver.log" || -e "$ROOT/depth_event_v3/config.json" ]]; then
  echo "Experiment exists; refusing to overwrite: $ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT/tensorboard"
export NAVRL_TENSORBOARD_DIR="$PWD/$ROOT/tensorboard"
date -Is > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/parent_commit.txt"
git status --short > "$ROOT/source_status.txt"
printf 'Pure dynamic v3 | 0 static + 120 dynamic | %s envs | %s updates | H96 | 20 Hz | wait_before_truncate=1.0s | goals in END_CLEARANCE | dual_event(24x32 early conv, history=4, pose=estimated, edge_suppression=1) + dyn_aux(coef=%s, sectors=8) | cuRoboV2 SDF | linear goal | seed=terrain_seed=0 | eval 256 ep seed=11000\n' \
  "$NUM_ENVS" "$UPDATES" "$COEF_AUX" > "$ROOT/protocol.txt"

name="depth_event_v3"
echo "START_TRAIN name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"
.venv/bin/python -u scripts/train_zhang.py \
  --log_dir "$ROOT/$name" \
  --stage pretrain_s1 \
  --num_static_obstacles 0 \
  --num_dynamic_obstacles 120 \
  --num_envs "$NUM_ENVS" \
  --report_episodes "$NUM_ENVS" \
  --max_iterations "$UPDATES" \
  --timesteps 96 \
  --geometry_backend curobo_v2 \
  --timing unified_20hz \
  --algorithm bptt \
  --variant dcol \
  --goal_speed_profile linear \
  --velocity_loss_mode full_window \
  --velocity_window_s 1.0 \
  --coef_v 1.2 \
  --coef_collide 10.0 \
  --coef_obj_avoidance 2.5 \
  --grad_decay 0.4 \
  --position_integration trapezoid \
  --loss_timing pre_reset \
  --lr 0.001 \
  --seed 0 \
  --terrain_seed 0 \
  --depth_event_channel 1 \
  --depth_event_dual_channel 1 \
  --depth_event_edge_suppression 1 \
  --depth_event_history_frames 4 \
  --depth_event_pose_source estimated \
  --coef_dyn_aux "$COEF_AUX" \
  > "$ROOT/$name.stdout.log" 2>&1
echo "DONE_TRAIN name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"

ckpt="$ROOT/$name/checkpoint_best_pretrain_s1.pt"
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
date -Is > "$ROOT/finished_at.txt"
