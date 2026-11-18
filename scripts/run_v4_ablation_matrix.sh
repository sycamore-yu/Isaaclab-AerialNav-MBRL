#!/usr/bin/env bash
# Pure-dynamic & S1 evaluation matrix (v4.1 mainline vs depth_only + odometry drift stress test):
#   1. depth_only
#   2. v4_kinematic (EG-TS + Radial Looming + Zero-Init Dual Stream + Inv-TTC Aux + Yielding)
#   3. ppo_navrl (optional, RUN_PPO=1)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMNI_KIT_ACCEPT_EULA=YES

UPDATES="${UPDATES:-256}"
NUM_ENVS="${NUM_ENVS:-1024}"
STATIC="${STATIC:-0}"
DYNAMIC="${DYNAMIC:-120}"
EVAL_EPISODES="${EVAL_EPISODES:-256}"
NOISE_SCALES="${NOISE_SCALES:-1.0 2.0 3.0}"
PPO_UPDATES="${PPO_UPDATES:-512}"
ROOT="${ROOT:-logs/v4_ablation_matrix_20261117}"

if [[ -e "$ROOT/driver.log" ]]; then
  echo "Experiment exists; refusing to overwrite: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT/tensorboard" "$ROOT/figures"
export NAVRL_TENSORBOARD_DIR="$PWD/$ROOT/tensorboard"
date -Is > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/parent_commit.txt"

COMMON_ZHANG=(
  --stage pretrain_s1
  --num_static_obstacles "$STATIC"
  --num_dynamic_obstacles "$DYNAMIC"
  --num_envs "$NUM_ENVS"
  --report_episodes "$NUM_ENVS"
  --camera_batch_size 256
  --activation_checkpointing 0
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

eval_arm() {
  local name="$1" ckpt="$2" noise="$3" out_suffix="$4"
  shift 4
  .venv/bin/python -u scripts/evaluate.py \
    --method zhang \
    --checkpoint "$ckpt" \
    --output "$ROOT/${name}-${out_suffix}-evaluation.json" \
    --stage pretrain_s1 \
    --static_obstacles "$STATIC" \
    --dynamic_obstacles "$DYNAMIC" \
    --geometry_backend curobo_v2 \
    --episodes "$EVAL_EPISODES" \
    --seed 11000 \
    --terrain_seed 11000 \
    --noise_scale "$noise" \
    "$@" \
    > "$ROOT/${name}-${out_suffix}-evaluation.stdout.log" 2>&1
}

run_zhang_arm() {
  local name="$1"
  shift
  echo "START_TRAIN name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"
  .venv/bin/python -u scripts/train_zhang.py \
    --log_dir "$ROOT/$name" "${COMMON_ZHANG[@]}" "$@" \
    > "$ROOT/$name.stdout.log" 2>&1
  echo "DONE_TRAIN name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"

  local ckpt="$ROOT/$name/checkpoint_best_pretrain_s1.pt"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$ROOT/$name/checkpoint_final.pt"
  fi
  local trace_flag=()
  if [[ "$name" == "v4_kinematic" ]]; then
    trace_flag=(--trace_output "$ROOT/${name}-traces.json" --trace_episodes 16)
  fi
  eval_arm "$name" "$ckpt" "0.0" "best" "${trace_flag[@]}"
  if [[ "$name" == "v4_kinematic" ]]; then
    for ns in $NOISE_SCALES; do
      eval_arm "$name" "$ckpt" "$ns" "noise${ns}"
    done
  fi
  echo "DONE_EVAL name=$name at=$(date -Is)" | tee -a "$ROOT/driver.log"
}

run_zhang_arm v4_kinematic \
  --depth_event_channel 1 --depth_event_kinematic_field 1 --depth_event_history_frames 4 \
  --depth_event_pose_source estimated --coef_dyn_aux 0.5 --dyn_aux_sectors 6 \
  --aux_from_perception 1 --aux_inv_ttc 1 --aux_survival_gate_s 3.5 \
  --smooth_approach_gate 1 --temporal_yielding_lambda 1.0 --temporal_yielding_floor 0.3 \
  --active_yaw_alignment 0.0

run_zhang_arm depth_only --depth_event_channel 0 --depth_event_history_frames 1

if [[ "${RUN_PPO:-0}" == "1" ]]; then
  echo "START_TRAIN name=ppo_navrl at=$(date -Is)" | tee -a "$ROOT/driver.log"
  .venv/bin/python -u scripts/train_navrl.py \
    --log_dir "$ROOT/ppo_navrl" --stage pretrain_s1 \
    --num_envs "$NUM_ENVS" --max_iterations "$PPO_UPDATES" --seed 0 --terrain_seed 0 \
    > "$ROOT/ppo_navrl.stdout.log" 2>&1 || true
  if [[ -f "$ROOT/ppo_navrl/checkpoint_final.pt" ]]; then
    .venv/bin/python -u scripts/evaluate.py --method navrl \
      --checkpoint "$ROOT/ppo_navrl/checkpoint_final.pt" \
      --output "$ROOT/ppo_navrl-best-evaluation.json" \
      --stage pretrain_s1 --static_obstacles "$STATIC" --dynamic_obstacles "$DYNAMIC" \
      --episodes 256 --seed 11000 --terrain_seed 11000 \
      > "$ROOT/ppo_navrl-best-evaluation.stdout.log" 2>&1
  fi
fi

# Generate figures & comparison summary.
eval_args=()
for arm in depth_only v4_kinematic ppo_navrl; do
  if [[ -f "$ROOT/${arm}-best-evaluation.json" ]]; then
    eval_args+=(--eval "${arm}=$ROOT/${arm}-best-evaluation.json")
  fi
done
noise_args=()
for arm in v4_kinematic; do
  if [[ -f "$ROOT/${arm}-best-evaluation.json" ]]; then
    noise_args+=(--noise "${arm}:0.0=$ROOT/${arm}-best-evaluation.json")
    for ns in $NOISE_SCALES; do
      if [[ -f "$ROOT/${arm}-noise${ns}-evaluation.json" ]]; then
        noise_args+=(--noise "${arm}:${ns}=$ROOT/${arm}-noise${ns}-evaluation.json")
      fi
    done
  fi
done
trace_arg=()
if [[ -f "$ROOT/v4_kinematic-traces.json" ]]; then
  trace_arg=(--trace "$ROOT/v4_kinematic-traces.json")
fi
.venv/bin/python scripts/analyze_v4_behaviors.py \
  --out_dir "$ROOT/figures" "${eval_args[@]}" "${noise_args[@]}" "${trace_arg[@]}"

date -Is > "$ROOT/finished_at.txt"
echo "MATRIX_COMPLETED at=$(date -Is)" | tee -a "$ROOT/driver.log"
