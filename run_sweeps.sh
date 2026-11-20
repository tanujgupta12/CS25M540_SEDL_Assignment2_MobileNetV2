#!/usr/bin/env bash
# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
# Creates the two W&B compression sweeps (Q3) and runs an agent for each, one after another.
# Needs a trained baseline at checkpoints/baseline/best.pt (python train.py ...).
# Usage: bash run_sweeps.sh [project-name]
set -euo pipefail
PROJECT=${1:-CS6886W-SEDL-Assignment2-MobileNetV2}

run_sweep () {
  local yaml=$1
  local id
  id=$(wandb sweep --project "$PROJECT" "$yaml" 2>&1 | tee /dev/stderr | sed 's/\x1b\[[0-9;]*m//g' | grep -o 'wandb agent [^ ]*' | awk '{print $3}')
  echo ">>> running sweep $id ($yaml)"
  wandb agent "$id"
}

run_sweep sweeps/ptq.yaml   # 30 runs, PTQ only, ~1 min each on a T4
run_sweep sweeps/qat.yaml   # 12 runs, 10 QAT epochs each
