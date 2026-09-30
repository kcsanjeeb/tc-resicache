#!/usr/bin/env bash
# SmartFRZ-rerun in the finetune script (same implementation as Table 1 CIFAR rows):
# fp32 (NO --amp), same script/workers/GPU as baseline/FO/TC CIFAR R50 rows.
set -uo pipefail
PY=/home/san/miniconda/envs/mac-linux/bin/python
SCRIPT=/mnt/data2/tc_resicache_resnet50_cifar_finetune_sfz.py
MONITOR=/mnt/data2/monitor_command.py
BASE=/mnt/data2/sfz_rerun_finetune
mkdir -p "$BASE/results" "$BASE/logs" "$BASE/metrics"

run_job() {
  local gpu="$1" dataset="$2" seed="$3"
  local name="sfz_${dataset}_s${seed}"
  local out="$BASE/results/${name}.jsonl"
  local log="$BASE/logs/${name}.log"
  if [[ -s "$out" ]] && tail -1 "$out" | grep -q '"epoch": 9'; then echo "[skip] $name"; return 0; fi
  echo "[launch] gpu=$gpu $name $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$MONITOR" --name "$name" --out-dir "$BASE/metrics" --gpu-index "$gpu" --interval 5 -- \
    $PY "$SCRIPT" --dataset "$dataset" --data_root /mnt/data2/data --output_json "$out" \
    --mode smartfrz --epochs 10 --batch_size 32 --num_workers 4 --seed "$seed" --target_acc 0 \
    --train_aug smartfrz --norm imagenet --weights default --lr 0.001 --scheduler none >> "$log" 2>&1
  echo "[done] $name rc=$?"
}

for seed in 42 43; do run_job 2 cifar10 "$seed"; done &
for seed in 42 43; do run_job 3 cifar100 "$seed"; done &
wait
echo "[all done]"
