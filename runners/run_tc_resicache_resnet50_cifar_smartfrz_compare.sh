#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# TC-ResiCache / ResNet50 / CIFAR-10+CIFAR-100 SmartFRZ-comparison runner
# =============================================================================
#
# This runner launches paper-aligned ResNet50 fine-tuning settings:
#   CIFAR-10 target:  96.10%  (SmartFRZ Table 2 full-training ResNet50)
#   CIFAR-100 target: 81.68%  (SmartFRZ Table 2 full-training ResNet50)
#
# Modes per dataset:
#   baseline
#   freeze_layer3_e4
#   tc_layer3_e4_c5
#   tc_layer4_e4_c5
#
# Why layer3/layer4?
#   ResNet50 feature banks are large at 224x224 input:
#     layer2 roughly 40GB fp16 for 50k CIFAR samples
#     layer3 roughly 20GB fp16
#     layer4 roughly 10GB fp16
#   layer3/layer4 are the practical first choices on 32GB V100.
#
# To run:
#   cd /mnt/data2
#   nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 \
#     bash /mnt/data2/run_tc_resicache_resnet50_cifar_smartfrz_compare.sh \
#     > /mnt/data2/tc_resicache_resnet50_cifar_smartfrz_compare_master.log 2>&1 &
# =============================================================================

SCRIPT=${SCRIPT:-/mnt/data2/tc_resicache_resnet50_cifar_finetune.py}
MONITOR=${MONITOR:-/mnt/data2/monitor_command.py}
DATA_ROOT=${DATA_ROOT:-/mnt/data2/data}
BASE=${BASE:-/mnt/data2/tc_resicache_resnet50_cifar_smartfrz_compare}
GPUS_CSV=${GPUS_CSV:-0,1,2,3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
TRAIN_AUG=${TRAIN_AUG:-smartfrz}
NORM=${NORM:-imagenet}
AMP_FLAG=${AMP_FLAG:-}
GPU_CACHE_SAMPLES=${GPU_CACHE_SAMPLES:-50000}

mkdir -p "$BASE/results" "$BASE/logs" "$BASE/metrics" "$BASE/summary" "$BASE/failures"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

echo "[config] SCRIPT=$SCRIPT"
echo "[config] DATA_ROOT=$DATA_ROOT"
echo "[config] BASE=$BASE"
echo "[config] GPUS=${GPUS[*]}"
echo "[config] EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE SEED=$SEED TRAIN_AUG=$TRAIN_AUG"

for f in "$SCRIPT" "$MONITOR"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing required file: $f" >&2
    exit 1
  fi
done

python3 -m py_compile "$SCRIPT" || exit 1
python3 -m py_compile "$MONITOR" || exit 1

json_last_epoch() {
  local file="$1"
  if [[ ! -s "$file" ]]; then echo -1; return; fi
  tail -n 1 "$file" | python3 -c 'import sys,json
try:
    s=sys.stdin.read().strip()
    print(json.loads(s).get("epoch",-1) if s else -1)
except Exception:
    print(-1)' 2>/dev/null || echo -1
}

is_complete() {
  local out="$1"
  local expected_last_epoch=$((EPOCHS - 1))
  [[ -s "$out" ]] || return 1
  local last_epoch
  last_epoch=$(json_last_epoch "$out")
  [[ "$last_epoch" == "$expected_last_epoch" ]]
}

target_for_dataset() {
  if [[ "$1" == "cifar10" ]]; then echo "96.10"; else echo "81.68"; fi
}

cmd_for_variant() {
  local dataset="$1"
  local variant="$2"
  local out="$3"
  local target
  target=$(target_for_dataset "$dataset")

  local common="python3 $SCRIPT --dataset $dataset --data_root $DATA_ROOT --output_json $out --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed $SEED --target_acc $target --train_aug $TRAIN_AUG --norm $NORM --weights default --lr 0.001 --scheduler none --gpu_cache_samples $GPU_CACHE_SAMPLES $AMP_FLAG"

  case "$variant" in
    baseline)
      echo "$common --mode baseline"
      ;;
    freeze_layer3_e4)
      echo "$common --mode freeze --boundary layer3 --freeze_epoch 4 --cache_start_epoch 999999"
      ;;
    tc_layer3_e4_c5)
      echo "$common --mode tc_resicache --boundary layer3 --freeze_epoch 4 --cache_start_epoch 5 --bank_dtype fp16"
      ;;
    tc_layer4_e4_c5)
      echo "$common --mode tc_resicache --boundary layer4 --freeze_epoch 4 --cache_start_epoch 5 --bank_dtype fp16"
      ;;
    *)
      echo "[error] unknown variant $variant" >&2
      return 1
      ;;
  esac
}

run_job() {
  local gpu="$1"
  local dataset="$2"
  local variant="$3"
  local name="${dataset}_${variant}_seed${SEED}"
  local out="$BASE/results/${name}.jsonl"
  local log="$BASE/logs/${name}.log"
  local fail="$BASE/failures/${name}.failed"

  if is_complete "$out"; then
    echo "[skip] $name completed"
    return 0
  fi
  rm -f "$fail"

  local cmd
  cmd=$(cmd_for_variant "$dataset" "$variant" "$out") || return 1

  echo "[launch] gpu=$gpu $name"
  echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu python3 $MONITOR --name $name --out-dir $BASE/metrics --gpu-index $gpu -- $cmd" > "$log"

  CUDA_VISIBLE_DEVICES="$gpu" python3 "$MONITOR" \
    --name "$name" \
    --out-dir "$BASE/metrics" \
    --gpu-index "$gpu" \
    --interval 5 \
    -- $cmd >> "$log" 2>&1

  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[fail] $name rc=$rc log=$log" | tee "$fail"
    return $rc
  fi
  echo "[done] $name"
}

# Run in two waves to avoid launching 8 jobs on 4 GPUs at once.
WAVE1=("cifar10:baseline" "cifar10:freeze_layer3_e4" "cifar100:baseline" "cifar100:freeze_layer3_e4")
WAVE2=("cifar10:tc_layer3_e4_c5" "cifar10:tc_layer4_e4_c5" "cifar100:tc_layer3_e4_c5" "cifar100:tc_layer4_e4_c5")

run_wave() {
  local -n arr=$1
  echo "[wave] ${arr[*]}"
  for i in "${!arr[@]}"; do
    spec="${arr[$i]}"
    dataset="${spec%%:*}"
    variant="${spec#*:}"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    run_job "$gpu" "$dataset" "$variant" &
  done
  wait
}

run_wave WAVE1
run_wave WAVE2

echo "[summary] collecting"

python3 - "$BASE/results" "$BASE/metrics" "$BASE/summary/summary_tc_resicache_resnet50_cifar_smartfrz_compare.csv" "$EPOCHS" <<'PY'
import csv, json, math, sys
from pathlib import Path

results=Path(sys.argv[1]); metrics=Path(sys.argv[2]); out_csv=Path(sys.argv[3]); epochs=int(sys.argv[4])
expected=epochs-1

def sf(x, default=None):
    try:
        if x is None: return default
        v=float(x)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except Exception:
        return default

def read_jsonl(p):
    recs=[]
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line=line.strip()
            if line:
                try: recs.append(json.loads(line))
                except Exception: pass
    return sorted(recs, key=lambda r:r.get("epoch",-1))

def metric(name):
    p=metrics/f"{name}_metrics_summary.json"
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}

rows=[]
for p in sorted(results.glob("*.jsonl")):
    recs=read_jsonl(p)
    if not recs: continue
    name=p.stem
    final=recs[-1]
    best=max(recs, key=lambda r: sf(r.get("test_acc"), -1))
    cfg=final.get("config") or {}
    m=metric(name)
    target=sf(final.get("target_acc"), None)
    tta=None; tta_epoch=None
    for r in recs:
        if target is not None and sf(r.get("test_acc"), -1) >= target:
            tta=sf(r.get("time_to_target_sec"), sf(r.get("elapsed_total_sec"), None))
            tta_epoch=r.get("epoch")
            break
    cs=final.get("cache_stats") or {}
    rows.append({
        "name": name,
        "dataset": cfg.get("dataset",""),
        "model": cfg.get("model",""),
        "mode": cfg.get("mode",""),
        "boundary": cfg.get("boundary",""),
        "freeze_epoch": cfg.get("freeze_epoch",""),
        "cache_start_epoch": cfg.get("cache_start_epoch",""),
        "train_aug": cfg.get("train_aug",""),
        "target_acc": target,
        "complete": max(r.get("epoch",-1) for r in recs)==expected,
        "best_acc": sf(best.get("test_acc"), ""),
        "best_epoch": best.get("epoch",""),
        "final_acc": sf(final.get("test_acc"), ""),
        "reached_target": tta is not None,
        "tta_epoch": tta_epoch if tta_epoch is not None else "",
        "tta_sec": tta if tta is not None else "",
        "total_time_sec": sf(final.get("elapsed_total_sec"), ""),
        "epoch_time_last_sec": sf(final.get("time"), ""),
        "memory_hits": cs.get("memory_hits",0),
        "stored": cs.get("stored",0),
        "misses": cs.get("misses",0),
        "bank_allocated_gb": cs.get("bank_allocated_gb",0),
        "disk_hits": cs.get("disk_hits",0),
        "energy_wh_est": m.get("energy_wh_est",""),
        "peak_gpu_mem_mb": m.get("peak_gpu_mem_mb",""),
        "peak_gpu_util_pct": m.get("peak_gpu_util_pct",""),
        "final_read_bytes": m.get("final_read_bytes",""),
        "final_write_bytes": m.get("final_write_bytes",""),
    })

out_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    keys=list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

print(f"[summary] wrote {out_csv}")
for r in rows:
    print(r)
PY

if compgen -G "$BASE/failures/*.failed" > /dev/null; then
  echo "[warning] failures:"
  ls "$BASE/failures"/*.failed
else
  echo "[ok] no failure markers"
fi

echo "[done]"
echo "Results: $BASE/results"
echo "Metrics: $BASE/metrics"
echo "Summary: $BASE/summary/summary_tc_resicache_resnet50_cifar_smartfrz_compare.csv"
