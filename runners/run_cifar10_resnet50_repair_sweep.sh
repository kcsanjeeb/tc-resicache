#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# CIFAR-10 / ResNet50 TC-ResiCache repair sweep
# =============================================================================
#
# Motivation:
#   Previous 3-seed ResNet50 comparison showed:
#     CIFAR-10 TC L4 e4/c5:
#       target reached = 2/3
#       seed44 missed 96.10 target
#       total time and energy were excellent
#
# Goal:
#   Repair CIFAR-10 target robustness while preserving most of the
#   total-runtime and energy savings.
#
# First-stage sweep:
#   seed44 only:
#     tc_l4_e5_c6
#     tc_l4_e6_c7
#     tc_l3_e4_c5
#     tc_l3_e5_c6
#
# If one candidate reaches target with good cost, later run seeds 42/43/44
# for only the selected candidate.
#
# Required:
#   /mnt/data2/tc_resicache_resnet50_cifar_finetune.py
#   /mnt/data2/monitor_command.py
# =============================================================================

DATA_ROOT=${DATA_ROOT:-/mnt/data2/data}
BASE=${BASE:-/mnt/data2/cifar10_resnet50_repair_sweep_seed44}
TC_SCRIPT=${TC_SCRIPT:-/mnt/data2/tc_resicache_resnet50_cifar_finetune.py}
MONITOR=${MONITOR:-/mnt/data2/monitor_command.py}

SEED=${SEED:-44}
GPUS_CSV=${GPUS_CSV:-0,1,2,3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
TARGET_ACC=${TARGET_ACC:-96.10}
GPU_CACHE_SAMPLES=${GPU_CACHE_SAMPLES:-50000}
AMP_FLAG=${AMP_FLAG:-}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-5}

mkdir -p "$BASE/results" "$BASE/logs" "$BASE/metrics" "$BASE/summary" "$BASE/failures"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

echo "[config] BASE=$BASE"
echo "[config] DATA_ROOT=$DATA_ROOT"
echo "[config] SEED=$SEED TARGET_ACC=$TARGET_ACC EPOCHS=$EPOCHS"
echo "[config] GPUS=${GPUS[*]}"
echo "[config] TC_SCRIPT=$TC_SCRIPT"
echo "[config] MONITOR=$MONITOR"

for f in "$TC_SCRIPT" "$MONITOR"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing required file: $f" >&2
    exit 1
  fi
done

python3 -m py_compile "$TC_SCRIPT" || exit 1
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

cmd_for_variant() {
  local variant="$1"
  local out="$2"

  local common="python3 $TC_SCRIPT --dataset cifar10 --data_root $DATA_ROOT --output_json $out \
    --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
    --seed $SEED --target_acc $TARGET_ACC --train_aug smartfrz --norm imagenet \
    --weights default --lr 0.001 --scheduler none --gpu_cache_samples $GPU_CACHE_SAMPLES \
    --mode tc_resicache --bank_dtype fp16 $AMP_FLAG"

  case "$variant" in
    tc_l4_e5_c6)
      echo "$common --boundary layer4 --freeze_epoch 5 --cache_start_epoch 6"
      ;;
    tc_l4_e6_c7)
      echo "$common --boundary layer4 --freeze_epoch 6 --cache_start_epoch 7"
      ;;
    tc_l3_e4_c5)
      echo "$common --boundary layer3 --freeze_epoch 4 --cache_start_epoch 5"
      ;;
    tc_l3_e5_c6)
      echo "$common --boundary layer3 --freeze_epoch 5 --cache_start_epoch 6"
      ;;
    freeze_l4_e5)
      echo "python3 $TC_SCRIPT --dataset cifar10 --data_root $DATA_ROOT --output_json $out \
        --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
        --seed $SEED --target_acc $TARGET_ACC --train_aug smartfrz --norm imagenet \
        --weights default --lr 0.001 --scheduler none --gpu_cache_samples $GPU_CACHE_SAMPLES \
        --mode freeze --boundary layer4 --freeze_epoch 5 --cache_start_epoch 999999 $AMP_FLAG"
      ;;
    freeze_l3_e4)
      echo "python3 $TC_SCRIPT --dataset cifar10 --data_root $DATA_ROOT --output_json $out \
        --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
        --seed $SEED --target_acc $TARGET_ACC --train_aug smartfrz --norm imagenet \
        --weights default --lr 0.001 --scheduler none --gpu_cache_samples $GPU_CACHE_SAMPLES \
        --mode freeze --boundary layer3 --freeze_epoch 4 --cache_start_epoch 999999 $AMP_FLAG"
      ;;
    *)
      echo "[error] unknown variant $variant" >&2
      return 1
      ;;
  esac
}

run_job() {
  local gpu="$1"
  local variant="$2"
  local name="cifar10_${variant}_seed${SEED}"
  local out="$BASE/results/${name}.jsonl"
  local log="$BASE/logs/${name}.log"
  local fail="$BASE/failures/${name}.failed"

  if is_complete "$out"; then
    echo "[skip] complete $name"
    return 0
  fi

  rm -f "$fail"
  local cmd
  cmd=$(cmd_for_variant "$variant" "$out") || return 1

  echo "[launch] gpu=$gpu $name"
  echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu python3 $MONITOR --name $name --out-dir $BASE/metrics --gpu-index $gpu -- $cmd" > "$log"

  CUDA_VISIBLE_DEVICES="$gpu" python3 "$MONITOR" \
    --name "$name" \
    --out-dir "$BASE/metrics" \
    --gpu-index "$gpu" \
    --interval "$MONITOR_INTERVAL" \
    -- $cmd >> "$log" 2>&1

  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[fail] $name rc=$rc log=$log" | tee "$fail"
    return $rc
  fi

  if is_complete "$out"; then
    echo "[done] $name"
  else
    echo "[warn] incomplete $name output=$out" | tee "$fail"
  fi
}

# Main repair sweep. Controls are optional.
VARIANTS=(tc_l4_e5_c6 tc_l4_e6_c7 tc_l3_e4_c5 tc_l3_e5_c6)

if [[ "${RUN_FREEZE_CONTROLS:-0}" == "1" ]]; then
  VARIANTS+=(freeze_l4_e5 freeze_l3_e4)
fi

echo "[variants] ${VARIANTS[*]}"

for i in "${!VARIANTS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_job "$gpu" "${VARIANTS[$i]}" &
  # Run only as many concurrent jobs as GPUs.
  if (( (i + 1) % ${#GPUS[@]} == 0 )); then
    wait
  fi
done
wait

echo "[summary] collecting"

python3 - "$BASE/results" "$BASE/metrics" "$BASE/summary/summary_cifar10_resnet50_repair_sweep_seed${SEED}.csv" "$EPOCHS" "$TARGET_ACC" <<'PY'
import csv, json, math, sys
from pathlib import Path

results = Path(sys.argv[1])
metrics = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
epochs = int(sys.argv[4])
target_default = float(sys.argv[5])
expected = epochs - 1

def sf(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def read_jsonl(p):
    recs = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return sorted(recs, key=lambda r: r.get("epoch", -1))

def metric(name):
    p = metrics / f"{name}_metrics_summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

rows = []
for p in sorted(results.glob("*.jsonl")):
    recs = read_jsonl(p)
    if not recs:
        continue
    name = p.stem
    final = recs[-1]
    best = max(recs, key=lambda r: sf(r.get("test_acc"), -1))
    cfg = final.get("config") or {}
    cs = final.get("cache_stats") or {}
    m = metric(name)
    target = sf(final.get("target_acc"), target_default)
    tta = None
    tta_epoch = None
    for r in recs:
        if sf(r.get("test_acc"), -1) >= target:
            tta = sf(r.get("time_to_target_sec"), sf(r.get("elapsed_total_sec"), None))
            tta_epoch = r.get("epoch")
            break

    method = cfg.get("mode", "")
    if method == "tc_resicache":
        method = f"tc_{cfg.get('boundary','')}_e{cfg.get('freeze_epoch','')}_c{cfg.get('cache_start_epoch','')}"
    elif method == "freeze":
        method = f"freeze_{cfg.get('boundary','')}_e{cfg.get('freeze_epoch','')}"
    else:
        method = name

    rows.append({
        "name": name,
        "method": method,
        "dataset": cfg.get("dataset", "cifar10"),
        "seed": cfg.get("seed", ""),
        "target_acc": target,
        "complete": max(r.get("epoch", -1) for r in recs) == expected,
        "target_reached": tta is not None,
        "final_ge_target": sf(final.get("test_acc"), -1) >= target,
        "best_acc": sf(best.get("test_acc"), ""),
        "best_epoch": best.get("epoch", ""),
        "final_acc": sf(final.get("test_acc"), ""),
        "tta_epoch": tta_epoch if tta_epoch is not None else "",
        "tta_sec": tta if tta is not None else "",
        "total_time_sec": sf(final.get("elapsed_total_sec"), ""),
        "epoch_time_last_sec": sf(final.get("time"), ""),
        "energy_wh_est": m.get("energy_wh_est", ""),
        "peak_gpu_mem_mb": m.get("peak_gpu_mem_mb", ""),
        "peak_gpu_util_pct": m.get("peak_gpu_util_pct", ""),
        "final_read_bytes": m.get("final_read_bytes", ""),
        "final_write_bytes": m.get("final_write_bytes", ""),
        "memory_hits": cs.get("memory_hits", 0),
        "stored": cs.get("stored", 0),
        "misses": cs.get("misses", 0),
        "disk_hits": cs.get("disk_hits", 0),
        "bank_allocated_gb": cs.get("bank_allocated_gb", 0),
    })

out_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
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
echo "Summary: $BASE/summary/summary_cifar10_resnet50_repair_sweep_seed${SEED}.csv"
