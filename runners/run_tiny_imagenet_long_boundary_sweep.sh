#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Tiny-ImageNet / TinyResNet-34 long boundary sweep
# =============================================================================
# Purpose:
#   Find a freeze-safe boundary first (Gate 1), then test shallow TC cache
#   candidates (Gate 2) using longer training.
#
# Important:
#   Tiny-ImageNet should NOT use target=92.1. That was CIFAR-10's target.
#   For Tiny-ImageNet/TinyResNet34-width16, the seed42 baseline final was ~57.3.
#   So the default target is 55.0, i.e., a realistic target-preservation threshold.
#
# Default:
#   EPOCHS=300
#   TARGET_ACC=55.0
#   SEED=42
#
# Runs:
#   baseline_long
#   freeze_stem
#   freeze_layer1
#   freeze_l20
#   freeze_l21
#   freeze_layer2
#   tc_stem_start100
#   tc_l1_start100
#   tc_l20_start100
#
# Notes:
#   - freeze_l21 and freeze_layer2 are included as deep-freeze stress tests.
#   - TC is only tested at safer shallow/early boundaries.
#   - TC script must already have dtype fix:
#       out = model.forward_suffix(feat.detach().float(), args.boundary)
#
# Usage:
#   cd /mnt/data2
#   nohup env EPOCHS=300 TARGET_ACC=55.0 SEED=42 DATA_ROOT=/mnt/data2/data/tiny-imagenet-200 \
#     bash /mnt/data2/run_tiny_imagenet_long_boundary_sweep.sh \
#     > /mnt/data2/tiny_imagenet_long_boundary_sweep_master.log 2>&1 &
# =============================================================================

SEED=${SEED:-42}
GPUS_CSV=${GPUS_CSV:-0,1,2,3}
EPOCHS=${EPOCHS:-300}
TARGET_ACC=${TARGET_ACC:-55.0}
DATA_ROOT=${DATA_ROOT:-/mnt/data2/data/tiny-imagenet-200}
BASE=${BASE:-/mnt/data2/tc_resicache_tiny_imagenet_long_sweep_seed${SEED}}
WIDTH=${WIDTH:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-5}

TRAIN_SCRIPT=${TRAIN_SCRIPT:-/mnt/data2/tiny_imagenet_resnet34_tc_resicache.py}
MONITOR=${MONITOR:-/mnt/data2/monitor_command_tiny.py}

mkdir -p "$BASE/results" "$BASE/logs" "$BASE/metrics" "$BASE/summary" "$BASE/failures" "$BASE/partial_backups"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

echo "[config] BASE=$BASE"
echo "[config] DATA_ROOT=$DATA_ROOT"
echo "[config] SEED=$SEED EPOCHS=$EPOCHS TARGET_ACC=$TARGET_ACC WIDTH=$WIDTH BATCH=$BATCH_SIZE"
echo "[config] GPUS=${GPUS[*]}"
echo "[config] TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "[config] MONITOR=$MONITOR"

for f in "$TRAIN_SCRIPT" "$MONITOR"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing required file: $f" >&2
    exit 1
  fi
done

if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/val" ]]; then
  echo "[error] Tiny-ImageNet folder must contain train and val: $DATA_ROOT" >&2
  exit 1
fi

python3 -m py_compile "$TRAIN_SCRIPT" || exit 1
python3 -m py_compile "$MONITOR" || exit 1

if ! grep -q "forward_suffix(feat.detach().float()" "$TRAIN_SCRIPT"; then
  echo "[warning] dtype fix not detected in $TRAIN_SCRIPT"
  echo "[warning] Expected: forward_suffix(feat.detach().float(), args.boundary)"
  echo "[warning] TC cache variants may crash if bank_dtype=fp16."
fi

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

backup_partial_if_needed() {
  local out="$1"
  if [[ -s "$out" ]] && ! is_complete "$out"; then
    local last_epoch stamp backup
    last_epoch=$(json_last_epoch "$out")
    stamp=$(date +%Y%m%d_%H%M%S)
    backup="$BASE/partial_backups/$(basename "$out").partial_${last_epoch}_${stamp}"
    echo "[backup] $out -> $backup"
    mv "$out" "$backup"
  fi
}

variant_cmd() {
  local variant="$1"
  local out="$2"
  case "$variant" in
    baseline_long)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode none --output_json $out"
      ;;
    freeze_stem)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode freeze --boundary stem --force_freeze_epoch 55 --output_json $out"
      ;;
    freeze_layer1)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode freeze --boundary layer1 --force_freeze_epoch 55 --output_json $out"
      ;;
    freeze_l20)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode freeze --boundary layer2.0 --force_freeze_epoch 55 --output_json $out"
      ;;
    freeze_l21)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode freeze --boundary layer2.1 --force_freeze_epoch 55 --output_json $out"
      ;;
    freeze_layer2)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode freeze --boundary layer2 --force_freeze_epoch 55 --output_json $out"
      ;;
    tc_stem_start100)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode cache --boundary stem --force_freeze_epoch 55 --cache_start_epoch 100 --output_json $out"
      ;;
    tc_l1_start100)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode cache --boundary layer1 --force_freeze_epoch 55 --cache_start_epoch 100 --output_json $out"
      ;;
    tc_l20_start100)
      echo "python3 $TRAIN_SCRIPT --data_root $DATA_ROOT --seed $SEED --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers 4 --target_acc $TARGET_ACC --width $WIDTH --tc_mode cache --boundary layer2.0 --force_freeze_epoch 55 --cache_start_epoch 100 --output_json $out"
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
  local out="$BASE/results/${variant}_seed${SEED}.jsonl"
  local log="$BASE/logs/${variant}_seed${SEED}.log"
  local failmark="$BASE/failures/${variant}_seed${SEED}.failed"

  if is_complete "$out"; then
    echo "[skip] completed $variant"
    return 0
  fi

  backup_partial_if_needed "$out"
  rm -f "$failmark"

  local cmd
  cmd=$(variant_cmd "$variant" "$out") || return 1

  echo "[launch] gpu=$gpu variant=$variant"
  echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu python3 $MONITOR --name ${variant}_seed${SEED} --out-dir $BASE/metrics --gpu-index $gpu -- $cmd" > "$log"

  CUDA_VISIBLE_DEVICES="$gpu" python3 "$MONITOR" \
    --name "${variant}_seed${SEED}" \
    --out-dir "$BASE/metrics" \
    --gpu-index "$gpu" \
    --interval "$MONITOR_INTERVAL" \
    -- $cmd >> "$log" 2>&1

  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[fail] $variant rc=$rc log=$log" | tee "$failmark"
    return $rc
  fi

  if is_complete "$out"; then
    echo "[done] $variant"
  else
    echo "[warn] incomplete $variant output=$out" | tee "$failmark"
  fi
}

# Wave 1: baseline + freeze-only gate-1 boundaries.
WAVE1=(baseline_long freeze_stem freeze_layer1 freeze_l20)
# Wave 2: deeper freeze stress + shallow TC candidates.
WAVE2=(freeze_l21 freeze_layer2 tc_stem_start100 tc_l1_start100)
# Wave 3: early layer2 cache candidate.
WAVE3=(tc_l20_start100)

run_wave() {
  local -n arr=$1
  echo "[wave] ${arr[*]}"
  for i in "${!arr[@]}"; do
    local gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    run_job "$gpu" "${arr[$i]}" &
  done
  wait
  echo "[wave] completed ${arr[*]}"
}

run_wave WAVE1
run_wave WAVE2
run_wave WAVE3

echo "[summary] collecting"

python3 - "$BASE/results" "$BASE/metrics" "$BASE/summary/summary_tiny_imagenet_long_sweep_seed${SEED}.csv" "$TARGET_ACC" "$EPOCHS" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

results=Path(sys.argv[1]); metrics=Path(sys.argv[2]); out_csv=Path(sys.argv[3]); target=float(sys.argv[4]); epochs=int(sys.argv[5])
expected=epochs-1

def sf(x, default=None):
    try:
        if x is None: return default
        v=float(x)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except Exception: return default

def read_jsonl(p):
    recs=[]
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line=line.strip()
            if line:
                try: recs.append(json.loads(line))
                except Exception: pass
    return sorted(recs, key=lambda r:r.get("epoch",-1))

def max_stat(recs,k,d=0):
    vals=[sf((r.get("cache_stats") or {}).get(k), None) for r in recs]
    vals=[v for v in vals if v is not None]
    return max(vals) if vals else d

def timing(row,k):
    return sf(((row.get("timing") or {}).get(k) or {}).get("total_sec"), 0.0)

def metric(name):
    p=metrics/f"{name}_metrics_summary.json"
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}

rows=[]
for p in sorted(results.glob("*.jsonl")):
    recs=read_jsonl(p)
    if not recs: continue
    base=p.name.replace(".jsonl","")
    variant, seed = base.rsplit("_seed",1)
    final=recs[-1]; best=max(recs, key=lambda r: sf(r.get("test_acc"), -1))
    tta=None; tta_epoch=None
    for r in recs:
        if sf(r.get("test_acc"), -1) >= target:
            tta=sf(r.get("time_to_target_sec"), sf(r.get("elapsed_total_sec"), None)); tta_epoch=r.get("epoch"); break
    m=metric(base); cfg=final.get("config") or {}; total=sf(final.get("elapsed_total_sec"), "")
    rows.append({
        "variant":variant, "seed":int(seed), "complete":max(r.get("epoch",-1) for r in recs)==expected,
        "target_acc":target,
        "best_acc":sf(best.get("test_acc"),""), "best_epoch":best.get("epoch",""),
        "final_acc":sf(final.get("test_acc"),""), "reached_target":tta is not None,
        "final_ge_target": sf(final.get("test_acc"), -1) >= target,
        "tta_epoch":tta_epoch if tta_epoch is not None else "", "tta_sec":tta if tta is not None else "",
        "total_time_sec":total, "avg_epoch_sec":total/len(recs) if isinstance(total,float) else "",
        "corrected_reuses":int(max_stat(recs,"corrected_reuses",0) or 0),
        "memory_hits":int(max_stat(recs,"memory_hits",0) or 0), "disk_hits":int(max_stat(recs,"disk_hits",0) or 0),
        "probe_error":max_stat(recs,"tcrc_mean_probe_error",0), "bank_gb":max_stat(recs,"bank_allocated_gb",0),
        "transform_sec":timing(final,"tcrc_get_transform_batch"), "populate_sec":timing(final,"tcrc_canonical_populate"),
        "energy_wh_est":m.get("energy_wh_est",""), "peak_gpu_mem_mb":m.get("peak_gpu_mem_mb",""),
        "peak_gpu_util_pct":m.get("peak_gpu_util_pct",""), "proc_read_bytes":m.get("final_read_bytes",""),
        "proc_write_bytes":m.get("final_write_bytes",""), "tc_mode":cfg.get("tc_mode",""), "boundary":cfg.get("boundary",""),
    })

out_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    keys=list(rows[0].keys())
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

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
echo "Summary: $BASE/summary/summary_tiny_imagenet_long_sweep_seed${SEED}.csv"
