#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# CIFAR-10 / ResNet-56 final TC-ResiCache hardware-metrics rerun
# =============================================================================
# Runs 3 seeds x 3 configurations:
#   1. baseline
#   2. noref_freeze_l23
#   3. tc_resicache_l23_start55
#
# Required:
#   /mnt/data2/baseline_resnet56_cifar10_jsonl.py
#   /mnt/data2/resicache_plantcrc_v6_subpixel_resnet56_cifar10.py
#   /mnt/data2/monitor_command.py
#   /mnt/data2/data/cifar-10-batches-py
# =============================================================================

SEEDS_CSV=${SEEDS_CSV:-42,43,44}
GPUS_CSV=${GPUS_CSV:-0,1,2,3}
EPOCHS=${EPOCHS:-200}
TARGET_ACC=${TARGET_ACC:-92.1}
DATA_ROOT=${DATA_ROOT:-/mnt/data2/data}
BASE=${BASE:-/mnt/data2/tc_resicache_cifar10_final_hardware_metrics}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-5}

BASELINE_SCRIPT=${BASELINE_SCRIPT:-/mnt/data2/baseline_resnet56_cifar10_jsonl.py}
TC_SCRIPT=${TC_SCRIPT:-/mnt/data2/resicache_plantcrc_v6_subpixel_resnet56_cifar10.py}
MONITOR=${MONITOR:-/mnt/data2/monitor_command.py}

mkdir -p "$BASE/results" "$BASE/logs" "$BASE/cache" "$BASE/metrics" "$BASE/summary" "$BASE/failures" "$BASE/partial_backups"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

echo "[config] BASE=$BASE"
echo "[config] DATA_ROOT=$DATA_ROOT"
echo "[config] SEEDS=${SEEDS[*]}"
echo "[config] GPUS=${GPUS[*]}"
echo "[config] TARGET_ACC=$TARGET_ACC EPOCHS=$EPOCHS"
echo "[config] BASELINE_SCRIPT=$BASELINE_SCRIPT"
echo "[config] TC_SCRIPT=$TC_SCRIPT"
echo "[config] MONITOR=$MONITOR"

for f in "$BASELINE_SCRIPT" "$TC_SCRIPT" "$MONITOR"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing required file: $f" >&2
    exit 1
  fi
done

if [[ ! -d "$DATA_ROOT/cifar-10-batches-py" ]]; then
  echo "[error] missing extracted CIFAR-10 folder: $DATA_ROOT/cifar-10-batches-py" >&2
  exit 1
fi

python3 -m py_compile "$BASELINE_SCRIPT" || {
  echo "[error] baseline script failed syntax check: $BASELINE_SCRIPT" >&2
  exit 1
}

python3 -m py_compile "$TC_SCRIPT" || {
  echo "[error] TC script failed syntax check: $TC_SCRIPT" >&2
  exit 1
}

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

COMMON_TC="--data_root $DATA_ROOT --epochs $EPOCHS --batch_size 128 --num_workers 2 --lr 0.1 --device cuda \
 --target_acc $TARGET_ACC --aug_mode random --cache_key_mode sample --gpu_cache_size 50000 --no_disk_cache \
 --hybrid_mode gradient_only --disable_reference_model \
 --bootstrap_epochs 45 --eval_interval 50 --grad_window_size 10 --grad_threshold 0.12 \
 --grad_track_interval 10 --grad_track_scope candidate \
 --min_compute_benefit 0.04 --cache_overhead_fraction 0.04 \
 --residual_momentum 0.95 --residual_clip 5.0 \
 --residual_refresh_interval 250 --residual_probe_prob 0.0 --profile_timing \
 --min_freeze_layer layer1.0 --max_freeze_layer layer2.3 \
 --tcrc_crop_rounding floor --tcrc_subpixel bilinear --tcrc_probe_error_threshold 0.0 \
 --plantcrc_bank_dtype fp16 --plantcrc_num_samples 50000 \
 --force_freeze_epoch 55"

variant_cmd() {
  local seed="$1"
  local variant="$2"
  local out="$3"
  local cache="$4"

  case "$variant" in
    baseline)
      echo "python3 $BASELINE_SCRIPT --data_root $DATA_ROOT --seed $seed --epochs $EPOCHS --batch_size 128 --num_workers 2 --lr 0.1 --target_acc $TARGET_ACC --device cuda --output_json $out"
      ;;
    noref_freeze_l23)
      echo "python3 $TC_SCRIPT $COMMON_TC --seed $seed --cache_impl delta --disable_cache_reuse --cache_start_epoch 999999 --output_json $out --cache_dir $cache"
      ;;
    tc_resicache_l23_start55)
      echo "python3 $TC_SCRIPT $COMMON_TC --seed $seed --cache_impl plantcrc_v2 --cache_start_epoch 55 --output_json $out --cache_dir $cache"
      ;;
    *)
      echo "[error] unknown variant $variant" >&2
      return 1
      ;;
  esac
}

run_job() {
  local gpu="$1"
  local seed="$2"
  local variant="$3"
  local out log cache failmark cmd rc

  out="$BASE/results/${variant}_seed${seed}.jsonl"
  log="$BASE/logs/${variant}_seed${seed}.log"
  cache="$BASE/cache/${variant}_seed${seed}"
  failmark="$BASE/failures/${variant}_seed${seed}.failed"

  if is_complete "$out"; then
    echo "[skip] completed variant=$variant seed=$seed"
    return 0
  fi

  backup_partial_if_needed "$out"
  rm -f "$failmark"

  cmd=$(variant_cmd "$seed" "$variant" "$out" "$cache") || return 1

  echo "[launch] GPU=$gpu seed=$seed variant=$variant"
  echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu python3 $MONITOR --name ${variant}_seed${seed} --out-dir $BASE/metrics --gpu-index $gpu -- $cmd" > "$log"

  CUDA_VISIBLE_DEVICES="$gpu" python3 "$MONITOR" \
    --name "${variant}_seed${seed}" \
    --out-dir "$BASE/metrics" \
    --gpu-index "$gpu" \
    --interval "$MONITOR_INTERVAL" \
    -- $cmd >> "$log" 2>&1

  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[fail] variant=$variant seed=$seed rc=$rc log=$log" | tee "$failmark"
    return $rc
  fi

  if is_complete "$out"; then
    echo "[done] variant=$variant seed=$seed"
  else
    echo "[warn] output incomplete variant=$variant seed=$seed out=$out" | tee "$failmark"
  fi
}

JOBS=()
for seed in "${SEEDS[@]}"; do
  JOBS+=("${seed}:baseline")
  JOBS+=("${seed}:noref_freeze_l23")
  JOBS+=("${seed}:tc_resicache_l23_start55")
done

run_wave() {
  local start="$1"
  local end="$2"
  local idx=0
  echo "[wave] jobs $start to $end"
  for ((j=start; j<=end && j<${#JOBS[@]}; j++)); do
    local job="${JOBS[$j]}"
    local seed="${job%%:*}"
    local variant="${job#*:}"
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    run_job "$gpu" "$seed" "$variant" &
    idx=$((idx+1))
  done
  wait
  echo "[wave] completed jobs $start to $end"
}

run_wave 0 3
run_wave 4 7
run_wave 8 8

echo "[summary] collecting results"

python3 - "$BASE/results" "$BASE/metrics" "$BASE/summary/summary_cifar10_final_hardware_metrics.csv" "$TARGET_ACC" "$EPOCHS" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

results = Path(sys.argv[1])
metrics = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
target = float(sys.argv[4])
epochs = int(sys.argv[5])
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

def read_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line=line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return sorted(recs, key=lambda r: r.get("epoch", -1))

def max_stat(recs, key, default=0):
    vals=[]
    for r in recs:
        v=sf((r.get("cache_stats") or {}).get(key), None)
        if v is not None:
            vals.append(v)
    return max(vals) if vals else default

def timing(row, key):
    return sf(((row.get("timing") or {}).get(key) or {}).get("total_sec"), 0.0)

def read_metric_summary(name):
    p = metrics / f"{name}_metrics_summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def parse_file_name(name):
    base = name.replace(".jsonl", "")
    variant, seed = base.rsplit("_seed", 1)
    return variant, int(seed)

rows=[]
for p in sorted(results.glob("*.jsonl")):
    recs=read_jsonl(p)
    if not recs:
        continue
    variant, seed = parse_file_name(p.name)
    final=recs[-1]
    best=max(recs, key=lambda r: sf(r.get("test_acc"), -1))
    tta=None
    tta_epoch=None
    for r in recs:
        if sf(r.get("test_acc"), -1) >= target:
            tta=sf(r.get("time_to_target_sec"), sf(r.get("elapsed_total_sec"), None))
            tta_epoch=r.get("epoch")
            break
    m=read_metric_summary(f"{variant}_seed{seed}")
    total=sf(final.get("elapsed_total_sec"), "")
    cfg=final.get("config") or {}
    rows.append({
        "variant": variant,
        "seed": seed,
        "file": p.name,
        "complete": max(r.get("epoch",-1) for r in recs) == expected,
        "best_acc": sf(best.get("test_acc"), ""),
        "best_epoch": best.get("epoch", ""),
        "final_acc": sf(final.get("test_acc"), ""),
        "reached_target": tta is not None,
        "tta_epoch": tta_epoch if tta_epoch is not None else "",
        "tta_sec": tta if tta is not None else "",
        "total_time_sec": total,
        "avg_epoch_sec": total / len(recs) if isinstance(total, float) and recs else "",
        "corrected_reuses": int(max_stat(recs, "corrected_reuses", 0) or 0),
        "memory_hits": int(max_stat(recs, "memory_hits", 0) or 0),
        "disk_hits": int(max_stat(recs, "disk_hits", 0) or 0),
        "probe_error": max_stat(recs, "tcrc_mean_probe_error", 0),
        "transform_sec": timing(final, "tcrc_get_transform_batch"),
        "forward_cached_sec": timing(final, "forward_from_cached_layer"),
        "full_forward_sec": timing(final, "full_forward_no_cache"),
        "backward_sec": timing(final, "backward"),
        "energy_wh_est": m.get("energy_wh_est", ""),
        "peak_gpu_mem_mb": m.get("peak_gpu_mem_mb", ""),
        "peak_gpu_util_pct": m.get("peak_gpu_util_pct", ""),
        "peak_rss_mb": m.get("peak_process_tree_rss_mb", ""),
        "proc_read_bytes": m.get("final_read_bytes", ""),
        "proc_write_bytes": m.get("final_write_bytes", ""),
        "script": cfg.get("script", ""),
        "max_freeze_layer": cfg.get("max_freeze_layer", ""),
        "cache_impl": cfg.get("cache_impl", ""),
        "cache_start_epoch": cfg.get("cache_start_epoch", ""),
        "disable_cache_reuse": cfg.get("disable_cache_reuse", ""),
    })

out_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    keys=list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

agg_path=out_csv.with_name("summary_cifar10_final_hardware_metrics_by_variant.csv")
groups={}
for r in rows:
    groups.setdefault(r["variant"], []).append(r)

agg=[]
for variant, rs in sorted(groups.items()):
    def vals(k):
        v=[sf(r.get(k), None) for r in rs]
        return [x for x in v if x is not None]
    def mean(k):
        v=vals(k)
        return statistics.mean(v) if v else ""
    def sd(k):
        v=vals(k)
        return statistics.stdev(v) if len(v)>=2 else (0 if len(v)==1 else "")
    agg.append({
        "variant": variant,
        "n": len(rs),
        "seeds": ",".join(str(r["seed"]) for r in sorted(rs, key=lambda x:x["seed"])),
        "target_success_count": sum(1 for r in rs if r["reached_target"]),
        "final_ge_target_count": sum(1 for r in rs if sf(r["final_acc"], -1) >= target),
        "mean_best_acc": mean("best_acc"),
        "std_best_acc": sd("best_acc"),
        "mean_final_acc": mean("final_acc"),
        "std_final_acc": sd("final_acc"),
        "mean_tta_sec": mean("tta_sec"),
        "std_tta_sec": sd("tta_sec"),
        "mean_total_time_sec": mean("total_time_sec"),
        "std_total_time_sec": sd("total_time_sec"),
        "mean_avg_epoch_sec": mean("avg_epoch_sec"),
        "mean_corrected_reuses": mean("corrected_reuses"),
        "mean_memory_hits": mean("memory_hits"),
        "mean_disk_hits": mean("disk_hits"),
        "mean_probe_error": mean("probe_error"),
        "mean_transform_sec": mean("transform_sec"),
        "mean_energy_wh_est": mean("energy_wh_est"),
        "std_energy_wh_est": sd("energy_wh_est"),
        "mean_peak_gpu_mem_mb": mean("peak_gpu_mem_mb"),
        "std_peak_gpu_mem_mb": sd("peak_gpu_mem_mb"),
        "mean_peak_gpu_util_pct": mean("peak_gpu_util_pct"),
        "mean_proc_read_bytes": mean("proc_read_bytes"),
        "mean_proc_write_bytes": mean("proc_write_bytes"),
    })

if agg:
    keys=list(agg[0].keys())
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(agg)

print(f"[summary] wrote {out_csv}")
print(f"[summary] wrote {agg_path}")
for r in rows:
    print(
        f"{r['variant']} seed={r['seed']} best={r['best_acc']} final={r['final_acc']} "
        f"target={r['reached_target']} tta={r['tta_sec']} total={r['total_time_sec']} "
        f"energyWh={r['energy_wh_est']} memMB={r['peak_gpu_mem_mb']} "
        f"reuses={r['corrected_reuses']} disk={r['disk_hits']} probe={r['probe_error']}"
    )
PY

if compgen -G "$BASE/failures/*.failed" > /dev/null; then
  echo "[warning] failures detected:"
  ls "$BASE/failures"/*.failed
else
  echo "[ok] no failure markers"
fi

echo "[done]"
echo "Results: $BASE/results"
echo "Metrics: $BASE/metrics"
echo "Per-run summary: $BASE/summary/summary_cifar10_final_hardware_metrics.csv"
echo "By-variant summary: $BASE/summary/summary_cifar10_final_hardware_metrics_by_variant.csv"
