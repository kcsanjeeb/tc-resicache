#!/usr/bin/env bash
set -uo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/data2/data}
BASE=${BASE:-/mnt/data2/resnet50_multiseed_strengthening}
TC_SCRIPT=${TC_SCRIPT:-/mnt/data2/tc_resicache_resnet50_cifar_finetune.py}
FO_SCRIPT=${FO_SCRIPT:-/mnt/data2/freezeout_resnet50_cifar_exact_mechanism.py}
MONITOR=${MONITOR:-/mnt/data2/monitor_command.py}

SEEDS_CSV=${SEEDS_CSV:-43,44}
GPUS_CSV=${GPUS_CSV:-0,1,2,3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP_FLAG=${AMP_FLAG:-}
GPU_CACHE_SAMPLES=${GPU_CACHE_SAMPLES:-50000}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-5}

mkdir -p "$BASE/results" "$BASE/logs" "$BASE/metrics" "$BASE/summary" "$BASE/failures"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

echo "[config] BASE=$BASE"
echo "[config] DATA_ROOT=$DATA_ROOT"
echo "[config] SEEDS=${SEEDS[*]}"
echo "[config] GPUS=${GPUS[*]}"
echo "[config] EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE"
echo "[config] TC_SCRIPT=$TC_SCRIPT"
echo "[config] FO_SCRIPT=$FO_SCRIPT"
echo "[config] MONITOR=$MONITOR"

for f in "$TC_SCRIPT" "$FO_SCRIPT" "$MONITOR"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing required file: $f" >&2
    exit 1
  fi
done

python3 -m py_compile "$TC_SCRIPT" || exit 1
python3 -m py_compile "$FO_SCRIPT" || exit 1
python3 -m py_compile "$MONITOR" || exit 1

target_for_dataset() {
  if [[ "$1" == "cifar10" ]]; then
    echo "96.10"
  else
    echo "81.68"
  fi
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

cmd_for_job() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local out="$4"
  local target
  target=$(target_for_dataset "$dataset")

  case "$method" in
    baseline)
      echo "python3 $TC_SCRIPT --dataset $dataset --data_root $DATA_ROOT --output_json $out --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed $seed --target_acc $target --train_aug smartfrz --norm imagenet --weights default --lr 0.001 --scheduler none --gpu_cache_samples $GPU_CACHE_SAMPLES --mode baseline $AMP_FLAG"
      ;;
    freezeout_cubic_scaled_t08)
      echo "python3 $FO_SCRIPT --dataset $dataset --data_root $DATA_ROOT --output_json $out --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed $seed --target_acc $target --weights default --lr 0.001 --schedule cubic --t0 0.8 --scale_lr $AMP_FLAG"
      ;;
    tc_l4_e4_c5)
      echo "python3 $TC_SCRIPT --dataset $dataset --data_root $DATA_ROOT --output_json $out --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed $seed --target_acc $target --train_aug smartfrz --norm imagenet --weights default --lr 0.001 --scheduler none --gpu_cache_samples $GPU_CACHE_SAMPLES --mode tc_resicache --boundary layer4 --freeze_epoch 4 --cache_start_epoch 5 --bank_dtype fp16 $AMP_FLAG"
      ;;
    tc_l3_e4_c5)
      echo "python3 $TC_SCRIPT --dataset $dataset --data_root $DATA_ROOT --output_json $out --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed $seed --target_acc $target --train_aug smartfrz --norm imagenet --weights default --lr 0.001 --scheduler none --gpu_cache_samples $GPU_CACHE_SAMPLES --mode tc_resicache --boundary layer3 --freeze_epoch 4 --cache_start_epoch 5 --bank_dtype fp16 $AMP_FLAG"
      ;;
    *)
      echo "[error] unknown method $method" >&2
      return 1
      ;;
  esac
}

run_job() {
  local gpu="$1"
  local dataset="$2"
  local method="$3"
  local seed="$4"

  local name="${dataset}_${method}_seed${seed}"
  local out="$BASE/results/${name}.jsonl"
  local log="$BASE/logs/${name}.log"
  local fail="$BASE/failures/${name}.failed"

  if is_complete "$out"; then
    echo "[skip] complete $name"
    return 0
  fi

  rm -f "$fail"
  local cmd
  cmd=$(cmd_for_job "$dataset" "$method" "$seed" "$out") || return 1

  echo "[launch] gpu=$gpu $name"
  echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu python3 $MONITOR --name $name --out-dir $BASE/metrics --gpu-index $gpu -- $cmd" > "$log"

  CUDA_VISIBLE_DEVICES="$gpu" python3 "$MONITOR"     --name "$name"     --out-dir "$BASE/metrics"     --gpu-index "$gpu"     --interval "$MONITOR_INTERVAL"     -- $cmd >> "$log" 2>&1

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

make_jobs_for_seed() {
  local seed="$1"
  echo "cifar10:baseline:$seed"
  echo "cifar10:freezeout_cubic_scaled_t08:$seed"
  echo "cifar10:tc_l4_e4_c5:$seed"
  echo "cifar100:baseline:$seed"
  echo "cifar100:freezeout_cubic_scaled_t08:$seed"
  echo "cifar100:tc_l3_e4_c5:$seed"
}

run_wave_from_array() {
  local -n jobs_ref=$1
  local start="$2"
  local end="$3"
  echo "[wave] jobs $start to $((end-1))"
  for ((i=start; i<end; i++)); do
    local spec="${jobs_ref[$i]}"
    local dataset method seed
    IFS=':' read -r dataset method seed <<< "$spec"
    local gpu="${GPUS[$(((i-start) % ${#GPUS[@]}))]}"
    run_job "$gpu" "$dataset" "$method" "$seed" &
  done
  wait
}

for seed in "${SEEDS[@]}"; do
  echo "[seed] $seed"
  mapfile -t JOBS < <(make_jobs_for_seed "$seed")
  run_wave_from_array JOBS 0 4
  run_wave_from_array JOBS 4 6
done

echo "[summary] collecting"

python3 - "$BASE/results" "$BASE/metrics" "$BASE/summary/summary_resnet50_multiseed_strengthening.csv" "$EPOCHS" <<'PY'
import csv, json, math, sys
from pathlib import Path

results = Path(sys.argv[1])
metrics = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
epochs = int(sys.argv[4])
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
    fr = final.get("freezeout") or {}
    m = metric(name)
    target = sf(final.get("target_acc"), None)
    tta = None
    tta_epoch = None
    for r in recs:
        if target is not None and sf(r.get("test_acc"), -1) >= target:
            tta = sf(r.get("time_to_target_sec"), sf(r.get("elapsed_total_sec"), None))
            tta_epoch = r.get("epoch")
            break

    seed = ""
    if "_seed" in name:
        seed = name.rsplit("_seed", 1)[1]

    method = cfg.get("mode", "")
    if "freezeout" in name:
        method = "freezeout_cubic_scaled_t08"
    elif method == "tc_resicache":
        method = f"tc_{cfg.get('boundary','')}_e{cfg.get('freeze_epoch','')}_c{cfg.get('cache_start_epoch','')}"
    elif method == "baseline":
        method = "baseline"

    rows.append({
        "name": name,
        "dataset": cfg.get("dataset", name.split("_")[0]),
        "seed": seed,
        "method": method,
        "model": cfg.get("model", "resnet50"),
        "target_acc": target,
        "complete": max(r.get("epoch", -1) for r in recs) == expected,
        "target_reached": tta is not None,
        "final_ge_target": sf(final.get("test_acc"), -1) >= (target if target is not None else 999),
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
        "freezeout_schedule": fr.get("schedule", ""),
        "freezeout_t0": fr.get("t0", ""),
        "freezeout_scale_lr": fr.get("scale_lr", ""),
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

grouped = {}
for r in rows:
    key = (r["dataset"], r["method"])
    grouped.setdefault(key, []).append(r)

mean_csv = out_csv.with_name("summary_resnet50_multiseed_strengthening_by_method.csv")
mean_rows = []
for (dataset, method), rs in sorted(grouped.items()):
    def mean(k):
        vals = [sf(x.get(k), None) for x in rs]
        vals = [v for v in vals if v is not None and v != ""]
        return sum(vals) / len(vals) if vals else ""
    mean_rows.append({
        "dataset": dataset,
        "method": method,
        "n": len(rs),
        "target_reached_count": sum(1 for x in rs if x.get("target_reached")),
        "final_ge_target_count": sum(1 for x in rs if x.get("final_ge_target")),
        "mean_best_acc": mean("best_acc"),
        "mean_final_acc": mean("final_acc"),
        "mean_tta_sec": mean("tta_sec"),
        "mean_total_time_sec": mean("total_time_sec"),
        "mean_energy_wh_est": mean("energy_wh_est"),
        "mean_peak_gpu_mem_mb": mean("peak_gpu_mem_mb"),
        "mean_memory_hits": mean("memory_hits"),
        "mean_disk_hits": mean("disk_hits"),
    })
if mean_rows:
    with open(mean_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mean_rows[0].keys()))
        w.writeheader()
        w.writerows(mean_rows)
print(f"[summary] wrote {mean_csv}")
for r in mean_rows:
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
echo "Per-run summary: $BASE/summary/summary_resnet50_multiseed_strengthening.csv"
echo "By-method summary: $BASE/summary/summary_resnet50_multiseed_strengthening_by_method.csv"
