#!/usr/bin/env bash
set -uo pipefail

SCRIPT=${SCRIPT:-/mnt/data2/freezeout_resnet50_cifar_exact_mechanism.py}
MONITOR=${MONITOR:-/mnt/data2/monitor_command.py}
DATA_ROOT=${DATA_ROOT:-/mnt/data2/data}
BASE=${BASE:-/mnt/data2/freezeout_resnet50_cifar_exact_compare}
GPUS_CSV=${GPUS_CSV:-0,1,2,3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
AMP_FLAG=${AMP_FLAG:-}

mkdir -p "$BASE/results" "$BASE/logs" "$BASE/metrics" "$BASE/summary" "$BASE/failures"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

for f in "$SCRIPT" "$MONITOR"; do
  if [[ ! -f "$f" ]]; then echo "[error] missing required file: $f" >&2; exit 1; fi
done

python3 -m py_compile "$SCRIPT" || exit 1
python3 -m py_compile "$MONITOR" || exit 1

target_for_dataset() {
  if [[ "$1" == "cifar10" ]]; then echo "96.10"; else echo "81.68"; fi
}

cmd_for() {
  local dataset="$1"; local variant="$2"; local out="$3"; local target
  target=$(target_for_dataset "$dataset")
  local common="python3 $SCRIPT --dataset $dataset --data_root $DATA_ROOT --output_json $out --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --seed $SEED --target_acc $target --weights default --lr 0.001 $AMP_FLAG"
  case "$variant" in
    cubic_scaled_t08) echo "$common --schedule cubic --t0 0.8 --scale_lr" ;;
    linear_unscaled_t05) echo "$common --schedule linear --t0 0.5 --no_scale_lr" ;;
    *) echo "[error] unknown variant $variant" >&2; return 1 ;;
  esac
}

run_job() {
  local gpu="$1"; local dataset="$2"; local variant="$3"
  local name="${dataset}_freezeout_${variant}_seed${SEED}"
  local out="$BASE/results/${name}.jsonl"
  local log="$BASE/logs/${name}.log"
  local fail="$BASE/failures/${name}.failed"
  rm -f "$fail"
  local cmd; cmd=$(cmd_for "$dataset" "$variant" "$out") || return 1
  echo "[launch] gpu=$gpu $name"
  echo "[cmd] CUDA_VISIBLE_DEVICES=$gpu python3 $MONITOR --name $name --out-dir $BASE/metrics --gpu-index $gpu -- $cmd" > "$log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 "$MONITOR" --name "$name" --out-dir "$BASE/metrics" --gpu-index "$gpu" --interval 5 -- $cmd >> "$log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "[fail] $name rc=$rc log=$log" | tee "$fail"; return $rc; fi
  echo "[done] $name"
}

JOBS=("cifar10:cubic_scaled_t08" "cifar10:linear_unscaled_t05" "cifar100:cubic_scaled_t08" "cifar100:linear_unscaled_t05")
for i in "${!JOBS[@]}"; do
  spec="${JOBS[$i]}"; dataset="${spec%%:*}"; variant="${spec#*:}"; gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_job "$gpu" "$dataset" "$variant" &
done
wait

python3 - "$BASE/results" "$BASE/metrics" "$BASE/summary/summary_freezeout_resnet50_cifar_exact_compare.csv" "$EPOCHS" <<'PY'
import csv, json, math, sys
from pathlib import Path
results=Path(sys.argv[1]); metrics=Path(sys.argv[2]); out_csv=Path(sys.argv[3]); epochs=int(sys.argv[4]); expected=epochs-1
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
def metric(name):
    p=metrics/f"{name}_metrics_summary.json"
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}
rows=[]
for p in sorted(results.glob("*.jsonl")):
    recs=read_jsonl(p)
    if not recs: continue
    name=p.stem; final=recs[-1]; best=max(recs, key=lambda r: sf(r.get("test_acc"), -1)); cfg=final.get("config") or {}; fr=final.get("freezeout") or {}; m=metric(name)
    target=sf(final.get("target_acc"), None); tta=None; tta_epoch=None
    for r in recs:
        if target is not None and sf(r.get("test_acc"), -1) >= target:
            tta=sf(r.get("time_to_target_sec"), sf(r.get("elapsed_total_sec"), None)); tta_epoch=r.get("epoch"); break
    rows.append({
        "name":name, "dataset":cfg.get("dataset",""), "model":cfg.get("model",""), "method":cfg.get("method",""),
        "schedule":fr.get("schedule",""), "t0":fr.get("t0",""), "scale_lr":fr.get("scale_lr",""),
        "target_acc":target, "complete":max(r.get("epoch",-1) for r in recs)==expected,
        "best_acc":sf(best.get("test_acc"), ""), "best_epoch":best.get("epoch",""), "final_acc":sf(final.get("test_acc"), ""),
        "reached_target":tta is not None, "tta_epoch":tta_epoch if tta_epoch is not None else "", "tta_sec":tta if tta is not None else "",
        "total_time_sec":sf(final.get("elapsed_total_sec"), ""), "active_units_final":fr.get("active_units",""), "frozen_units_final":fr.get("frozen_units",""), "num_units":fr.get("num_units",""),
        "energy_wh_est":m.get("energy_wh_est",""), "peak_gpu_mem_mb":m.get("peak_gpu_mem_mb",""), "peak_gpu_util_pct":m.get("peak_gpu_util_pct",""), "final_read_bytes":m.get("final_read_bytes",""), "final_write_bytes":m.get("final_write_bytes",""),
    })
out_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[summary] wrote {out_csv}")
for r in rows: print(r)
PY

if compgen -G "$BASE/failures/*.failed" > /dev/null; then echo "[warning] failures:"; ls "$BASE/failures"/*.failed; else echo "[ok] no failure markers"; fi
echo "[done]"
echo "Summary: $BASE/summary/summary_freezeout_resnet50_cifar_exact_compare.csv"
