#!/usr/bin/env bash
set -uo pipefail
DATA_ROOT=${DATA_ROOT:-/mnt/data2/data}
BASE=${BASE:-/mnt/data2/egeria_argument_experiments}
GPUS_CSV=${GPUS_CSV:-0,1,2,3}
SEEDS_CSV=${SEEDS_CSV:-42,43,44}
RUN_TC=${RUN_TC:-0}
FORWARD_SCRIPT=${FORWARD_SCRIPT:-/mnt/data2/run_forward_removal_control_experiment.sh}
MICRO_SCRIPT=${MICRO_SCRIPT:-/mnt/data2/microbench_prefix_vs_cache.py}

mkdir -p "$BASE/logs" "$BASE/microbench" "$BASE/summary"

if [[ -f "$FORWARD_SCRIPT" ]]; then
  echo "[step] forward-removal controls"
  env DATA_ROOT="$DATA_ROOT" GPUS_CSV="$GPUS_CSV" SEEDS_CSV="$SEEDS_CSV" RUN_TC="$RUN_TC" BASE="$BASE/forward_removal" \
    bash "$FORWARD_SCRIPT" > "$BASE/logs/forward_removal.log" 2>&1
else
  echo "[warn] missing $FORWARD_SCRIPT; skipping forward-removal controls"
fi

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
python3 -m py_compile "$MICRO_SCRIPT" || exit 1

echo "[step] microbenchmarks"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python3 "$MICRO_SCRIPT" --dataset cifar10 --data_root "$DATA_ROOT" --boundary layer4 \
  --output_json "$BASE/microbench/cifar10_layer4.json" --output_csv "$BASE/microbench/cifar10_layer4.csv" > "$BASE/logs/microbench_cifar10_layer4.log" 2>&1 &
CUDA_VISIBLE_DEVICES="${GPUS[1 % ${#GPUS[@]}]}" python3 "$MICRO_SCRIPT" --dataset cifar100 --data_root "$DATA_ROOT" --boundary layer3 \
  --output_json "$BASE/microbench/cifar100_layer3.json" --output_csv "$BASE/microbench/cifar100_layer3.csv" > "$BASE/logs/microbench_cifar100_layer3.log" 2>&1 &
wait

python3 - "$BASE/microbench" "$BASE/summary/microbench_prefix_vs_cache_summary.csv" <<'PY'
from pathlib import Path
import csv, sys
rows=[]
for p in sorted(Path(sys.argv[1]).glob("*.csv")):
    rows += list(csv.DictReader(open(p)))
out=Path(sys.argv[2]); out.parent.mkdir(parents=True,exist_ok=True)
if rows:
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[summary] wrote {out}")
for r in rows: print(r)
PY

echo "[done]"
echo "$BASE/summary/microbench_prefix_vs_cache_summary.csv"
echo "$BASE/forward_removal/summary/summary_forward_removal_control_by_method.csv"
