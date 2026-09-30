# Egeria-argument experiments

This package does **not** reproduce Egeria. It produces stronger TC-ResiCache evidence for the argument that our method removes frozen-prefix forward computation, not only backward computation.

## Experiments

1. Forward-removal control:
   - CIFAR-10 / ResNet50 freeze-only L4 e5
   - CIFAR-100 / ResNet50 freeze-only L3 e4
   - optional TC reruns
2. Prefix-vs-cache microbenchmark:
   - measures ResNet50 prefix forward time
   - measures GPU activation-bank lookup/copy time
   - reports lookup speedup over recomputing the frozen prefix

## Copy

```bash
cp microbench_prefix_vs_cache.py /mnt/data2/
cp run_egeria_argument_experiments.sh /mnt/data2/
chmod +x /mnt/data2/microbench_prefix_vs_cache.py
chmod +x /mnt/data2/run_egeria_argument_experiments.sh
```

Also ensure this exists:

```bash
ls /mnt/data2/run_forward_removal_control_experiment.sh
```

## Run full suite

```bash
cd /mnt/data2

nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 SEEDS_CSV=42,43,44 RUN_TC=0 \
  bash /mnt/data2/run_egeria_argument_experiments.sh \
  > /mnt/data2/egeria_argument_experiments_master.log 2>&1 &
```

## Run only the microbenchmarks

```bash
python3 /mnt/data2/microbench_prefix_vs_cache.py \
  --dataset cifar10 --data_root /mnt/data2/data --boundary layer4 \
  --output_json /mnt/data2/egeria_argument_experiments/microbench/cifar10_layer4.json \
  --output_csv /mnt/data2/egeria_argument_experiments/microbench/cifar10_layer4.csv

python3 /mnt/data2/microbench_prefix_vs_cache.py \
  --dataset cifar100 --data_root /mnt/data2/data --boundary layer3 \
  --output_json /mnt/data2/egeria_argument_experiments/microbench/cifar100_layer3.json \
  --output_csv /mnt/data2/egeria_argument_experiments/microbench/cifar100_layer3.csv
```

## Upload after completion

```text
/mnt/data2/egeria_argument_experiments/summary/microbench_prefix_vs_cache_summary.csv
/mnt/data2/egeria_argument_experiments/forward_removal/summary/summary_forward_removal_control.csv
/mnt/data2/egeria_argument_experiments/forward_removal/summary/summary_forward_removal_control_by_method.csv
/mnt/data2/egeria_argument_experiments/microbench/*.json
/mnt/data2/egeria_argument_experiments/forward_removal/results/*.jsonl
/mnt/data2/egeria_argument_experiments/forward_removal/metrics/*_metrics_summary.json
```

## Paper-safe interpretation

Use:
"TC-ResiCache reaches Egeria-scale speedups in our local fine-tuning setting and demonstrates low-overhead GPU-resident forward-prefix removal."

Avoid:
"TC-ResiCache is faster than Egeria."
