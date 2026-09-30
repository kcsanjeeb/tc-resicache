# ResNet50 multi-seed strengthening runs

This package runs seeds 43 and 44 for the final ResNet50 comparison set.

## Runs

CIFAR-10:
- baseline
- FreezeOut cubic scaled t0=0.8
- TC-ResiCache L4 e4/c5

CIFAR-100:
- baseline
- FreezeOut cubic scaled t0=0.8
- TC-ResiCache L3 e4/c5

Targets:
- CIFAR-10: 96.10
- CIFAR-100: 81.68

## Required scripts already on server

/mnt/data2/tc_resicache_resnet50_cifar_finetune.py
/mnt/data2/freezeout_resnet50_cifar_exact_mechanism.py
/mnt/data2/monitor_command.py

## Copy

```bash
cp run_resnet50_multiseed_strengthening.sh /mnt/data2/
chmod +x /mnt/data2/run_resnet50_multiseed_strengthening.sh
```

## Run

```bash
cd /mnt/data2

nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 SEEDS_CSV=43,44 \
  bash /mnt/data2/run_resnet50_multiseed_strengthening.sh \
  > /mnt/data2/resnet50_multiseed_strengthening_master.log 2>&1 &
```

## Monitor

```bash
tail -f /mnt/data2/resnet50_multiseed_strengthening_master.log
nvidia-smi
```

## Output

```bash
cat /mnt/data2/resnet50_multiseed_strengthening/summary/summary_resnet50_multiseed_strengthening.csv

cat /mnt/data2/resnet50_multiseed_strengthening/summary/summary_resnet50_multiseed_strengthening_by_method.csv
```

Upload after completion:
- summary_resnet50_multiseed_strengthening.csv
- summary_resnet50_multiseed_strengthening_by_method.csv
- all JSONL files from results/
- all *_metrics_summary.json files from metrics/

Seed42 from previous runs should be combined later with these seed43/44 results for the final 3-seed table.
