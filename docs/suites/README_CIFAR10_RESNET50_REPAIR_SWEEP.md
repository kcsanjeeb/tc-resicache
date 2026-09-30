# CIFAR-10 / ResNet50 TC-ResiCache repair sweep

This is a focused seed44 repair sweep.

## Why

Previous 3-seed result:

```text
CIFAR-10 TC L4 e4/c5:
  target reached = 2/3
  seed44 missed 96.10 target
  total runtime and energy were excellent
```

This sweep tests slightly safer schedules and a layer3 alternative.

## Variants

```text
tc_l4_e5_c6
tc_l4_e6_c7
tc_l3_e4_c5
tc_l3_e5_c6
```

Optional controls:

```text
freeze_l4_e5
freeze_l3_e4
```

Enable controls with:

```bash
RUN_FREEZE_CONTROLS=1
```

## Copy

```bash
cp run_cifar10_resnet50_repair_sweep.sh /mnt/data2/
chmod +x /mnt/data2/run_cifar10_resnet50_repair_sweep.sh
```

## Run seed44 repair sweep

```bash
cd /mnt/data2

nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 SEED=44 \
  bash /mnt/data2/run_cifar10_resnet50_repair_sweep.sh \
  > /mnt/data2/cifar10_resnet50_repair_sweep_seed44_master.log 2>&1 &
```

## Monitor

```bash
tail -f /mnt/data2/cifar10_resnet50_repair_sweep_seed44_master.log
nvidia-smi
```

## Output

```bash
cat /mnt/data2/cifar10_resnet50_repair_sweep_seed44/summary/summary_cifar10_resnet50_repair_sweep_seed44.csv
```

Upload:
- summary CSV
- JSONL files from results/
- *_metrics_summary.json files from metrics/

## Candidate decision

Pick the first candidate that:

```text
target_reached = True
final_ge_target = True
total_time_sec remains much lower than baseline
energy remains much lower than baseline
```

If a candidate works on seed44, then run that candidate on seeds 42 and 43 for the final 3-seed table.
