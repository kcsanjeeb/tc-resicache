# FreezeOut Reproduction for TC-ResiCache Comparison

This package provides a single modern PyTorch reproduction adapter:

```bash
freezeout_resnet_cifar_repro.py
```

It implements the FreezeOut idea for CIFAR ResNet-56/110/164:
- block-wise cosine LR schedules
- progressively freezes blocks when their scheduled LR reaches zero
- frozen blocks are set to eval mode and detached from backward
- JSONL output compatible with our TC-ResiCache analysis

## Why not use the original code directly?

The original FreezeOut code is useful for understanding the method, but it is old PyTorch/Python2-style and targets DenseNet, WRN, and VGG, not our CIFAR ResNet-110 settings. This file is a clean reproduction adapter for our experiments.

## Copy

```bash
cp freezeout_resnet_cifar_repro.py /mnt/data2/
cp run_freezeout_repro.sh /mnt/data2/
chmod +x /mnt/data2/freezeout_resnet_cifar_repro.py
chmod +x /mnt/data2/run_freezeout_repro.sh
```

## Run CIFAR-100 / ResNet-110

```bash
cd /mnt/data2

nohup env DATASET=cifar100 DEPTH=110 TARGET_ACC=70.0 SEED=42 EPOCHS=200 DATA_ROOT=/mnt/data2/data \
  bash /mnt/data2/run_freezeout_repro.sh \
  > /mnt/data2/freezeout_cifar100_resnet110_seed42_master.log 2>&1 &
```

## Run CIFAR-10 / ResNet-110

```bash
cd /mnt/data2

nohup env DATASET=cifar10 DEPTH=110 TARGET_ACC=92.1 SEED=42 EPOCHS=200 DATA_ROOT=/mnt/data2/data \
  bash /mnt/data2/run_freezeout_repro.sh \
  > /mnt/data2/freezeout_cifar10_resnet110_seed42_master.log 2>&1 &
```

## Monitor

```bash
tail -f /mnt/data2/freezeout_cifar100_resnet110_seed42_master.log
nvidia-smi
```

## Summary

```bash
cat /mnt/data2/freezeout_repro_cifar100_resnet110_seed42/summary/summary_freezeout_cifar100_resnet110_seed42.csv
cat /mnt/data2/freezeout_repro_cifar10_resnet110_seed42/summary/summary_freezeout_cifar10_resnet110_seed42.csv
```

## Variants

The runner executes:
- `freezeout_cubic_scaled_t08`: paper-recommended default
- `freezeout_linear_unscaled_t05`: paper-recommended close alternative

## Comparison use

Use this as an external baseline class:

| Method | Purpose |
|---|---|
| Baseline | no freeze/cache |
| FreezeOut reproduction | scheduled progressive freezing |
| no-reference freeze-only | our adaptive freeze gate |
| TC-ResiCache | freeze + augmentation-consistent cached reuse |
