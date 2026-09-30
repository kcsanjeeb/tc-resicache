# FreezeOut exact-mechanism comparison for TC-ResiCache

There are two meanings of “exact”:

1. Exact original FreezeOut paper reproduction: run the original repo on the original paper architectures/settings. This is not directly comparable to TC-ResiCache unless TC-ResiCache is also implemented for those exact settings.

2. Exact FreezeOut mechanism in our TC-ResiCache setting: same ResNet50/CIFAR-10/CIFAR-100 fine-tuning setup as the TC-ResiCache SmartFRZ comparison, using FreezeOut schedules and freezing behavior. This package does option 2.

It runs:
- CIFAR-10 / ResNet50 / target 96.10
- CIFAR-100 / ResNet50 / target 81.68
- cubic scaled t0=0.8
- linear unscaled t0=0.5

## Copy

```bash
cp freezeout_resnet50_cifar_exact_mechanism.py /mnt/data2/
cp run_freezeout_resnet50_cifar_exact_compare.sh /mnt/data2/
chmod +x /mnt/data2/freezeout_resnet50_cifar_exact_mechanism.py
chmod +x /mnt/data2/run_freezeout_resnet50_cifar_exact_compare.sh
```

## Run

```bash
cd /mnt/data2

nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 \
  bash /mnt/data2/run_freezeout_resnet50_cifar_exact_compare.sh \
  > /mnt/data2/freezeout_resnet50_cifar_exact_compare_master.log 2>&1 &
```

## Monitor

```bash
tail -f /mnt/data2/freezeout_resnet50_cifar_exact_compare_master.log
nvidia-smi
```

## Output

```bash
cat /mnt/data2/freezeout_resnet50_cifar_exact_compare/summary/summary_freezeout_resnet50_cifar_exact_compare.csv
```

Upload the summary CSV, JSONL files, and metric summaries after completion.
