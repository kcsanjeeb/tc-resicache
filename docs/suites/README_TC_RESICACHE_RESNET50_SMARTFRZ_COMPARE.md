# TC-ResiCache ResNet50 CIFAR SmartFRZ Comparison

This package runs TC-ResiCache-style experiments in the same high-level setting as SmartFRZ Table 2:

- ImageNet-pretrained ResNet50
- CIFAR-10 and CIFAR-100
- 10 epochs
- batch size 32
- input size 224
- SmartFRZ-like augmentation

## Targets

The default target accuracies are from the SmartFRZ paper's ResNet50 full-training rows:

```text
CIFAR-10:  96.10%
CIFAR-100: 81.68%
```

SmartFRZ itself reports:

```text
CIFAR-10 SmartFRZ:  96.12%, 1955s
CIFAR-100 SmartFRZ: 81.73%, 1787s
```

## Copy

```bash
cp tc_resicache_resnet50_cifar_finetune.py /mnt/data2/
cp run_tc_resicache_resnet50_cifar_smartfrz_compare.sh /mnt/data2/

chmod +x /mnt/data2/tc_resicache_resnet50_cifar_finetune.py
chmod +x /mnt/data2/run_tc_resicache_resnet50_cifar_smartfrz_compare.sh
```

## Run

```bash
cd /mnt/data2

nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 \
  bash /mnt/data2/run_tc_resicache_resnet50_cifar_smartfrz_compare.sh \
  > /mnt/data2/tc_resicache_resnet50_cifar_smartfrz_compare_master.log 2>&1 &
```

## Monitor

```bash
tail -f /mnt/data2/tc_resicache_resnet50_cifar_smartfrz_compare_master.log
nvidia-smi
```

## Output

```bash
cat /mnt/data2/tc_resicache_resnet50_cifar_smartfrz_compare/summary/summary_tc_resicache_resnet50_cifar_smartfrz_compare.csv
```

Upload:
- the summary CSV
- all JSONL files from results/
- all *_metrics_summary.json files from metrics/

## Notes

The runner launches:

```text
cifar10 baseline
cifar10 freeze_layer3_e4
cifar10 tc_layer3_e4_c5
cifar10 tc_layer4_e4_c5

cifar100 baseline
cifar100 freeze_layer3_e4
cifar100 tc_layer3_e4_c5
cifar100 tc_layer4_e4_c5
```

Layer3/layer4 are used because ResNet50 feature banks at 224x224 are large:
- layer3 is about 20GB fp16 for 50k samples
- layer4 is about 10GB fp16 for 50k samples

## Scientific caution

SmartFRZ uses stochastic RandomResizedCrop. TC-ResiCache cache reuse under stochastic augmentation is the hard part of our method. This script is a screening experiment. If TC cache under `--train_aug smartfrz` harms accuracy, rerun with:

```bash
TRAIN_AUG=deterministic
```

to isolate systems overhead from augmentation correctness.
