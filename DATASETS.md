# Datasets

All TC-ResiCache scripts take a `--data_root` directory (or a `DATA_ROOT` environment
variable for the shell runners). Raw datasets and pretrained checkpoints are **not**
shipped in this repository; download them as below.

## CIFAR-10 / CIFAR-100

Used by: `code/core/tc_resicache_resnet50_cifar_finetune*.py`,
`code/core/resicache_plantcrc_v6_subpixel_resnet56_cifar10.py`,
`code/core/baseline_*_cifar10/100_*.py`, `code/baselines/freezeout_*`.

Downloaded automatically by torchvision on first run into `<data_root>`:

```bash
python - <<'PY'
import torchvision
for cls in (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100):
    cls(root="/path/to/data", train=True, download=True)
    cls(root="/path/to/data", train=False, download=True)
PY
```

## Tiny-ImageNet-200

Used by: `code/core/tiny_imagenet_resnet34_tc_resicache.py`.

```bash
env DATA_ROOT=/path/to/data bash scripts/download_tiny_imagenet.sh
```

The script downloads Tiny-ImageNet-200 and prepares a validation folder in ImageFolder
format (`val/<wnid>/*.JPEG`, keeping the original layout as `val_original/`).

> Note: the reference experiments used the standard 200-class folder layout. An
> intermediate experiment state on the original server temporarily flattened
> `val/images` (which ImageFolder reads as 201 classes and crashes); if you reproduce
> from a legacy mirror, ensure the per-`wnid` validation layout above.

## ImageNet-100 (ImageNet subset)

Used by: `code/core/tc_resicache_imagenet_sub_finetune.py`,
`code/baselines/smartfrz_resnet50_cifar_imagenet.py`.

Provide a standard ILSVRC/ImageNet directory:

```
<data_root>/imagenet/
  train/<synset>/*.JPEG
  val/<synset>/*.JPEG        # not used by the scripts; see below
```

The scripts take the first `--imagenet_classes` (default 100) synsets in sorted order
from `train/` and carve a deterministic per-class holdout from the end of `train/`
for validation, so a full `train/` copy is sufficient.

## Pretrained weights

Fine-tuning scripts use torchvision pretrained weights
(`--weights default`, ResNet ImageNet weights) downloaded automatically on first use.
