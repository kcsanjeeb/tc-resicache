# Main reproducibility commands

## ResNet50 seed42 screening
```bash
nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 bash run_tc_resicache_resnet50_cifar_smartfrz_compare.sh > tc_resicache_resnet50_cifar_smartfrz_compare_master.log 2>&1 &
```

## ResNet50 seeds 43/44 strengthening
```bash
nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 SEEDS_CSV=43,44 bash run_resnet50_multiseed_strengthening.sh > resnet50_multiseed_strengthening_master.log 2>&1 &
```

## CIFAR-10 ResNet50 repair sweep
```bash
nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 SEED=44 bash run_cifar10_resnet50_repair_sweep.sh > cifar10_resnet50_repair_sweep_seed44_master.log 2>&1 &
```

## Egeria-facing forward-removal evidence
```bash
nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 SEEDS_CSV=42,43,44 RUN_TC=0 bash run_egeria_argument_experiments.sh > egeria_argument_experiments_master.log 2>&1 &
```

## CIFAR-100 / ResNet-110 full training
```bash
nohup env DATA_ROOT=/mnt/data2/data GPUS_CSV=0,1,2,3 bash run_cifar100_resnet110_tcresicache_4gpu.sh > cifar100_resnet110_master.log 2>&1 &
```

## Running from this repository

The commands above were captured from the original GPU server (paths under
`/mnt/data2/`). To run from this repository:

- Runner scripts in `runners/` assume the training scripts in `code/` are importable —
  either put the repo's `code/` directory on `PYTHONPATH`, or run the scripts from the
  repo root with the script paths adjusted accordingly.
- The environment variables `DATA_ROOT` (dataset location) and `GPUS_CSV` (comma-separated
  GPU ids) must be set before invoking any runner; `SEED` / `SEEDS_CSV` are required by
  the seed-parameterized runners.
- Use `scripts/download_tiny_imagenet.sh` to fetch Tiny-ImageNet-200 into `DATA_ROOT`.
