# Reproducibility Guide

This guide reproduces the TC-ResiCache experimental results from scratch. It assumes
you have a Linux machine with one or more NVIDIA GPUs (the reference results used a
4× V100 32 GB server; any modern GPU works for the smaller configs).

## 0. Reference environment

| Component | Reference version |
|---|---|
| OS | Ubuntu 22.04 (kernel 4.15+), x86_64 |
| GPU | 4× NVIDIA V100 32 GB (single-GPU works for all fine-tuning runs) |
| Python | 3.10 |
| PyTorch | 2.6.0 + CUDA 12.4 |
| torchvision | 0.21.0 |

Establish it with either:

```bash
conda env create -f environment.yml && conda activate tc-resicache
```

```bash
docker build -t tc-resicache .
docker run --gpus all -it -v $(pwd):/workspace tc-resicache
```

Verify:

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
```

## 1. Datasets

Follow [DATASETS.md](../DATASETS.md). Minimum for the main tables: CIFAR-10 and
CIFAR-100 (auto-downloaded by torchvision). Tiny-ImageNet-200:
`env DATA_ROOT=$PWD/data bash scripts/download_tiny_imagenet.sh`. ImageNet-100 needs a
local ILSVRC `train/` copy.

## 2. What each suite corresponds to

| Suite (in `results/`) | Experiment family | Runner |
|---|---|---|
| `resnet50_finetuning/` | TC-SI fine-tuning, CIFAR-10/100 ResNet-50 @224px: seed-42 screening, seeds 43/44 strengthening, repair sweep | `run_tc_resicache_resnet50_cifar_smartfrz_compare.sh`, `run_resnet50_multiseed_strengthening.sh`, `run_cifar10_resnet50_repair_sweep.sh` |
| `full_training/` | TC-CR full training: C10/RN56 (seeds 42–44), C100/RN110, Tiny-ImageNet TinyResNet-34 | `run_cifar10_final_hardware_metrics_4gpu.sh`, `run_cifar100_resnet110_tcresicache_4gpu.sh`, `run_tiny_imagenet_long_boundary_sweep.sh` |
| `baselines_freezeout/` | FreezeOut exact-mechanism reproductions (C10/C100, RN50 exact + RN110) | `run_freezeout_resnet50_cifar_exact_compare.sh` |
| `egeria_argument/` | Forward-removal controls (cache disabled) — the Egeria-style counter-argument | `run_egeria_argument_experiments.sh`, `run_forward_removal_control_experiment.sh` |
| `appendix_sweeps/` | Per-seed boundary/candidate sweeps (C10/C100 RN110, C100 seeds 42–44) | scripts inside each suite dir |

Per-suite details: [RESULTS.md](RESULTS.md) and [suites/](suites/). The mapping from
every thesis result to its script, runner, and data location:
[EXPERIMENT_CODE_MAP.md](EXPERIMENT_CODE_MAP.md).

## 3. Running an experiment

All runners take configuration through environment variables, exactly as used on the
reference server:

```bash
env DATA_ROOT=$PWD/data GPUS_CSV=0,1,2,3 \
  bash runners/run_tc_resicache_resnet50_cifar_smartfrz_compare.sh \
  > smartfrz_compare_master.log 2>&1 &
```

- `DATA_ROOT` — dataset directory (see DATASETS.md).
- `GPUS_CSV` — comma-separated GPU ids for intra-suite scheduling.
- `SEEDS_CSV` / `SEED` — seed selection where the runner supports it.

The runners invoke the scripts in `code/` and parameterize their locations through
environment variables (they default to the original server's `/mnt/data2` layout) —
point them at this repository, e.g.:

```bash
env DATA_ROOT=$PWD/data GPUS_CSV=0,1,2,3 \
    SCRIPT=$PWD/code/core/tc_resicache_resnet50_cifar_finetune.py \
    MONITOR=$PWD/code/monitoring/monitor_command.py \
  bash runners/run_tc_resicache_resnet50_cifar_smartfrz_compare.sh \
  > smartfrz_compare_master.log 2>&1 &
```

Other overridables include `BASE` (output directory), `SEEDS_CSV`/`SEED`, `EPOCHS`,
`BATCH_SIZE`, and `TRAIN_AUG` — see the header of each runner.

### Single-run equivalent (no runner)

Each runner is a thin scheduler over direct script invocations, e.g.:

```bash
python code/core/tc_resicache_resnet50_cifar_finetune.py \
  --dataset cifar100 --data_root data --output_json runs/c100_tc.jsonl \
  --mode tc_resicache --boundary layer3 --freeze_epoch 4 --cache_start_epoch 5 \
  --model resnet50 --epochs 10 --seed 43
```

Use `--help` on any script for the full argument list.

## 4. Hardware measurement

Reference time/energy figures were collected on **exclusive GPUs** only (one run per
GPU); contended runs are not comparable. Per-run power/energy/GPU sampling used:

```bash
python code/monitoring/monitor_command.py \
  --name my_run --out-dir runs/metrics --gpu-index 0 --interval 5 \
  -- python code/core/tc_resicache_resnet50_cifar_finetune.py <args>
```

## 5. Verifying a reproduction

1. Every training run writes a JSONL log (`--output_json`) with per-epoch metrics
   plus cache counters (memory hits, corrected reuses, probe error, transform time).
2. Aggregate with `code/analysis/analyze_results.py <results dir>` to produce the
   summary tables, then compare against the reference `summary/*.csv` for that suite.
3. Check integrity of the shipped reference files at any time:

   ```bash
   shasum -a 256 -c SHA256SUMS.txt
   ```

## 6. Probes and ablations (optional but recommended)

- Cache-correctness gates: `code/probes/tc_cache_correctness_probe.py`,
  `probe_transform_sensitivity.py`, `probe_reconstruction_limits.py`.
- TC-CR faithfulness / noise tolerance (thesis §6.6):
  `code/probes/probe_tccr_faithfulness.py`.
- Augmentation ablations: `code/ablations/resicache_aug_*_resnet56_cifar10.py`.
- Lookup-vs-prefix cost: `code/microbenchmarks/microbench_prefix_vs_cache.py`.

## 7. Known gaps (server-only artifacts)

A few scripts referenced in the historical experiment map were never archived off the
server and are therefore not in this repository — most notably the generated CIFAR-100
RN110 TC-CR training script and the SmartFRZ direct-compare driver. Their reference
outputs *are* included under `results/`. See the note at the end of
[EXPERIMENT_CODE_MAP.md](EXPERIMENT_CODE_MAP.md) for the full list and provenance
details in [provenance/artifact_generation_summary.md](provenance/artifact_generation_summary.md).
