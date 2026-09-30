# Results index

The `results/` tree holds the paper/thesis reference outputs (see `results/README.md`
for the layout convention). SHA-256 checksums for the whole tree are in
`SHA256SUMS.txt` at the repository root. This document indexes each top-level suite.

## resnet50_finetuning

TC-SI fine-tuning experiments with a ResNet-50 backbone on CIFAR-10/100.
`seed42_screening/` is the initial operating-point screening; `multiseed_strengthening/`
re-runs the main configurations across seeds 42–44 to tighten confidence intervals;
`cifar10_repair_sweep/` is the repair sweep that re-measures CIFAR-10 configurations
after the dataset/transform fix. Raw per-epoch JSONL logs live in each sub-suite's
`results/`, hardware/accuracy metrics in `metrics/`, aggregated CSVs in `summary/`,
and full training logs in `logs/`.

## full_training

TC-CR full-training experiments: `cifar10_resnet56/` (CIFAR-10 ResNet-56, the
canonical TC-CR regime with final 4-GPU hardware metrics), `cifar100_resnet110/`
(CIFAR-100 ResNet-110 with the L2/L3 boundary sweep), and the Tiny-ImageNet
ResNet-34 runs (`tinyimagenet_seed42/`, `tinyimagenet_seed42_alt/`,
`tinyimagenet_multiseed/`) covering both the primary and alternative configurations
across seeds.

## baselines_freezeout

FreezeOut reproduction runs used for the baseline column of the fine-tuning table:
`resnet50_exact/` (exact-mechanism reproduction on CIFAR ResNet-50) and the
`cifar10_resnet110_seed42/` / `cifar100_resnet110_seed42/` runs. Same
results/metrics/summary/logs convention as the other suites.

## egeria_argument

Forward-removal control experiments (§6.4 of the thesis): cache-disabled runs of the
fine-tuning script to separate the effect of layer removal from caching, plus the
`microbench/` lookup-vs-prefix microbenchmark (Tab. 5.2, 154–298× lookup speedups).
`forward_removal/` holds the per-seed suites; top-level `summary/` and `logs/` hold
the aggregated view and master logs.

## appendix_sweeps

Per-seed sweeps reported in the appendix: `cifar100_resnet56_seed42/`,
`cifar100_seed42/`, `cifar100_seed43_44/`, and `cifar10_resnet110_seed42/` — the
held-out-seed and cross-model robustness checks for the full-training results.

## Subfolder convention

Within every suite, each experiment subfolder uses:

- `results/*.jsonl` — raw per-epoch logs (loss, accuracy, timing, cache stats).
- `metrics/*.{csv,json}` — per-run hardware/accuracy metrics (runtime, energy,
  peak memory, final/best accuracy).
- `summary/*.csv` — aggregated, table-ready summaries across runs.
- `logs/*.log` — full stdout/stderr training logs from the runners.
