# Results

Reference experiment outputs for TC-ResiCache (paper/thesis). These are the exact
numbers the reported tables and figures are traced to; nothing here was regenerated.

## Top-level suites

- `resnet50_finetuning/` — TC-SI (fine-tuning) experiments with ResNet-50 on
  CIFAR-10/100: seed-42 screening, multiseed strengthening, and the CIFAR-10 repair sweep.
- `full_training/` — TC-CR (full-training) experiments: CIFAR-10 ResNet-56,
  CIFAR-100 ResNet-110, and Tiny-ImageNet ResNet-34 (seed 42, alt config, multiseed).
- `baselines_freezeout/` — FreezeOut reproductions (exact-mechanism ResNet-50 and
  CIFAR ResNet-110 runs).
- `egeria_argument/` — forward-removal control experiments and the lookup-vs-prefix
  microbenchmark supporting the Egeria-argument analysis.
- `appendix_sweeps/` — per-seed appendix sweeps (CIFAR-10/CIFAR-100, ResNet-56/110).

## Per-suite layout

Each suite (or each experiment subfolder within it) follows this convention:

- `results/*.jsonl` — raw per-epoch training logs (one line per epoch with loss,
  accuracy, timing, and cache statistics).
- `metrics/*.{csv,json}` — hardware and accuracy metrics per run (runtime, energy,
  peak memory, final/best accuracy).
- `summary/*.csv` — aggregated, table-ready summaries across the runs of the suite.
- `logs/*.log` — full training stdout/stderr logs captured from the runner scripts.

`SHA256SUMS.txt` at the repository root covers every file in this `results/` tree
(except this README) for integrity verification.
