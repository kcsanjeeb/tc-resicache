# TC-ResiCache

**Transformation-Consistent Caching for Compute Reuse in Deep Network Training**

Reference implementation and reproducibility artifact for the TC-ResiCache master's
thesis. TC-ResiCache speeds up deep network training by caching dense intermediate
activations in a GPU-resident bank and reusing them instead of re-running frozen
prefix layers — with explicit, testable correctness gates that decide *when a cached
activation may be served for a new sample*.

Two gating regimes cover the two training settings:

| Regime | Gate | Setting | Idea |
|---|---|---|---|
| **TC-SI** | sample-index key `(i, L)` | fine-tuning | After the data pipeline is frozen (deterministic augmentation from a fixed seed), the transform of sample *i* at layer *L* is reproducible, so the cached activation can be served by sample index. |
| **TC-CR** | transformation-consistent residual reconstruction | full training | Random crop/flip parameters are recorded per sample; the residual between the true transform and the cached entry's transform is reconstructed with an explicit bilinear kernel under a proven L∞ error bound, and served only when the residual is provably suffix-tolerable. |

The gate is the contribution: naive "precompute once" caching collapses under
distribution shift (the project history includes a stale-cache bug that drove
accuracy from ~91% to ~37% — the gates exist so that cannot happen silently).

## Headline results (reference runs, included in `results/`)

Numbers below are computed from the reference logs shipped in this repository;
every value is traceable to a JSONL/CSV in `results/` (see `docs/RESULTS.md`).

- **Fine-tuning (ResNet-50, CIFAR-10/100 @ 224px, seeds 43/44)** — TC-SI reaches the
  target accuracy in **~44% less time and ~45–55% less energy** than the no-cache
  baseline, beating an exact-mechanism FreezeOut reimplementation on both axes
  (`results/resnet50_finetuning/multiseed_strengthening/summary/`).
- **Full training (CIFAR-10, ResNet-56, 3 seeds)** — TC-CR reaches the accuracy
  target with **~13% less total time and ~25% less estimated energy** than baseline
  (`results/full_training/cifar10_resnet56/summary/`).
- **Lookup vs prefix-forward microbenchmark** — a cache bank lookup is **154–298×**
  faster than a cached-prefix forward pass at CIFAR-10 ResNet-56 scale
  (`code/microbenchmarks/microbench_prefix_vs_cache.py`).
- Full experimental sweep: CIFAR-10/100, Tiny-ImageNet, ImageNet-100; ResNet-18/34/50/101/110,
  ResNet-56, TinyResNet-34, ViT-B/16; 5 suites, 371 result files, all checksummed
  (`SHA256SUMS.txt`).

## Repository layout

```
code/
  core/            TC-ResiCache training scripts (the method)
                   - tc_resicache_resnet50_cifar_finetune{,_sfz}.py   TC-SI fine-tuning (RN18/34/50/101)
                   - resicache_plantcrc_v6_subpixel_resnet56_cifar10.py  TC-CR full training (C10/RN56)
                   - tiny_imagenet_resnet34_tc_resicache.py           TC-CR full training (Tiny-ImageNet)
                   - tc_resicache_imagenet_sub_finetune.py            TC-SI fine-tuning (ImageNet-100, ViT)
                   - baseline_*_jsonl.py                              no-cache full-training baselines
  baselines/       egeria.py, autofreeze.py (reimplementations),
                   freezeout_resnet*.py (exact-mechanism reproductions),
                   smartfrz_*.py (SmartFRZ comparison points)
  probes/          cache-correctness, transform-sensitivity, reconstruction-limit,
                   and TC-CR faithfulness probes (the evidence behind the gates)
  ablations/       augmentation-policy ablations (C10/RN56)
  microbenchmarks/ prefix-forward vs cache-lookup cost
  analysis/        jsonl -> summary aggregation
  monitoring/      per-run power/energy/GPU sampler (5 s)
runners/           end-to-end multi-GPU experiment runners (the exact commands used)
results/           reference outputs: results/ metrics/ summary/ logs/ per suite
docs/              REPRODUCIBILITY.md (the guide), COMMANDS.md,
                   EXPERIMENT_CODE_MAP.md (thesis result -> script -> data map),
                   RESULTS.md (results index), suites/, provenance/
scripts/           dataset download/preparation helpers
```

Thesis LaTeX sources and PDFs are intentionally **not** part of this repository.

## Quickstart

```bash
# 1. Environment (pick one)
conda env create -f environment.yml && conda activate tc-resicache
#   or
docker build -t tc-resicache . && docker run --gpus all -it -v $(pwd):/workspace tc-resicache

# 2. Datasets (see DATASETS.md)
python -c "import torchvision; torchvision.datasets.CIFAR10(root='data', train=True, download=True)"

# 3. One fine-tuning run (TC-SI, ~10 epochs on one GPU)
python code/core/tc_resicache_resnet50_cifar_finetune.py \
  --dataset cifar10 --data_root data --output_json out.jsonl \
  --mode tc_resicache --boundary layer4 --freeze_epoch 4 --cache_start_epoch 5

# 4. Compare against the reference run
python code/analysis/analyze_results.py results/resnet50_finetuning/   # aggregate jsonl -> summary tables
```

## Reproducing the full experimental suites

The shell runners in `runners/` are the exact commands used to produce the reference
results (originally on a 4× NVIDIA V100 32 GB server). Start with
[**docs/REPRODUCIBILITY.md**](docs/REPRODUCIBILITY.md), then
[docs/COMMANDS.md](docs/COMMANDS.md) and
[docs/EXPERIMENT_CODE_MAP.md](docs/EXPERIMENT_CODE_MAP.md).

## Integrity

All 371 reference result files are covered by `SHA256SUMS.txt`:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

## Citation

```bibtex
@software{kc2026tcresicache,
  author = {K C, Sanjeeb},
  title  = {TC-ResiCache: Transformation-Consistent Caching for Compute Reuse in Deep Network Training},
  year   = {2026},
  url    = {https://github.com/kcsanjeeb/tc-resicache}
}
```

## License

[MIT](LICENSE)
