> **Note:** Archived 2026-09-30 into the tc-resicache repository; server paths refer to the original GPU server.

# Experiment ↔ Code Reference Map — TC-ResiCache Thesis

**Purpose:** which experiment produced which thesis result, with the exact training script,
runner, and data locations. Compiled 2026-09-20 from the aggregator's `INDEX.md`
(`thesis/tables_data/`), `04_results/RUNS_STATUS.md`, and server inspection.

**Locations:**
- **Server**: `san@10.249.185.176`, main store `/mnt/data2/`, older mirror `/home/san/mac-linux/implementation-compute-reuse/`
- **Local**: `~/Research/Thesis_Final/04_results/` (results archive), `~/Research/Thesis_Final/06_code/` (code archive)
- **Python env on server**: `/home/san/miniconda/envs/mac-linux/bin` (torch 2.6.0+cu124). System `python3` is 3.6 and has NO torch — always launch with the conda env on PATH.

---

## 1. Thesis result → experiment → code (headline table)

| Thesis result | Suite(s) | Training script | Runner | Raw data (local archive) |
|---|---|---|---|---|
| Fine-tune main table (Tab. 6.4) C10/C100 RN50 | `resnet50_multiseed_strengthening`, `tc_resicache_resnet50_cifar_smartfrz_compare`, `cifar10_resnet50_repair_sweep` | `tc_resicache_resnet50_cifar_finetune.py` (server `_sfz` variant) | `run_resnet50_multiseed_strengthening.sh`, `run_tc_resicache_resnet50_cifar_smartfrz_compare.sh`, `run_cifar10_resnet50_repair_sweep.sh` | `04_results/generalization/comparision_tinyimagenet_suites/` |
| Fine-tune main table, SmartFRZ column | `smartfrz_direct_compare_results`, `sfz_rerun_finetune` | `smartfrz_direct_compare/smartfrz_resnet50_cifar.py` | `run_smartfrz_direct_compare.sh` | `04_results/icassp_core/` |
| Fine-tune main table, FreezeOut column | `resnet_variant_multiseed`, strengthening suites | `freezeout_resnet50_cifar_exact_mechanism.py` | `run_freezeout_resnet50_cifar_exact_compare.sh` | `04_results/icassp_core/` |
| Fine-tune ViT (C10/C100) | `vit_cifar10_extension_seq`, `vit_matrix` (+`_seq` rerun) | `smartfrz_vit_cifar.py` + fine-tune script | `run_vit_*` (in suite dirs) | `04_results/extensions_post_icassp/` |
| Fine-tune IM-100 | `imagenet_sub_extension`, `vit_matrix` | `tc_resicache_imagenet_sub_finetune.py`, `smartfrz_resnet50_cifar_imagenet.py` | suite runners | `04_results/extensions_post_icassp/` |
| Full training C10/RN56, 5-seed (Tab. 6.x) | `tc_resicache_cifar10_final_hardware_metrics` (seeds 42–44) + `cifar10_fulltrain_clean_s45_s46` (seeds 45,46) | `resicache_plantcrc_v6_subpixel_resnet56_cifar10.py` (TC/freeze arms), `baseline_resnet56_cifar10_jsonl.py` (baseline) | `run_cifar10_final_hardware_metrics_4gpu.sh`, `run_c10_clean_sequential_s45_s46.sh` | `04_results/energy_hardware/`, `04_results/extensions_post_icassp/cifar10_fulltrain_clean_s45_s46/` |
| Full training C100/RN110 + s85 sweep | `table4_cifar100_resnet110_tc_cr_l23_*` (~20 sweep dirs) | `tc_resicache_cifar100_resnet110_lrms.py` (server `/mnt/data2/`) | `run_table4_*.sh` | `04_results/icassp_core/` |
| Full training Tiny-ImageNet | `tiny_imagenet_best_candidates_multiseed_s43_s44` (TC), `tiny_imagenet_controls_s43_s44` (baseline/freeze, new) | `tiny_imagenet_resnet34_tc_resicache.py` | `run_tiny_best_candidates_multiseed_s43_s44.sh`, `run_tiny_controls_s43_s44.sh` | `04_results/generalization/generalization_3/`, `04_results/extensions_post_icassp/tiny_imagenet_controls_s43_s44/` |
| Forward-removal controls (§6.4) | `egeria_argument_experiments/forward_removal` | fine-tune script (cache disabled) | `run_egeria_argument_experiments.sh` | `04_results/generalization/comparision_tinyimagenet_suites/egeria_argument_experiments/` |
| Microbenchmark 154–298× (Tab. 5.2) | submission archive | `microbench_prefix_vs_cache.py` | — | `06_code/main_scripts/` (from `icassp_submission_archive/`) |
| Cache-correctness probes (Ch.3) | `tc_cache_correctness_probes` (server backup) | `tc_cache_correctness_probe.py`, `probe_transform_sensitivity.py` | — | `06_code/main_scripts/` |
| Fidelity + noise tolerance (§6.6) | `tccr_probe/fullregime{,_s43,_s44}` | `probe_tccr_faithfulness.py` | — | `04_results/ablations/tccr_probe/` |
| Delta-update null (§6.5.4) | `delta_ablation` | `resicache_plantcrc_v6_subpixel_resnet56_cifar10.py` | delta_ablation runner | `04_results/ablations/delta_ablation/` |
| Held-out seed 45 (fine-tune) | `heldout_seed` | fine-tune script | — | `04_results/extensions_post_icassp/heldout_seed/` |
| Precompute-once failure (§7.1) | `precompute_once` | fine-tune script | — | `04_results/extensions_post_icassp/precompute_once/` |
| Conservative points (§6.2.7) | `conservative_operating_points` | fine-tune script | — | `04_results/extensions_post_icassp/conservative_operating_points/` |
| Fair compares (§6.7) | `fair_compare_ext` | fine-tune + freezeout scripts | — | `04_results/extensions_post_icassp/fair_compare_ext/` |
| RN18 extension (§6.7) | `resnet18_cifar10_extension_seq` | fine-tune script (`--model resnet18`) | — | `04_results/extensions_post_icassp/` |
| **RN50 full-train matrix (new, running)** | `fulltrain_rn50_translate224` | `smartfrz_resnet50_cifar.py` (baseline/smartfrz), `freezeout_resnet50_cifar_exact_mechanism.py` (freezeout) | `run_ft_rn50_matrix_queue.sh` | server only until pulled |
| Baselines (Egeria/AutoFreeze reimplementations) | local `results/` | `baselines/egeria.py`, `baselines/autofreeze.py` | — | `04_results/baselines/results_local_repo/`, `06_code/baselines/` |
| Early era (RedSync, superseded) | `results/` Apr, `san-backup/results/` | `exp1..7` scripts, `unified_*.py` | `run_*.sh` | `04_results/baselines/results_local_repo/` |

## 2. Main code inventory (with locations)

| Script | Role | Server path | Local archive |
|---|---|---|---|
| `resicache_plantcrc_v6_subpixel_resnet56_cifar10.py` | TC-CR full training (CIFAR-10 RN56) — the canonical TC-CR implementation | `/mnt/data2/` | `06_code/main_scripts/server_latest/` |
| `tc_resicache_cifar100_resnet110_lrms.py` | TC-CR full training (CIFAR-100 RN110) | `/mnt/data2/` | — |
| `tiny_imagenet_resnet34_tc_resicache.py` | TC-CR full training (Tiny, TinyResNet34-w16) | `/mnt/data2/` (restored from san-backup 2026-09-19) | — |
| `tc_resicache_resnet50_cifar_finetune.py` (+ `_sfz`) | TC-SI fine-tuning (RN50/ViT, CIFAR/IM-100) | `/mnt/data2/` | `06_code/main_scripts/` |
| `smartfrz_resnet50_cifar.py` | SmartFRZ reimplementation + baseline/freeze/TC-SI modes | `/mnt/data2/smartfrz_direct_compare/` | — |
| `smartfrz_vit_cifar.py` | SmartFRZ on ViT-B/16 | `/mnt/data2/` | — |
| `smartfrz_resnet50_cifar_imagenet.py` | SmartFRZ on IM-100 | `/mnt/data2/` | — |
| `tc_resicache_imagenet_sub_finetune.py` | TC-SI on IM-100 | `/mnt/data2/` | `06_code/main_scripts/server_latest/` |
| `freezeout_resnet50_cifar_exact_mechanism.py` | FreezeOut exact-mechanism reproduction | `/mnt/data2/` | — |
| `freezeout_resnet_cifar_repro.py` | FreezeOut on CIFAR ResNets | `/mnt/data2/san-backup/` | `04_results/baselines/freezeout_repro_package_server/` |
| `baseline_resnet56_cifar10_jsonl.py` | plain full-training baseline (C10 RN56) | `/mnt/data2/` | — |
| `monitor_command.py` / `monitor_command_tiny.py` | per-run power/energy/memory sampler (5-s) | `/mnt/data2/` | — |
| `tc_cache_correctness_probe.py`, `probe_transform_sensitivity.py`, `probe_reconstruction_limits.py` | Ch.3 probes | server + local | `06_code/main_scripts/` |
| `probe_tccr_faithfulness.py` | §6.6 fidelity probe | server | `06_code/main_scripts/` (from archive) |
| `microbench_prefix_vs_cache.py` | lookup vs prefix-forward microbenchmark | server | `06_code/main_scripts/` |
| `baselines/egeria.py`, `baselines/autofreeze.py` | baseline reimplementations (early era) | — | `06_code/baselines/` |
| `aggregate_results.py` | raw jsonl → summary/curve CSVs | local | `thesis/tables_data/` |
| `make_fig_*.py` | all thesis figures | local | `thesis/figures/` |

## 3. Notes and provenance rules

1. **Numbers in the thesis trace to `04_results/` or the ICASSP submission**; anything older
   (RedSync/workshop era) is design history only — never cite its result numbers.
2. **Contention rule**: time/energy values are valid only from exclusive-GPU runs
   (clean suite, matrix queues). Contended runs are marked in table footnotes.
3. **Dataset fix (2026-09-19)**: `/mnt/data2/data/tiny-imagenet-200/val/images` was moved to
   `val_images_flat` (ImageFolder was seeing 201 classes; crashed all Tiny runs).
   Reversible: move it back if the flat layout is ever needed.
4. **vit_matrix vs vit_matrix_seq** are one experiment run twice (report once).
5. Suites without a listed runner were launched ad hoc; their master logs
   (`*_master.log` in each suite dir on the server) record the exact commands.
6. To re-verify any row: find the suite's `summary/` CSV or the per-run
   `results/*.jsonl`, or re-run `thesis/tables_data/aggregate_results.py` and read
   `INDEX.md`.

## 4. Repository locations

Location of each main script inside this repository (paths relative to the repo root):

| Script (as named above) | Location in this repository |
|---|---|
| `tc_resicache_resnet50_cifar_finetune.py` | `code/core/tc_resicache_resnet50_cifar_finetune.py` |
| `tc_resicache_resnet50_cifar_finetune_sfz.py` (the `_sfz` variant) | `code/core/tc_resicache_resnet50_cifar_finetune_sfz.py` |
| `resicache_plantcrc_v6_subpixel_resnet56_cifar10.py` | `code/core/resicache_plantcrc_v6_subpixel_resnet56_cifar10.py` |
| `tc_resicache_cifar100_resnet110_lrms.py` (CIFAR-100 RN110 TC-CR; server-only, see note below) | — (results under `results/full_training/cifar100_resnet110/`) |
| `baseline_cifar100_resnet110_jsonl.py` (CIFAR-100 RN110 plain baseline) | `code/core/baseline_cifar100_resnet110_jsonl.py` |
| `baseline_resnet56_cifar10_jsonl.py` (CIFAR-10 RN56 plain baseline) | `code/core/baseline_resnet56_cifar10_jsonl.py` |
| `tiny_imagenet_resnet34_tc_resicache.py` | `code/core/tiny_imagenet_resnet34_tc_resicache.py` |
| `tc_resicache_imagenet_sub_finetune.py` | `code/core/tc_resicache_imagenet_sub_finetune.py` |
| `smartfrz_vit_cifar.py` | `code/baselines/smartfrz_vit_cifar.py` |
| `smartfrz_resnet50_cifar_imagenet.py` | `code/baselines/smartfrz_resnet50_cifar_imagenet.py` |
| `freezeout_resnet50_cifar_exact_mechanism.py` | `code/baselines/freezeout_resnet50_cifar_exact_mechanism.py` |
| `freezeout_resnet_cifar_repro.py` | `code/baselines/freezeout_resnet_cifar_repro.py` |
| `baselines/egeria.py`, `baselines/autofreeze.py` | `code/baselines/egeria.py`, `code/baselines/autofreeze.py` |
| `tc_cache_correctness_probe.py`, `probe_transform_sensitivity.py`, `probe_reconstruction_limits.py`, `probe_cache_correctness_calibrated.py`, `probe_two_views.py`, `probe_tccr_faithfulness.py` | `code/probes/` |
| `resicache_aug_ablation_resnet56_cifar10.py`, `resicache_aug_safe_lowoverhead_resnet56_cifar10.py` | `code/ablations/` |
| `microbench_prefix_vs_cache.py` | `code/microbenchmarks/microbench_prefix_vs_cache.py` |
| `analyze_results.py` | `code/analysis/analyze_results.py` |
| `monitor_command.py` / `monitor_command_tiny.py` | `code/monitoring/` |
| `run_resnet50_multiseed_strengthening.sh`, `run_tc_resicache_resnet50_cifar_smartfrz_compare.sh`, `run_cifar10_resnet50_repair_sweep.sh`, `run_egeria_argument_experiments.sh`, `run_forward_removal_control_experiment.sh`, `run_sfz_rerun.sh` | `runners/` (same filenames) |
| `run_freezeout_resnet50_cifar_exact_compare.sh`, `run_cifar100_resnet110_tcresicache_4gpu.sh`, `run_cifar10_final_hardware_metrics_4gpu.sh`, `run_tiny_imagenet_long_boundary_sweep.sh` | `runners/` (same filenames) |

Note: some scripts referenced in the sections above are **not** included in this repository
because they were never archived from the server (server-only at archive time):
`tc_resicache_cifar100_resnet110_lrms.py` (the generated CIFAR-100 RN110 TC-CR script),
`smartfrz_resnet50_cifar.py`, `run_smartfrz_direct_compare.sh`, and the
`run_vit_*` / `run_tiny_*` / `run_table4_*` / `run_ft_rn50_*` suite runners. See
`docs/provenance/artifact_generation_summary.md` for the archive manifest.

Note: `make_tc_cifar100_resnet110.py` (a regex patcher that derived the CIFAR-100 RN110
script from the ResNet-56 script on the server) is intentionally **excluded** — every
archived copy of it is truncated mid-file and cannot run. The CIFAR-100 RN110 TC-CR
training script itself (`tc_resicache_cifar100_resnet110_lrms.py`) was never archived; its
reference outputs are preserved under `results/full_training/cifar100_resnet110/`.
