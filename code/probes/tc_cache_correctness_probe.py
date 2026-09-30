#!/usr/bin/env python3
"""
TC-ResiCache cache-correctness probe.

Purpose:
  Verify that sample-only activation caching is unsafe under stochastic
  augmentation, while transform-consistent caching is safe.

What this script does:
  1. Loads CIFAR-10 or CIFAR-100.
  2. Applies deterministic but stochastic-like transform instances:
       Resize -> crop at transform-specific coordinates -> optional flip.
  3. Extracts real ResNet50 boundary activations.
  4. Simulates two cache policies:
       A. sample_only: key = sample index
       B. transform_consistent: key = (sample index, transform id)
  5. Computes stale-hit rate and activation mismatch distances.

This is a diagnostic/proof experiment, not a training-speed benchmark.
It intentionally recomputes the current activation every time so we can
verify whether a cache hit would be correct.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash_int(a: int, b: int, c: int) -> int:
    """
    Deterministic integer hash, independent of Python's randomized hash().
    """
    x = (a + 0x9E3779B9) & 0xFFFFFFFF
    x ^= ((b + 0x85EBCA6B) * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= ((c + 0x27D4EB2F) * 0x165667B1) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return int(x & 0xFFFFFFFF)


class TransformPool:
    """
    Finite transform pool.

    We use a fixed set of crop/flip transform instances so the cache gate
    can see both hits and misses during the probe. Each transform id maps
    to a specific crop coordinate and flip flag.
    """

    def __init__(self, input_size: int = 224, resize_size: int = 256,
                 n_transforms: int = 16, seed: int = 42):
        self.input_size = int(input_size)
        self.resize_size = int(resize_size)
        self.n_transforms = int(n_transforms)

        rng = random.Random(seed)
        max_off = self.resize_size - self.input_size
        if max_off < 0:
            raise ValueError("resize_size must be >= input_size")

        self.specs = []
        for tid in range(self.n_transforms):
            # Use deterministic but varied crop positions and flip flags.
            top = int(round(max_off * rng.random()))
            left = int(round(max_off * rng.random()))
            flip = bool((tid + rng.randint(0, 1000)) % 2)
            spec = {
                "tid": tid,
                "resize_size": self.resize_size,
                "crop_top": top,
                "crop_left": left,
                "crop_h": self.input_size,
                "crop_w": self.input_size,
                "hflip": flip,
            }
            self.specs.append(spec)

    def signature(self, tid: int) -> str:
        s = self.specs[int(tid)]
        return (
            f"resize={s['resize_size']};"
            f"crop=({s['crop_top']},{s['crop_left']},{s['crop_h']},{s['crop_w']});"
            f"hflip={int(s['hflip'])}"
        )

    def apply(self, img: Image.Image, tid: int) -> torch.Tensor:
        s = self.specs[int(tid)]

        img = F.resize(img, [s["resize_size"], s["resize_size"]])
        img = F.crop(img, s["crop_top"], s["crop_left"], s["crop_h"], s["crop_w"])

        if s["hflip"]:
            img = F.hflip(img)

        x = F.to_tensor(img)
        x = F.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        return x


class IndexedTransformDataset(Dataset):
    def __init__(self, base, transform_pool: TransformPool, seed: int):
        self.base = base
        self.pool = transform_pool
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def transform_id(self, sample_index: int) -> int:
        h = stable_hash_int(int(sample_index), int(self.epoch), int(self.seed))
        return h % self.pool.n_transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL image before transform.")

        tid = self.transform_id(idx)
        x = self.pool.apply(img, tid)
        sig = self.pool.signature(tid)

        return x, int(y), int(idx), int(tid), sig


def build_dataset(args):
    if args.dataset == "cifar10":
        base = torchvision.datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=True,
            transform=None,
        )
    elif args.dataset == "cifar100":
        base = torchvision.datasets.CIFAR100(
            root=args.data_root,
            train=True,
            download=True,
            transform=None,
        )
    else:
        raise ValueError(args.dataset)

    pool = TransformPool(
        input_size=args.input_size,
        resize_size=args.resize_size,
        n_transforms=args.n_transforms,
        seed=args.seed,
    )

    ds = IndexedTransformDataset(base, pool, args.seed)

    if args.n_samples > 0 and args.n_samples < len(ds):
        indices = list(range(args.n_samples))
        ds = Subset(ds, indices)
        # Subset hides set_epoch, so attach helper.
        ds.set_epoch = lambda epoch: ds.dataset.set_epoch(epoch)
        ds.pool = ds.dataset.pool

    return ds, pool


def build_model(device):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def prefix_forward(model, x, boundary: str):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    if boundary == "layer1":
        return x

    x = model.layer2(x)
    if boundary == "layer2":
        return x

    x = model.layer3(x)
    if boundary == "layer3":
        return x

    x = model.layer4(x)
    if boundary == "layer4":
        return x

    raise ValueError(f"Unknown boundary: {boundary}")


def tensor_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.float() - b.float()).item())


def summarize_values(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "std": None,
            "max": None,
        }
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "max": float(arr.max()),
    }


@torch.no_grad()
def compute_distance_matrix(model, base_dataset, pool, sample_index: int,
                            boundary: str, device, output_csv: Path):
    img, _ = base_dataset[sample_index]
    acts = []

    for tid in range(pool.n_transforms):
        x = pool.apply(img, tid).unsqueeze(0).to(device)
        z = prefix_forward(model, x, boundary).detach().cpu().float().view(-1)
        acts.append(z)

    n = len(acts)
    mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            mat[i, j] = float(torch.linalg.vector_norm(acts[i] - acts[j]).item())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"T{j}" for j in range(n)])
        for i in range(n):
            w.writerow([f"T{i}"] + [f"{mat[i, j]:.6f}" for j in range(n)])

    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag.append(mat[i, j])

    return mat, summarize_values(off_diag)


@torch.no_grad()
def run_probe(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds, pool = build_dataset(args)

    # For distance matrix, use the original unwrapped base dataset.
    if isinstance(ds, Subset):
        base_dataset = ds.dataset.base
    else:
        base_dataset = ds.base

    model = build_model(device)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample_only_cache: Dict[int, Tuple[int, torch.Tensor]] = {}
    tc_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    sample_only_hits = 0
    sample_only_misses = 0
    sample_only_stale_hits = 0
    sample_only_correct_hits = 0
    sample_only_hit_l2 = []
    sample_only_stale_l2 = []
    sample_only_correct_l2 = []

    tc_hits = 0
    tc_misses = 0
    tc_stale_hits = 0
    tc_hit_l2 = []

    records_csv = outdir / "cache_probe_events.csv"
    with open(records_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "epoch",
                "sample_index",
                "transform_id",
                "transform_signature",
                "sample_only_hit",
                "sample_only_cached_tid",
                "sample_only_stale",
                "sample_only_l2",
                "tc_hit",
                "tc_l2",
            ],
        )
        writer.writeheader()

        t_start = time.perf_counter()

        for epoch in range(args.n_epochs):
            ds.set_epoch(epoch)

            for x, y, idx, tid, sig in loader:
                x = x.to(device, non_blocking=True)
                idx_list = [int(v) for v in idx.tolist()]
                tid_list = [int(v) for v in tid.tolist()]

                z_batch = prefix_forward(model, x, args.boundary)
                z_batch_cpu = z_batch.detach().cpu().half()

                for b, sample_index in enumerate(idx_list):
                    current_tid = tid_list[b]
                    current_z = z_batch_cpu[b]

                    # Sample-only cache probe.
                    so_hit = sample_index in sample_only_cache
                    so_cached_tid = ""
                    so_stale = False
                    so_l2 = ""

                    if so_hit:
                        sample_only_hits += 1
                        cached_tid, cached_z = sample_only_cache[sample_index]
                        so_cached_tid = cached_tid
                        so_l2_value = tensor_l2(cached_z, current_z)
                        so_l2 = so_l2_value
                        sample_only_hit_l2.append(so_l2_value)

                        if cached_tid != current_tid:
                            sample_only_stale_hits += 1
                            so_stale = True
                            sample_only_stale_l2.append(so_l2_value)
                        else:
                            sample_only_correct_hits += 1
                            sample_only_correct_l2.append(so_l2_value)
                    else:
                        sample_only_misses += 1
                        sample_only_cache[sample_index] = (current_tid, current_z.clone())

                    # Transform-consistent cache probe.
                    key = (sample_index, current_tid)
                    tc_hit = key in tc_cache
                    tc_l2 = ""

                    if tc_hit:
                        tc_hits += 1
                        cached_z = tc_cache[key]
                        tc_l2_value = tensor_l2(cached_z, current_z)
                        tc_l2 = tc_l2_value
                        tc_hit_l2.append(tc_l2_value)

                        # This should never happen because the key includes tid.
                        if tc_l2_value > args.tc_l2_tolerance:
                            tc_stale_hits += 1
                    else:
                        tc_misses += 1
                        tc_cache[key] = current_z.clone()

                    if args.write_event_csv:
                        writer.writerow({
                            "epoch": epoch,
                            "sample_index": sample_index,
                            "transform_id": current_tid,
                            "transform_signature": sig[b],
                            "sample_only_hit": int(so_hit),
                            "sample_only_cached_tid": so_cached_tid,
                            "sample_only_stale": int(so_stale),
                            "sample_only_l2": so_l2,
                            "tc_hit": int(tc_hit),
                            "tc_l2": tc_l2,
                        })

        elapsed = time.perf_counter() - t_start

    dist_csv = outdir / "activation_distance_matrix_sample.csv"
    _, offdiag_stats = compute_distance_matrix(
        model=model,
        base_dataset=base_dataset,
        pool=pool,
        sample_index=args.distance_sample_index,
        boundary=args.boundary,
        device=device,
        output_csv=dist_csv,
    )

    sample_only_stale_rate = (
        sample_only_stale_hits / sample_only_hits if sample_only_hits else 0.0
    )
    sample_only_hit_rate = (
        sample_only_hits / (sample_only_hits + sample_only_misses)
        if (sample_only_hits + sample_only_misses) else 0.0
    )
    tc_hit_rate = (
        tc_hits / (tc_hits + tc_misses)
        if (tc_hits + tc_misses) else 0.0
    )

    summary = {
        "dataset": args.dataset,
        "model": "resnet50",
        "boundary": args.boundary,
        "device": str(device),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_epochs": args.n_epochs,
        "n_transforms": args.n_transforms,
        "input_size": args.input_size,
        "resize_size": args.resize_size,
        "batch_size": args.batch_size,
        "elapsed_sec": elapsed,

        "sample_only_hits": sample_only_hits,
        "sample_only_misses": sample_only_misses,
        "sample_only_hit_rate": sample_only_hit_rate,
        "sample_only_stale_hits": sample_only_stale_hits,
        "sample_only_correct_hits": sample_only_correct_hits,
        "sample_only_stale_rate_among_hits": sample_only_stale_rate,
        "sample_only_hit_l2": summarize_values(sample_only_hit_l2),
        "sample_only_stale_l2": summarize_values(sample_only_stale_l2),
        "sample_only_correct_l2": summarize_values(sample_only_correct_l2),

        "tc_hits": tc_hits,
        "tc_misses": tc_misses,
        "tc_hit_rate": tc_hit_rate,
        "tc_stale_hits_by_l2_tolerance": tc_stale_hits,
        "tc_stale_rate_among_hits": (tc_stale_hits / tc_hits if tc_hits else 0.0),
        "tc_hit_l2": summarize_values(tc_hit_l2),

        "single_sample_offdiag_distance": offdiag_stats,
        "transform_specs": pool.specs,
        "records_csv": str(records_csv),
        "distance_matrix_csv": str(dist_csv),
    }

    summary_json = outdir / "cache_correctness_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_csv = outdir / "cache_correctness_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        flat = {
            "dataset": summary["dataset"],
            "model": summary["model"],
            "boundary": summary["boundary"],
            "n_samples": summary["n_samples"],
            "n_epochs": summary["n_epochs"],
            "n_transforms": summary["n_transforms"],
            "sample_only_hit_rate": summary["sample_only_hit_rate"],
            "sample_only_stale_rate_among_hits": summary["sample_only_stale_rate_among_hits"],
            "sample_only_stale_l2_mean": summary["sample_only_stale_l2"]["mean"],
            "sample_only_stale_l2_min": summary["sample_only_stale_l2"]["min"],
            "sample_only_stale_l2_max": summary["sample_only_stale_l2"]["max"],
            "tc_hit_rate": summary["tc_hit_rate"],
            "tc_stale_rate_among_hits": summary["tc_stale_rate_among_hits"],
            "tc_hit_l2_max": summary["tc_hit_l2"]["max"],
            "offdiag_l2_mean": summary["single_sample_offdiag_distance"]["mean"],
            "offdiag_l2_min": summary["single_sample_offdiag_distance"]["min"],
            "offdiag_l2_max": summary["single_sample_offdiag_distance"]["max"],
            "elapsed_sec": summary["elapsed_sec"],
        }
        for k, v in flat.items():
            w.writerow([k, v])

    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {summary_json}")
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {dist_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--data-root", default="/mnt/data2/data")
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--boundary", choices=["layer1", "layer2", "layer3", "layer4"], required=True)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--n-epochs", type=int, default=6)
    ap.add_argument("--n-transforms", type=int, default=16)

    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--resize-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    ap.add_argument("--distance-sample-index", type=int, default=42)
    ap.add_argument("--tc-l2-tolerance", type=float, default=1e-5)
    ap.add_argument("--write-event-csv", action="store_true")

    args = ap.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
