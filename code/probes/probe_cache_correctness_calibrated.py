#!/usr/bin/env python3
"""Calibrated cache-correctness probe.

Extends `tc_cache_correctness_probe.py::compute_distance_matrix` in the two
ways Chapter 3 needs:

  1. The 16x16 boundary-activation distance matrix is averaged over N samples
     instead of being taken from a single image (index 42).  The transform
     pool is deterministic and sample-independent, so cell (i, j) means the
     same thing for every sample and averaging is well defined.

  2. A different-image reference is computed, so the off-diagonal entries can
     be calibrated.  Without it "different transforms of the same image are
     nearly as far apart as different images" is an assertion, not a
     measurement.

Both raw L2 and the relative L2 of Equation (3.2) are reported, since the
existing tables use raw L2 while the chapter defines the relative form.

Outputs, per setting, into --out-dir:
    <tag>_rel_l2_mean_16.csv    16x16, relative L2, averaged over samples
    <tag>_raw_l2_mean_16.csv    16x16, raw L2, averaged over samples
    <tag>_summary.json          off-diagonal stats + different-image reference
"""
import argparse
import csv
import json
import os
import random

import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import models
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformPool:
    """Deterministic, sample-independent pool (same semantics as the original)."""

    def __init__(self, input_size=224, resize_size=256, n_transforms=16, seed=42):
        self.input_size, self.resize_size = int(input_size), int(resize_size)
        self.n_transforms = int(n_transforms)
        rng = random.Random(seed)
        max_off = self.resize_size - self.input_size
        self.specs = []
        for tid in range(self.n_transforms):
            top = int(round(max_off * rng.random()))
            left = int(round(max_off * rng.random()))
            flip = bool((tid + rng.randint(0, 1000)) % 2)
            self.specs.append(dict(crop_top=top, crop_left=left,
                                   crop_h=self.input_size, crop_w=self.input_size,
                                   hflip=flip))

    def apply(self, img, tid):
        s = self.specs[int(tid)]
        img = F.resize(img, [self.resize_size, self.resize_size])
        img = F.crop(img, s["crop_top"], s["crop_left"], s["crop_h"], s["crop_w"])
        if s["hflip"]:
            img = F.hflip(img)
        return F.normalize(F.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)


def build_dataset(name, root):
    if name == "cifar10":
        return CIFAR10(root, train=True, download=False)
    if name == "cifar100":
        return CIFAR100(root, train=True, download=False)
    if name == "tinyimagenet":
        return ImageFolder(os.path.join(root, "tiny-imagenet-200", "train"))
    raise ValueError(name)


def build_model(arch, device):
    if arch == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif arch == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(arch)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def boundary_acts(net, boundary, batch):
    """Flattened activations at `boundary` for a batch of views."""
    store = {}
    h = getattr(net, boundary).register_forward_hook(
        lambda m, i, o: store.__setitem__("z", o.detach().float().flatten(1)))
    with torch.no_grad():
        net(batch)
    h.remove()
    return store["z"]


def stats(vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
    return dict(min=min(vals), mean=m, max=max(vals), sd=sd, n=len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["cifar10", "cifar100", "tinyimagenet"])
    ap.add_argument("--arch", required=True, choices=["resnet50", "resnet34"])
    ap.add_argument("--boundary", required=True,
                    choices=["layer1", "layer2", "layer3", "layer4"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--n-transforms", type=int, default=16)
    ap.add_argument("--data-root", default="/mnt/data2/data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    ds = build_dataset(a.dataset, a.data_root)
    pool = TransformPool(n_transforms=a.n_transforms, seed=a.seed)
    net = build_model(a.arch, device)
    K = a.n_transforms

    raw_sum = [[0.0] * K for _ in range(K)]
    rel_sum = [[0.0] * K for _ in range(K)]

    # Different-image reference, accumulated over consecutive sample pairs.
    xi_same_t_rel, xi_any_t_rel, xi_any_t_cos = [], [], []
    same_img_cos = []

    prev = None
    for s in range(a.samples):
        img = ds[s][0]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        views = torch.stack([pool.apply(img, t) for t in range(K)]).to(device)
        Z = boundary_acts(net, a.boundary, views)           # K x D
        nrm = Z.norm(dim=1)

        D = torch.cdist(Z.unsqueeze(0), Z.unsqueeze(0)).squeeze(0)   # K x K raw L2
        R = D / nrm.unsqueeze(0)                                     # relative to z_j
        for i in range(K):
            for j in range(K):
                raw_sum[i][j] += D[i, j].item()
                rel_sum[i][j] += R[i, j].item()
        C = NF.cosine_similarity(Z.unsqueeze(1), Z.unsqueeze(0), dim=2)
        same_img_cos += [C[i, j].item() for i in range(K) for j in range(K) if i != j]

        if prev is not None:
            P, pn = prev
            # different image, same transform id
            d_same = (Z - P).norm(dim=1) / pn
            xi_same_t_rel += d_same.tolist()
            # different image, any transform pair
            Dx = torch.cdist(Z.unsqueeze(0), P.unsqueeze(0)).squeeze(0)
            Rx = Dx / pn.unsqueeze(0)
            Cx = NF.cosine_similarity(Z.unsqueeze(1), P.unsqueeze(0), dim=2)
            xi_any_t_rel += Rx.flatten().tolist()
            xi_any_t_cos += Cx.flatten().tolist()
        prev = (Z, nrm)

        if (s + 1) % 16 == 0:
            print(f"  ...{s + 1}/{a.samples}", flush=True)

    n = float(a.samples)
    raw = [[v / n for v in row] for row in raw_sum]
    rel = [[v / n for v in row] for row in rel_sum]

    for name, M in (("raw_l2", raw), ("rel_l2", rel)):
        path = os.path.join(a.out_dir, f"{a.tag}_{name}_mean_{K}.csv")
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(M)
        print("wrote", path)

    off_raw = [raw[i][j] for i in range(K) for j in range(K) if i != j]
    off_rel = [rel[i][j] for i in range(K) for j in range(K) if i != j]

    summary = dict(
        tag=a.tag, dataset=a.dataset, arch=a.arch, boundary=a.boundary,
        n_samples=a.samples, n_transforms=K, seed=a.seed,
        diag_raw_l2_max=max(raw[i][i] for i in range(K)),
        offdiag_raw_l2=stats(off_raw),
        offdiag_rel_l2=stats(off_rel),
        same_image_cosine=stats(same_img_cos),
        diff_image_same_transform_rel_l2=stats(xi_same_t_rel),
        diff_image_any_transform_rel_l2=stats(xi_any_t_rel),
        diff_image_any_transform_cosine=stats(xi_any_t_cos),
    )
    sp = os.path.join(a.out_dir, f"{a.tag}_summary.json")
    json.dump(summary, open(sp, "w"), indent=2)
    print("wrote", sp)
    print(json.dumps({k: v for k, v in summary.items()
                      if k.startswith(("offdiag", "diff_image", "same_image"))}, indent=2))


if __name__ == "__main__":
    main()
