#!/usr/bin/env python3
"""
TC-ResiCache / ResNet50 / CIFAR-10+CIFAR-100 fine-tuning experiment.

Purpose
-------
Run a TC-ResiCache-style experiment on the same high-level setting used by
SmartFRZ Table 2:

  - ImageNet-pretrained ResNet50
  - CIFAR-10 or CIFAR-100
  - 10 epochs
  - batch size 32
  - input size 224
  - SGD momentum
  - RandomResizedCrop + RandomHorizontalFlip training transform

Default target accuracies are taken from the SmartFRZ paper's full-training
ResNet50 rows:
  CIFAR-10:  96.10%
  CIFAR-100: 81.68%

The script supports three modes:
  baseline      normal fine-tuning
  freeze        freeze a prefix at --freeze_epoch
  tc_resicache  freeze prefix and cache the boundary activation after --cache_start_epoch

Important scientific note
-------------------------
SmartFRZ is a freezing method, not an activation-caching method. For fair
comparison to SmartFRZ, start with baseline and freeze modes. The tc_resicache
mode is useful for testing whether cached reuse can reduce total cost in the
same ResNet50/CIFAR fine-tuning setting, but stochastic RandomResizedCrop means
cached sample-key activations may become stale with respect to augmentation.
Therefore, this script also supports:

  --train_aug smartfrz   # paper-like augmentation, strongest but cache correctness is harder
  --train_aug weak       # RandomCrop(32,pad=4)+resize to 224, still stochastic
  --train_aug deterministic # Resize+CenterCrop; cache-safe sanity check

For paper claims about TC-ResiCache under stochastic augmentation, use this as a
screening script and report cache correctness/augmentation caveats unless you
integrate your full sub-pixel augmentation-consistent reconstruction path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SMARTFRZ_FULL_TARGETS = {
    "cifar10": 96.10,
    "cifar100": 81.68,
}
SMARTFRZ_OURS_TARGETS = {
    "cifar10": 96.12,
    "cifar100": 81.73,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Leave cudnn benchmark on for performance unless strict deterministic requested by env.
    if os.environ.get("STRICT_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx


def make_transforms(dataset: str, train_aug: str, input_size: int, norm: str):
    if norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD

    if train_aug == "smartfrz":
        # This mirrors the uploaded SmartFRZ utils.py: RandomResizedCrop(input_size),
        # RandomHorizontalFlip, ImageNet normalization.
        train_tf = T.Compose([
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif train_aug == "weak":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif train_aug == "deterministic":
        train_tf = T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"unknown train_aug={train_aug}")

    val_tf = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, val_tf


def make_loaders(args):
    train_tf, val_tf = make_transforms(args.dataset, args.train_aug, args.input_size, args.norm)
    if args.dataset == "cifar10":
        cls = torchvision.datasets.CIFAR10
        num_classes = 10
    elif args.dataset == "cifar100":
        cls = torchvision.datasets.CIFAR100
        num_classes = 100
    elif args.dataset == "imagenet_sub":
        # ImageNet subset: classes present in both train/ and val/ of
        # <data_root>/imagenet, first --imagenet_classes synsets in sorted order.
        # Labels are remapped to 0..N-1. Val uses the standard ImageNet
        # Resize(256) + CenterCrop(224) instead of the CIFAR val transform.
        from torchvision.datasets.folder import default_loader

        class _ImageNetSubset(Dataset):
            def __init__(self, root, transform, classes, holdout=0, holdout_mode=False):
                base = torchvision.datasets.ImageFolder(root=root)
                self.transform = transform
                self.loader = default_loader
                c2i = {c: i for i, c in enumerate(sorted(classes))}
                per_class = {}
                for p_, t in base.samples:
                    cname = base.classes[t]
                    if cname in c2i:
                        per_class.setdefault(cname, []).append(p_)
                self.items = []
                for cname in sorted(c2i):
                    paths = sorted(per_class[cname])
                    va = paths[-holdout:] if holdout else []
                    tr = paths[:-holdout] if holdout else paths
                    chosen = va if holdout_mode else tr
                    self.items.extend((p_, c2i[cname]) for p_ in chosen)

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                path, y = self.items[i]
                return self.transform(self.loader(path)), y

        # The on-disk val/ split is incomplete, so validation is a held-out
        # slice of train/: the last --imagenet_val_per_class images per class
        # (sorted by filename), deterministic across methods and seeds.
        tr_root = os.path.join(args.data_root, "imagenet", "train")
        classes = sorted(torchvision.datasets.ImageFolder(root=tr_root).classes)[: args.imagenet_classes]
        val_tf = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(args.input_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        train = _ImageNetSubset(tr_root, train_tf, classes,
                                holdout=args.imagenet_val_per_class, holdout_mode=False)
        val = _ImageNetSubset(tr_root, val_tf, classes,
                              holdout=args.imagenet_val_per_class, holdout_mode=True)
        num_classes = len(classes)
    else:
        raise ValueError("dataset must be cifar10, cifar100, or imagenet_sub")

    if args.dataset != "imagenet_sub":
        train = cls(root=args.data_root, train=True, download=False, transform=train_tf)
        val = cls(root=args.data_root, train=False, download=False, transform=val_tf)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        IndexedDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=g,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        IndexedDataset(val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader, num_classes


def build_model(model_name: str, num_classes: int, weights_mode: str, weights_path: str = "") -> nn.Module:
    model_name = model_name.lower()
    builder = getattr(torchvision.models, model_name, None)
    if builder is None:
        raise ValueError(f"unknown model {model_name}")
    if weights_path:
        model = builder(weights=None)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    elif weights_mode == "default":
        try:
            weight_map = {
                "resnet18": torchvision.models.ResNet18_Weights,
                "resnet34": torchvision.models.ResNet34_Weights,
                "resnet50": torchvision.models.ResNet50_Weights,
                "resnet101": torchvision.models.ResNet101_Weights,
            }
            weight_cls = weight_map[model_name]
            model = builder(weights=weight_cls.DEFAULT)
        except Exception:
            model = builder(pretrained=True)
    elif weights_mode == "none":
        try:
            model = builder(weights=None)
        except Exception:
            model = builder(pretrained=False)
    else:
        raise ValueError("weights must be default, none, or use --weights_path")

    if model_name.startswith("vit"):
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, num_classes)
    else:
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    return model


class ResNetSplit(nn.Module):
    """Wrap torchvision ResNet50 and expose prefix/suffix by named boundary."""

    valid_boundaries = ("stem", "layer1", "layer2", "layer3", "layer4")

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def prefix_modules(self, boundary: str) -> List[nn.Module]:
        m = self.model
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        if boundary == "stem":
            return [stem]
        if boundary == "layer1":
            return [stem, m.layer1]
        if boundary == "layer2":
            return [stem, m.layer1, m.layer2]
        if boundary == "layer3":
            return [stem, m.layer1, m.layer2, m.layer3]
        if boundary == "layer4":
            return [stem, m.layer1, m.layer2, m.layer3, m.layer4]
        raise ValueError(f"invalid boundary {boundary}")

    def suffix_modules(self, boundary: str) -> List[nn.Module]:
        m = self.model
        tail = nn.Sequential(m.avgpool, nn.Flatten(1), m.fc)
        if boundary == "stem":
            return [m.layer1, m.layer2, m.layer3, m.layer4, tail]
        if boundary == "layer1":
            return [m.layer2, m.layer3, m.layer4, tail]
        if boundary == "layer2":
            return [m.layer3, m.layer4, tail]
        if boundary == "layer3":
            return [m.layer4, tail]
        if boundary == "layer4":
            return [tail]
        raise ValueError(f"invalid boundary {boundary}")

    def forward_prefix(self, x: torch.Tensor, boundary: str) -> torch.Tensor:
        out = x
        for mod in self.prefix_modules(boundary):
            out = mod(out)
        return out

    def forward_suffix(self, feat: torch.Tensor, boundary: str) -> torch.Tensor:
        out = feat
        for mod in self.suffix_modules(boundary):
            out = mod(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_prefix(self, boundary: str, eval_prefix: bool = True) -> None:
        for mod in self.prefix_modules(boundary):
            for p in mod.parameters():
                p.requires_grad = False
            if eval_prefix:
                mod.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        return self


# Backward-compatible alias.
ResNet50Split = ResNetSplit


class ViTSplit(nn.Module):
    """Same prefix/suffix interface for torchvision ViT (e.g. vit_b_16).

    Boundary "blockK" = after K transformer encoder blocks. The cached
    feature is the full hidden-state sequence (B, 197, 768).
    """

    valid_boundaries = ("block4", "block8", "block10")

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _k(boundary: str) -> int:
        return int(boundary.replace("block", ""))

    def prefix_modules(self, boundary: str):
        k = self._k(boundary)
        m = self.model
        return [m.conv_proj] + [m.encoder.layers[i] for i in range(k)]

    def suffix_modules(self, boundary: str):
        k = self._k(boundary)
        m = self.model
        return [m.encoder.layers[i] for i in range(k, len(m.encoder.layers))]

    def forward_prefix(self, x: torch.Tensor, boundary: str) -> torch.Tensor:
        m = self.model
        k = self._k(boundary)
        out = m._process_input(x)
        n = out.shape[0]
        out = torch.cat([m.class_token.expand(n, -1, -1), out], dim=1)
        out = m.encoder.dropout(out + m.encoder.pos_embedding)
        for i in range(k):
            out = m.encoder.layers[i](out)
        return out

    def forward_suffix(self, feat: torch.Tensor, boundary: str) -> torch.Tensor:
        m = self.model
        k = self._k(boundary)
        out = feat
        for i in range(k, len(m.encoder.layers)):
            out = m.encoder.layers[i](out)
        out = m.encoder.ln(out)
        return m.heads(out[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_prefix(self, boundary: str, eval_prefix: bool = True) -> None:
        for p in self.model.class_token, self.model.encoder.pos_embedding:
            p.requires_grad = False
        for mod in self.prefix_modules(boundary):
            for p in mod.parameters():
                p.requires_grad = False
            if eval_prefix:
                mod.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        return self


@dataclass
class FeatureBank:
    num_samples: int
    shape: Tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    max_gpu_samples: int
    cpu_bank: Optional[torch.Tensor] = None
    gpu_bank: Optional[torch.Tensor] = None
    valid_cpu: Optional[torch.Tensor] = None
    valid_gpu: Optional[torch.Tensor] = None

    @classmethod
    def create(cls, num_samples: int, feat: torch.Tensor, device: torch.device, dtype: str, max_gpu_samples: int):
        tdtype = torch.float16 if dtype == "fp16" else torch.float32
        shape = tuple(feat.shape[1:])
        obj = cls(num_samples, shape, device, tdtype, max_gpu_samples)
        if max_gpu_samples > 0:
            n = min(max_gpu_samples, num_samples)
            obj.gpu_bank = torch.empty((n, *shape), device=device, dtype=tdtype)
            obj.valid_gpu = torch.zeros(n, device=device, dtype=torch.bool)
        if max_gpu_samples < num_samples:
            obj.cpu_bank = torch.empty((num_samples, *shape), device="cpu", dtype=tdtype, pin_memory=False)
            obj.valid_cpu = torch.zeros(num_samples, dtype=torch.bool)
        return obj

    @property
    def allocated_gb(self) -> float:
        total = 0
        if self.gpu_bank is not None:
            total += self.gpu_bank.numel() * self.gpu_bank.element_size()
        if self.cpu_bank is not None:
            total += self.cpu_bank.numel() * self.cpu_bank.element_size()
        return total / (1024 ** 3)

    def get(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return features and hit mask. CPU bank path copies per-sample and is slower."""
        idx_cpu = idx.detach().cpu().long()
        b = len(idx_cpu)
        out = torch.empty((b, *self.shape), device=self.device, dtype=self.dtype)
        hit = torch.zeros(b, device=self.device, dtype=torch.bool)

        for j, sample_idx in enumerate(idx_cpu.tolist()):
            if self.gpu_bank is not None and sample_idx < self.gpu_bank.shape[0] and bool(self.valid_gpu[sample_idx].item()):
                out[j].copy_(self.gpu_bank[sample_idx])
                hit[j] = True
            elif self.cpu_bank is not None and self.valid_cpu is not None and bool(self.valid_cpu[sample_idx].item()):
                out[j].copy_(self.cpu_bank[sample_idx].to(self.device, non_blocking=True))
                hit[j] = True
        return out, hit

    def put(self, idx: torch.Tensor, feat: torch.Tensor) -> None:
        idx_cpu = idx.detach().cpu().long()
        f = feat.detach().to(dtype=self.dtype)
        for j, sample_idx in enumerate(idx_cpu.tolist()):
            if self.gpu_bank is not None and sample_idx < self.gpu_bank.shape[0]:
                self.gpu_bank[sample_idx].copy_(f[j])
                self.valid_gpu[sample_idx] = True
            elif self.cpu_bank is not None and self.valid_cpu is not None:
                self.cpu_bank[sample_idx].copy_(f[j].detach().cpu())
                self.valid_cpu[sample_idx] = True


def optimizer_for_trainable(model: nn.Module, lr: float, momentum: float, wd: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)


def adjust_lr(optimizer, base_lr: float, epoch: int, epochs: int, scheduler: str):
    if scheduler == "none":
        lr = base_lr
    elif scheduler == "cosine":
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * epoch / max(epochs, 1)))
    elif scheduler == "step":
        lr = base_lr
        if epoch >= int(0.75 * epochs):
            lr *= 0.01
        elif epoch >= int(0.5 * epochs):
            lr *= 0.1
    else:
        raise ValueError(scheduler)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device, amp: bool):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += float(loss.item())
        pred = logits.argmax(1)
        total += y.numel()
        correct += int(pred.eq(y).sum().item())
    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1)


def train_epoch(split: ResNetSplit, loader, optimizer, criterion, device, args, epoch, frozen: bool, bank: Optional[FeatureBank], scaler):
    split.model.train()
    if frozen:
        # Keep prefix in eval once frozen.
        for mod in split.prefix_modules(args.boundary):
            mod.eval()

    loss_sum = 0.0
    total = 0
    correct = 0

    stats = {
        "memory_hits": 0,
        "misses": 0,
        "stored": 0,
        "corrected_reuses": 0,
        "disk_hits": 0,
        "bank_allocated_gb": bank.allocated_gb if bank is not None else 0.0,
    }

    t_prefix = 0.0
    t_suffix = 0.0
    t_bank_get = 0.0
    t_bank_put = 0.0

    for batch_i, (x, y, idx) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_cache = args.mode == "tc_resicache" and frozen and epoch >= args.cache_start_epoch and bank is not None

        if use_cache:
            t0 = time.time()
            cached, hit = bank.get(idx)
            t_bank_get += time.time() - t0
            stats["memory_hits"] += int(hit.sum().item())
            stats["misses"] += int((~hit).sum().item())

            if bool(hit.all().item()):
                feat = cached
            else:
                t0 = time.time()
                with torch.no_grad():
                    fresh = split.forward_prefix(x, args.boundary)
                t_prefix += time.time() - t0
                feat = fresh.detach().to(cached.dtype)
                if bool(hit.any().item()):
                    feat[hit] = cached[hit]
                t0 = time.time()
                bank.put(idx[~hit], fresh[~hit].detach())
                t_bank_put += time.time() - t0
                stats["stored"] += int((~hit).sum().item())

            stats["corrected_reuses"] += int(hit.sum().item())

            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = split.forward_suffix(feat.float(), args.boundary)
                loss = criterion(logits, y)
            t_suffix += time.time() - t0

        elif args.mode in ("freeze", "tc_resicache") and frozen:
            t0 = time.time()
            with torch.no_grad():
                feat = split.forward_prefix(x, args.boundary)
            t_prefix += time.time() - t0
            if args.mode == "tc_resicache" and epoch >= args.cache_start_epoch and bank is not None:
                t1 = time.time()
                bank.put(idx, feat.detach())
                t_bank_put += time.time() - t1
                stats["stored"] += int(x.shape[0])
            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = split.forward_suffix(feat.detach(), args.boundary)
                loss = criterion(logits, y)
            t_suffix += time.time() - t0

        else:
            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = split.model(x)
                loss = criterion(logits, y)
            t_suffix += time.time() - t0

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item())
        pred = logits.detach().argmax(1)
        total += y.numel()
        correct += int(pred.eq(y).sum().item())

        if batch_i % args.log_interval == 0:
            print(
                f"    [Train] epoch={epoch} [{batch_i}/{len(loader)}] "
                f"loss={loss_sum/(batch_i+1):.4f} acc={100.0*correct/max(total,1):.2f} "
                f"frozen={frozen} cache={use_cache} hits={stats['memory_hits']} stored={stats['stored']}",
                flush=True,
            )

    timing = {
        "prefix_sec": t_prefix,
        "suffix_or_full_sec": t_suffix,
        "bank_get_sec": t_bank_get,
        "bank_put_sec": t_bank_put,
    }
    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1), stats, timing


def infer_feature_shape(split, loader, device, boundary):
    split.model.eval()
    x, y, idx = next(iter(loader))
    x = x.to(device, non_blocking=True)
    with torch.no_grad():
        feat = split.forward_prefix(x[:1], boundary)
    return feat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet_sub"], required=True)
    p.add_argument("--data_root", default="/mnt/data2/data")
    p.add_argument("--output_json", required=True)
    p.add_argument("--model", choices=["resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16"], default="resnet50")
    p.add_argument("--imagenet_classes", type=int, default=100)
    p.add_argument("--imagenet_val_per_class", type=int, default=100)

    p.add_argument("--mode", choices=["baseline", "freeze", "tc_resicache"], default="baseline")
    p.add_argument("--boundary", choices=list(ResNetSplit.valid_boundaries) + list(ViTSplit.valid_boundaries), default="layer3")
    p.add_argument("--freeze_epoch", type=int, default=4)
    p.add_argument("--cache_start_epoch", type=int, default=5)

    p.add_argument("--target_acc", type=float, default=None)
    p.add_argument("--target_source", choices=["smartfrz_full", "smartfrz_ours"], default="smartfrz_full")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--train_aug", choices=["smartfrz", "weak", "deterministic"], default="smartfrz")
    p.add_argument("--norm", choices=["imagenet", "cifar"], default="imagenet")

    p.add_argument("--weights", choices=["default", "none"], default="default")
    p.add_argument("--weights_path", default="")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler", choices=["none", "cosine", "step"], default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--bank_dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--gpu_cache_samples", type=int, default=50000,
                   help="Number of sample slots to keep on GPU. If lower than dataset size, rest uses CPU bank.")
    p.add_argument("--disable_cache", action="store_true")
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.target_acc is None:
        targets = SMARTFRZ_FULL_TARGETS if args.target_source == "smartfrz_full" else SMARTFRZ_OURS_TARGETS
        args.target_acc = targets[args.dataset]

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = make_loaders(args)
    model = build_model(args.model, num_classes, args.weights, args.weights_path).to(device)
    split = ViTSplit(model) if args.model.startswith("vit") else ResNetSplit(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_for_trainable(split.model, args.lr, args.momentum, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print("[Main] TC-ResiCache ResNet50 CIFAR fine-tuning")
    print(f"[Main] dataset={args.dataset} classes={num_classes} mode={args.mode} boundary={args.boundary}")
    print(f"[Main] epochs={args.epochs} target={args.target_acc} aug={args.train_aug} norm={args.norm}")
    print(f"[Main] weights={args.weights} lr={args.lr} scheduler={args.scheduler} amp={args.amp}")
    print(f"[Main] batch={args.batch_size} input={args.input_size} seed={args.seed}")

    start = time.time()
    best_acc = 0.0
    time_to_target = None
    epoch_to_target = None
    frozen = False
    bank = None

    for epoch in range(args.epochs):
        lr = adjust_lr(optimizer, args.lr, epoch, args.epochs, args.scheduler)

        if args.mode in ("freeze", "tc_resicache") and (not frozen) and epoch >= args.freeze_epoch:
            print(f"[Freeze] epoch={epoch} boundary={args.boundary}", flush=True)
            split.freeze_prefix(args.boundary, eval_prefix=True)
            optimizer = optimizer_for_trainable(split.model, lr, args.momentum, args.weight_decay)
            frozen = True

        if args.mode == "tc_resicache" and frozen and (bank is None) and (not args.disable_cache) and epoch >= args.cache_start_epoch:
            feat1 = infer_feature_shape(split, train_loader, device, args.boundary)
            bank = FeatureBank.create(
                num_samples=len(train_loader.dataset),
                feat=feat1,
                device=device,
                dtype=args.bank_dtype,
                max_gpu_samples=args.gpu_cache_samples,
            )
            print(f"[Bank] boundary={args.boundary} shape={(len(train_loader.dataset), *feat1.shape[1:])} "
                  f"allocated_gb={bank.allocated_gb:.2f} gpu_cache_samples={args.gpu_cache_samples}", flush=True)

        ep_start = time.time()
        train_loss, train_acc, cache_stats, timing = train_epoch(
            split, train_loader, optimizer, criterion, device, args, epoch, frozen, bank, scaler
        )
        val_loss, val_acc = evaluate(split.model, val_loader, criterion, device, args.amp)
        ep_time = time.time() - ep_start
        elapsed = time.time() - start

        best_acc = max(best_acc, val_acc)
        if time_to_target is None and val_acc >= args.target_acc:
            time_to_target = elapsed
            epoch_to_target = epoch

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": val_loss,
            "test_acc": val_acc,
            "best_test_acc_so_far": best_acc,
            "target_acc": args.target_acc,
            "time_to_target_sec": time_to_target,
            "epoch_to_target": epoch_to_target,
            "elapsed_total_sec": elapsed,
            "time": ep_time,
            "lr": lr,
            "frozen": frozen,
            "boundary": args.boundary,
            "cache_stats": cache_stats,
            "timing": timing,
            "config": vars(args) | {
                "script": "tc_resicache_resnet50_cifar_finetune.py",
                "num_classes": num_classes,
                "note": "SmartFRZ comparison screening run. Cache correctness under stochastic augmentation requires care.",
            },
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[Main] Epoch {epoch} | Test {val_acc:.2f}% | Best {best_acc:.2f}% "
            f"| TTA@{args.target_acc}: {time_to_target} | Time {ep_time:.1f}s "
            f"| frozen={frozen} | hits={cache_stats.get('memory_hits',0)} | stored={cache_stats.get('stored',0)}",
            flush=True,
        )

    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
