#!/usr/bin/env python3
"""
SmartFRZ-style dynamic layer freezing for ResNet50 CIFAR fine-tuning.

This is a reimplementation of the high-level idea from:
  Li et al., "SmartFRZ: An Efficient Training Framework using Attention-Based
  Layer Freezing", ICLR 2023.

The paper trains an attention-based predictor offline using CKA-based labels.
For direct comparison with TC-ResiCache, this script implements an online
gradient-history attention predictor that decides which layer groups to freeze
and when. It is intentionally compatible with the TC-ResiCache fine-tuning
script so that baseline / freeze / smartfrz / tc_resicache can be compared on
the same testbed.

Suggested setting (matches SmartFRZ Table 2 and the TC-ResiCache SmartFRZ
comparison):
  - ImageNet-pretrained ResNet50
  - CIFAR-10 or CIFAR-100
  - 10 epochs
  - batch size 32
  - SGD momentum, cosine LR schedule
  - RandomResizedCrop(224) + RandomHorizontalFlip
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

SMARTFRZ_TARGETS = {
    "cifar10": 96.10,
    "cifar100": 81.68,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        # 100-class ImageNet subset; validation is a deterministic held-out
        # slice of train/ (last 100 images per class), identical across methods.
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
                    self.items.extend((p_, c2i[cname]) for p_ in (va if holdout_mode else tr))

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                path, y = self.items[i]
                return self.transform(self.loader(path)), y

        tr_root = os.path.join(args.data_root, "imagenet", "train")
        classes = sorted(torchvision.datasets.ImageFolder(root=tr_root).classes)[:100]
        val_tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                            T.CenterCrop(args.input_size), T.ToTensor(),
                            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        train = _ImageNetSubset(tr_root, train_tf, classes, holdout=100, holdout_mode=False)
        val = _ImageNetSubset(tr_root, val_tf, classes, holdout=100, holdout_mode=True)
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


def build_resnet50(num_classes: int, weights_mode: str, weights_path: str = "") -> nn.Module:
    if weights_path:
        model = torchvision.models.resnet50(weights=None)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    elif weights_mode == "default":
        try:
            from torchvision.models import ResNet50_Weights
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            model = torchvision.models.resnet50(pretrained=True)
    elif weights_mode == "none":
        try:
            model = torchvision.models.resnet50(weights=None)
        except Exception:
            model = torchvision.models.resnet50(pretrained=False)
    else:
        raise ValueError("weights must be default, none, or use --weights_path")

    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model


class ResNet50Split(nn.Module):
    """Wrap torchvision ResNet50 and expose layer groups for SmartFRZ."""

    # Order from low-level to high-level features.
    valid_boundaries = ("stem", "layer1", "layer2", "layer3", "layer4", "fc")

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.tail = nn.Sequential(model.avgpool, nn.Flatten(1), model.fc)

    def layer_group(self, name: str) -> nn.Module:
        if name == "stem":
            return self.stem
        if name == "layer1":
            return self.model.layer1
        if name == "layer2":
            return self.model.layer2
        if name == "layer3":
            return self.model.layer3
        if name == "layer4":
            return self.model.layer4
        if name == "fc":
            return self.tail
        raise ValueError(f"invalid layer group {name}")

    def layer_groups_in_order(self) -> List[str]:
        return list(self.valid_boundaries)

    def prefix_modules(self, boundary: str) -> List[nn.Module]:
        mods = []
        for name in self.valid_boundaries:
            mods.append(self.layer_group(name))
            if name == boundary:
                break
        return mods

    def suffix_modules(self, boundary: str) -> List[nn.Module]:
        mods = []
        passed = False
        for name in self.valid_boundaries:
            if passed:
                mods.append(self.layer_group(name))
            if name == boundary:
                passed = True
        return mods

    def forward_prefix(self, x: torch.Tensor, boundary: str) -> torch.Tensor:
        out = x
        for name in self.valid_boundaries:
            out = self.layer_group(name)(out)
            if name == boundary:
                break
        return out

    def forward_suffix(self, feat: torch.Tensor, boundary: str) -> torch.Tensor:
        out = feat
        for mod in self.suffix_modules(boundary):
            out = mod(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_group(self, name: str, eval_mode: bool = True) -> None:
        for p in self.layer_group(name).parameters():
            p.requires_grad = False
        if eval_mode:
            self.layer_group(name).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        return self


class AttentionStability(nn.Module):
    """
    Lightweight attention over a layer's recent gradient-norm history.

    The paper uses attention over sampled weight vectors; here we use gradient
    norms as a lightweight proxy that captures the same intuition: weight
    history contains noisy steps, and attention lets the predictor focus on
    the most informative steps when deciding convergence.
    """

    def __init__(self, window: int, hidden: int = 64):
        super().__init__()
        self.window = window
        self.hidden = hidden
        # Input: per-step gradient norm + normalized position.
        self.mlp_k = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mlp_q = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mlp_v = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (window,) tensor of gradient norms.  May contain NaNs
                     for steps before the window is full.
        Returns:
            score: scalar freeze probability (higher = more converged).
        """
        # Replace leading NaNs with the first valid value, then normalize.
        valid = ~torch.isnan(history)
        if valid.any():
            first_valid = history[valid.nonzero(as_tuple=True)[0][0]]
        else:
            first_valid = torch.tensor(0.0, device=history.device, dtype=history.dtype)
        h = torch.where(valid, history, first_valid)

        mean = h.mean().detach().clamp_min(1e-8)
        std = h.std().detach().clamp_min(1e-8)
        h_norm = (h - mean) / std

        positions = torch.linspace(0.0, 1.0, self.window, device=h.device, dtype=h.dtype)
        x = torch.stack([h_norm, positions], dim=-1)  # (window, 2)

        K = self.mlp_k(x)  # (window, hidden)
        Q = self.mlp_q(x)  # (window, hidden)
        V = self.mlp_v(x)  # (window, hidden)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden)
        # Mask invalid positions so they do not contribute.
        mask = ~valid
        if mask.any():
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (window, window)
        # Zero out attention rows for invalid query positions.
        if mask.any():
            attn = attn.masked_fill(mask.unsqueeze(1), 0.0)
        ctx = torch.matmul(attn, V).mean(dim=0)  # (hidden,)
        return torch.sigmoid(self.classifier(ctx).squeeze(0))


@dataclass
class SmartFRZState:
    """Tracks gradient history and frozen status for each layer group."""

    layer_names: List[str]
    window: int
    device: torch.device
    freeze_threshold: float
    cv_threshold: float
    relative_grad_threshold: float
    convergence_mode: str
    patience: int
    min_epoch: int
    eval_frozen: bool
    max_freeze_idx: int = field(default=-1)
    history: Dict[str, List[float]] = field(init=False)
    max_grad: Dict[str, float] = field(init=False)
    frozen: Dict[str, bool] = field(init=False)
    frozen_since_epoch: Dict[str, int] = field(init=False)
    predictor: AttentionStability = field(init=False)
    frozen_count: int = 0

    def __post_init__(self):
        self.history = {n: [] for n in self.layer_names}
        self.max_grad = {n: 0.0 for n in self.layer_names}
        self.frozen = {n: False for n in self.layer_names}
        self.frozen_since_epoch = {}
        self.predictor = AttentionStability(window=self.window).to(self.device)

    def update(self, split: ResNet50Split):
        for name in self.layer_names:
            if self.frozen[name]:
                continue
            grp = split.layer_group(name)
            total_norm = 0.0
            count = 0
            for p in grp.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.norm(2).item())
                    count += 1
            avg_norm = total_norm / max(count, 1)
            self.history[name].append(avg_norm)
            self.max_grad[name] = max(self.max_grad[name], avg_norm)
            # Keep only the last window entries.
            if len(self.history[name]) > self.window:
                self.history[name].pop(0)

    def decide_freezing(self, epoch: int) -> List[str]:
        """Return list of layer groups to freeze at this check point."""
        # Do not freeze before the minimum epoch.
        if epoch < self.min_epoch:
            return []

        to_freeze = []
        diagnostics = []
        for idx, name in enumerate(self.layer_names):
            # In combined SmartFRZ+TC mode, do not freeze beyond the TC cache boundary.
            if self.max_freeze_idx >= 0 and idx > self.max_freeze_idx:
                continue
            if self.frozen[name] or len(self.history[name]) < 2:
                continue

            # Always keep the final classifier trainable.
            if name == "fc":
                continue

            hist_list = self.history[name]
            arr = np.array(hist_list, dtype=np.float64)
            mean = arr.mean()
            std = arr.std()
            cv = std / max(mean, 1e-8)
            score_cv = 1.0 - cv
            max_grad = max(self.max_grad[name], 1e-8)
            score_rg = mean / max_grad  # lower = more converged

            diagnostics.append(
                f"{name}:cv={cv:.3f} score_cv={score_cv:.3f} rel_grad={score_rg:.3f}"
            )

            # Determine if this layer meets the convergence criterion.
            if self.convergence_mode == "cv":
                hit = score_cv >= (1.0 - self.cv_threshold)
            elif self.convergence_mode == "relative_grad":
                hit = score_rg <= self.relative_grad_threshold
            else:
                hit = False

            # Bottom-up freezing: a layer can only be frozen if all preceding
            # trainable layers are already frozen.  This prevents freezing the
            # whole network at once and matches SmartFRZ's observed behavior.
            predecessors = self.layer_names[:idx]
            can_freeze = all(self.frozen.get(p, False) for p in predecessors)

            recent_scores = getattr(self, "_recent_scores", {})
            if hit and can_freeze:
                recent_scores[name] = recent_scores.get(name, 0) + 1
                if recent_scores[name] >= self.patience:
                    to_freeze.append(name)
            else:
                recent_scores[name] = 0
            setattr(self, "_recent_scores", recent_scores)

        print(f"[SmartFRZ] epoch={epoch} diagnostics={' | '.join(diagnostics)}", flush=True)
        return to_freeze

    def freeze(self, name: str, split: ResNet50Split, epoch: int):
        if self.frozen[name]:
            return
        split.freeze_group(name, eval_mode=self.eval_frozen)
        self.frozen[name] = True
        self.frozen_since_epoch[name] = epoch
        self.frozen_count += 1
        print(f"[SmartFRZ] epoch={epoch} frozen={name} count={self.frozen_count}", flush=True)


class FeatureBank:
    """Minimal CPU/GPU activation bank used for tc_resicache mode."""

    def __init__(self, num_samples: int, feat: torch.Tensor, device: torch.device, dtype: torch.dtype, max_gpu_samples: int):
        self.shape = tuple(feat.shape[1:])
        self.device = device
        self.dtype = dtype
        self.max_gpu_samples = max_gpu_samples
        self.gpu_bank = None
        self.valid_gpu = None
        self.cpu_bank = None
        self.valid_cpu = None
        if max_gpu_samples > 0:
            n = min(max_gpu_samples, num_samples)
            self.gpu_bank = torch.empty((n, *self.shape), device=device, dtype=dtype)
            self.valid_gpu = torch.zeros(n, device=device, dtype=torch.bool)
        if max_gpu_samples < num_samples:
            self.cpu_bank = torch.empty((num_samples, *self.shape), device="cpu", dtype=dtype)
            self.valid_cpu = torch.zeros(num_samples, dtype=torch.bool)

    @property
    def allocated_gb(self) -> float:
        total = 0
        if self.gpu_bank is not None:
            total += self.gpu_bank.numel() * self.gpu_bank.element_size()
        if self.cpu_bank is not None:
            total += self.cpu_bank.numel() * self.cpu_bank.element_size()
        return total / (1024 ** 3)

    def get(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_cpu = idx.detach().cpu().long()
        b = len(idx_cpu)
        out = torch.empty((b, *self.shape), device=self.device, dtype=self.dtype)
        hit = torch.zeros(b, device=self.device, dtype=torch.bool)
        for j, sample_idx in enumerate(idx_cpu.tolist()):
            if self.gpu_bank is not None and sample_idx < self.gpu_bank.shape[0] and bool(self.valid_gpu[sample_idx].item()):
                out[j].copy_(self.gpu_bank[sample_idx])
                hit[j] = True
            elif self.cpu_bank is not None and self.valid_cpu is not None and bool(self.valid_cpu[sample_idx].item()):
                out[j].copy_(self.cpu_bank[sample_idx].to(self.device))
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


def train_epoch(
    split: ResNet50Split,
    loader,
    optimizer,
    criterion,
    device,
    args,
    epoch: int,
    smartfrz: Optional[SmartFRZState],
    bank: Optional[FeatureBank],
    scaler,
):
    split.model.train()
    # Keep frozen prefix/layers in eval mode (important for BN).
    if smartfrz is not None:
        for name in smartfrz.layer_names:
            if smartfrz.frozen.get(name):
                split.layer_group(name).eval()
    if args.mode in ("freeze", "tc_resicache", "smartfrz_tc") and args.boundary:
        for mod in split.prefix_modules(args.boundary):
            mod.eval()

    loss_sum = 0.0
    total = 0
    correct = 0
    cache_stats = {"memory_hits": 0, "misses": 0, "stored": 0, "bank_allocated_gb": bank.allocated_gb if bank else 0.0}

    for batch_i, (x, y, idx) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_cache = args.mode in ("tc_resicache", "smartfrz_tc") and bank is not None and epoch >= args.cache_start_epoch
        if use_cache:
            cached, hit = bank.get(idx)
            cache_stats["memory_hits"] += int(hit.sum().item())
            cache_stats["misses"] += int((~hit).sum().item())
            if hit.all():
                feat = cached
            else:
                with torch.no_grad():
                    fresh = split.forward_prefix(x, args.boundary)
                feat = fresh.detach().to(cached.dtype)
                if hit.any():
                    feat[hit] = cached[hit]
                bank.put(idx[~hit], fresh[~hit].detach())
                cache_stats["stored"] += int((~hit).sum().item())
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = split.forward_suffix(feat.float(), args.boundary)
                loss = criterion(logits, y)
        else:
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = split.model(x)
                loss = criterion(logits, y)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update SmartFRZ gradient history after backward.
        if smartfrz is not None and args.mode in ("smartfrz", "smartfrz_tc"):
            smartfrz.update(split)

        loss_sum += float(loss.item())
        pred = logits.detach().argmax(1)
        total += y.numel()
        correct += int(pred.eq(y).sum().item())

        if batch_i % args.log_interval == 0:
            print(
                f"    [Train] epoch={epoch} [{batch_i}/{len(loader)}] "
                f"loss={loss_sum/(batch_i+1):.4f} acc={100.0*correct/max(total,1):.2f} "
                f"mode={args.mode}",
                flush=True,
            )

    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1), cache_stats


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

    p.add_argument("--mode", choices=["baseline", "freeze", "smartfrz", "tc_resicache", "smartfrz_tc"], default="baseline")
    p.add_argument("--boundary", choices=["stem", "layer1", "layer2", "layer3", "layer4"], default="layer3")
    p.add_argument("--freeze_epoch", type=int, default=4)
    p.add_argument("--cache_start_epoch", type=int, default=5)

    p.add_argument("--target_acc", type=float, default=None)
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
    p.add_argument("--scheduler", choices=["none", "cosine", "step"], default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")

    # SmartFRZ-specific.
    p.add_argument("--sfz_window", type=int, default=30, help="Gradient history window size.")
    p.add_argument("--sfz_check_every", type=int, default=1, help="Check freezing every N epochs.")
    p.add_argument("--sfz_convergence_mode", choices=["cv", "relative_grad"], default="relative_grad",
                   help="Convergence criterion: cv=coefficient-of-variation, relative_grad=grad norm vs max.")
    p.add_argument("--sfz_freeze_threshold", type=float, default=0.70, help="Freeze probability threshold (attention mode).")
    p.add_argument("--sfz_cv_threshold", type=float, default=0.25, help="Coefficient-of-variation threshold (cv mode).")
    p.add_argument("--sfz_relative_grad_threshold", type=float, default=0.20,
                   help="Freeze when current grad norm is below this fraction of the max observed (relative_grad mode).")
    p.add_argument("--sfz_patience", type=int, default=1, help="Consecutive checks above threshold.")
    p.add_argument("--sfz_min_epoch", type=int, default=2, help="Do not freeze any layer before this epoch.")
    p.add_argument("--sfz_eval_frozen", type=int, default=1, help="Set frozen layers to eval mode.")

    # TC-ResiCache-specific.
    p.add_argument("--bank_dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--gpu_cache_samples", type=int, default=50000)
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.target_acc is None:
        args.target_acc = SMARTFRZ_TARGETS[args.dataset]

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = make_loaders(args)
    model = build_resnet50(num_classes, args.weights, args.weights_path).to(device)
    split = ResNet50Split(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_for_trainable(split.model, args.lr, args.momentum, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    smartfrz = None
    bank = None
    if args.mode == "smartfrz":
        smartfrz = SmartFRZState(
            layer_names=split.layer_groups_in_order(),
            window=args.sfz_window,
            device=device,
            freeze_threshold=args.sfz_freeze_threshold,
            cv_threshold=args.sfz_cv_threshold,
            relative_grad_threshold=args.sfz_relative_grad_threshold,
            convergence_mode=args.sfz_convergence_mode,
            patience=args.sfz_patience,
            min_epoch=args.sfz_min_epoch,
            eval_frozen=bool(args.sfz_eval_frozen),
        )
    elif args.mode == "smartfrz_tc":
        boundary_idx = split.layer_groups_in_order().index(args.boundary)
        smartfrz = SmartFRZState(
            layer_names=split.layer_groups_in_order(),
            window=args.sfz_window,
            device=device,
            freeze_threshold=args.sfz_freeze_threshold,
            cv_threshold=args.sfz_cv_threshold,
            relative_grad_threshold=args.sfz_relative_grad_threshold,
            convergence_mode=args.sfz_convergence_mode,
            patience=args.sfz_patience,
            min_epoch=args.sfz_min_epoch,
            eval_frozen=bool(args.sfz_eval_frozen),
            max_freeze_idx=boundary_idx,
        )
    elif args.mode == "freeze":
        pass
    elif args.mode == "tc_resicache":
        pass

    print(f"[Main] mode={args.mode} dataset={args.dataset} epochs={args.epochs} target={args.target_acc}")
    print(f"[Main] lr={args.lr} scheduler={args.scheduler} amp={args.amp} seed={args.seed}")

    start = time.time()
    best_acc = 0.0
    time_to_target = None
    epoch_to_target = None

    for epoch in range(args.epochs):
        lr = adjust_lr(optimizer, args.lr, epoch, args.epochs, args.scheduler)

        if args.mode == "freeze" and epoch >= args.freeze_epoch:
            # Freeze prefix up to boundary.
            for name in split.layer_groups_in_order():
                if name == args.boundary:
                    split.freeze_group(name, eval_mode=True)
                    break
                else:
                    split.freeze_group(name, eval_mode=True)
            optimizer = optimizer_for_trainable(split.model, lr, args.momentum, args.weight_decay)

        if args.mode in ("tc_resicache", "smartfrz_tc") and epoch >= args.freeze_epoch and bank is None:
            for name in split.layer_groups_in_order():
                if name == args.boundary:
                    split.freeze_group(name, eval_mode=True)
                    break
                else:
                    split.freeze_group(name, eval_mode=True)
            optimizer = optimizer_for_trainable(split.model, lr, args.momentum, args.weight_decay)

        if args.mode in ("tc_resicache", "smartfrz_tc") and bank is None and epoch >= args.cache_start_epoch:
            feat1 = infer_feature_shape(split, train_loader, device, args.boundary)
            bank = FeatureBank(
                num_samples=len(train_loader.dataset),
                feat=feat1,
                device=device,
                dtype=torch.float16 if args.bank_dtype == "fp16" else torch.float32,
                max_gpu_samples=args.gpu_cache_samples,
            )
            print(f"[Bank] allocated_gb={bank.allocated_gb:.2f}", flush=True)

        ep_start = time.time()
        train_loss, train_acc, cache_stats = train_epoch(
            split, train_loader, optimizer, criterion, device, args, epoch, smartfrz, bank, scaler
        )

        # SmartFRZ freezing decision after each epoch (or every sfz_check_every epochs).
        if smartfrz is not None and (epoch + 1) % args.sfz_check_every == 0:
            to_freeze = smartfrz.decide_freezing(epoch)
            for name in to_freeze:
                smartfrz.freeze(name, split, epoch)
                optimizer = optimizer_for_trainable(split.model, lr, args.momentum, args.weight_decay)

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
            "mode": args.mode,
            "cache_stats": cache_stats,
            "config": vars(args) | {
                "script": "smartfrz_resnet50_cifar.py",
                "model": "resnet50",
                "num_classes": num_classes,
            },
        }
        if smartfrz is not None:
            row["smartfrz"] = {
                "frozen": {k: v for k, v in smartfrz.frozen.items()},
                "frozen_since_epoch": {k: v for k, v in smartfrz.frozen_since_epoch.items()},
            }

        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[Main] Epoch {epoch} | Test {val_acc:.2f}% | Best {best_acc:.2f}% "
            f"| TTA@{args.target_acc}: {time_to_target} | Time {ep_time:.1f}s | lr={lr:.6f}",
            flush=True,
        )

    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
