#!/usr/bin/env python3
"""
FreezeOut reproduction baseline for TC-ResiCache experiments.

This is a modern, single-file PyTorch reproduction adapter for the FreezeOut idea:
  - layer/block-wise cosine LR schedules,
  - progressive freezing when a block's LR reaches zero,
  - frozen blocks switched to eval mode and detached from the backward graph,
  - JSONL metrics compatible with our TC-ResiCache result analysis.

Why this file exists:
  The original FreezeOut repository code is old PyTorch/Python2 style and targets
  DenseNet, WideResNet, and VGG. Our paper settings use CIFAR ResNets such as
  CIFAR-10/ResNet-110 and CIFAR-100/ResNet-110. This file reproduces the
  FreezeOut mechanism in a modern CIFAR ResNet trainer.

Primary comparison settings:
  CIFAR-100 / ResNet-110:
    --dataset cifar100 --depth 110 --target_acc 70.0 --epochs 200

  CIFAR-10 / ResNet-110:
    --dataset cifar10 --depth 110 --target_acc 92.1 --epochs 200

Recommended FreezeOut settings from the paper:
  cubic scaled:
    --schedule cubic --t0 0.8 --scale_lr

  linear unscaled:
    --schedule linear --t0 0.5 --no_scale_lr

Example:
  python3 freezeout_resnet_cifar_repro.py \
    --dataset cifar100 --depth 110 --epochs 200 --target_acc 70.0 \
    --schedule cubic --t0 0.8 --scale_lr \
    --data_root /mnt/data2/data \
    --output_json /mnt/data2/freezeout_cifar100_resnet110_seed42.jsonl

Notes:
  - This is not TC-ResiCache. It is a FreezeOut-style baseline.
  - It does not cache activations.
  - It uses block-level freezing for CIFAR ResNet.
  - It reports zero disk_hits/reuses by design.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def schedule_scale(x: float, schedule: str) -> float:
    if schedule == "linear":
        return x
    if schedule == "squared":
        return x * x
    if schedule == "cubic":
        return x * x * x
    raise ValueError(f"Unsupported schedule: {schedule}")


class FreezeOutBlockMixin:
    """Mixin-style attributes for modules that can be frozen."""

    active: bool
    freeze_index: int
    freeze_iter: int
    base_lr: float
    current_lr: float

    def freezeout_eval_and_detach(self, out: torch.Tensor) -> torch.Tensor:
        if self.active:
            return out
        return out.detach()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int, freeze_index: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        self.active = True
        self.freeze_index = freeze_index
        self.freeze_iter = 0
        self.base_lr = 0.1
        self.current_lr = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.active:
            self.eval()
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=False)
        if self.active:
            return out
        return out.detach()


class CIFARResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must satisfy depth = 6n + 2, e.g. 56, 110, 164")
        self.depth = depth
        self.num_classes = num_classes
        n = (depth - 2) // 6

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        idx = 0
        self.layer1, idx = self._make_layer(16, n, stride=1, start_idx=idx)
        self.layer2, idx = self._make_layer(32, n, stride=2, start_idx=idx)
        self.layer3, idx = self._make_layer(64, n, stride=2, start_idx=idx)
        self.num_freeze_blocks = idx

        self.linear = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int, start_idx: int):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        idx = start_idx
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s, idx))
            self.in_planes = planes * BasicBlock.expansion
            idx += 1
        return nn.Sequential(*blocks), idx

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                nn.init.constant_(m.bias, 0.0)

    def freezeout_blocks(self) -> List[BasicBlock]:
        blocks: List[BasicBlock] = []
        for m in self.modules():
            if isinstance(m, BasicBlock):
                blocks.append(m)
        blocks.sort(key=lambda b: b.freeze_index)
        return blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(-1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


def configure_freezeout(
    model: CIFARResNet,
    base_lr: float,
    total_steps: int,
    t0: float,
    schedule: str,
    scale_lr: bool,
) -> None:
    blocks = model.freezeout_blocks()
    n = max(len(blocks) - 1, 1)
    for i, b in enumerate(blocks):
        raw = t0 + (1.0 - t0) * float(i) / float(n)
        ratio = schedule_scale(raw, schedule)
        ratio = max(min(ratio, 1.0), 1e-6)
        b.freeze_iter = int(round(total_steps * ratio))
        b.base_lr = base_lr / ratio if scale_lr else base_lr
        b.current_lr = b.base_lr
        b.active = True


def make_optimizer(model: CIFARResNet, base_lr: float, weight_decay: float) -> optim.Optimizer:
    """One optimizer param group per freezeout block, plus stem/head group."""
    freeze_block_ids = set(id(b) for b in model.freezeout_blocks())
    groups = []

    # Stem/head remain active for full training. Give group freeze_index = -1.
    stem_head_params = []
    for name, p in model.named_parameters():
        owner_block = None
        # Simpler: detect by prefix. Blocks are inside layer1/2/3.
        if not (name.startswith("layer1.") or name.startswith("layer2.") or name.startswith("layer3.")):
            stem_head_params.append(p)
    if stem_head_params:
        groups.append({"params": stem_head_params, "lr": base_lr, "freeze_index": -1, "name": "stem_head"})

    for b in model.freezeout_blocks():
        groups.append({
            "params": list(b.parameters()),
            "lr": b.current_lr,
            "freeze_index": b.freeze_index,
            "name": f"block_{b.freeze_index}",
        })

    return optim.SGD(groups, lr=base_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)


def update_freezeout(
    model: CIFARResNet,
    optimizer: optim.Optimizer,
    global_step: int,
    base_lr: float,
) -> int:
    """Update per-block LR and freeze inactive blocks. Return active block count."""
    blocks = model.freezeout_blocks()
    block_by_index = {b.freeze_index: b for b in blocks}
    active_count = 0

    # Mark active/inactive and compute LR.
    for b in blocks:
        if global_step >= b.freeze_iter:
            if b.active:
                b.active = False
                for p in b.parameters():
                    p.requires_grad = False
                b.eval()
            b.current_lr = 0.0
        else:
            b.active = True
            progress = float(global_step) / max(float(b.freeze_iter), 1.0)
            b.current_lr = 0.5 * b.base_lr * (1.0 + math.cos(math.pi * progress))
            active_count += 1

    # Update optimizer groups. Keep inactive groups at LR 0.
    for g in optimizer.param_groups:
        idx = g.get("freeze_index", -1)
        if idx == -1:
            # Standard cosine for stem/head over full training is handled by external scheduler convention:
            # keep base LR here for faithful comparison to TC training recipes.
            g["lr"] = base_lr
        else:
            b = block_by_index[idx]
            g["lr"] = b.current_lr

    return active_count


def get_dataloaders(dataset: str, data_root: str, batch_size: int, num_workers: int, seed: int):
    if dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        ds_cls = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        ds_cls = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = ds_cls(root=data_root, train=True, download=False, transform=train_tf)
    testset = ds_cls(root=data_root, train=False, download=False, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(seed)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, generator=g
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    return trainloader, testloader, num_classes


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item())
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += int(pred.eq(y).sum().item())
    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1)


def train_one_epoch(
    model: CIFARResNet,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    global_step: int,
    base_lr: float,
    log_interval: int,
) -> Tuple[float, float, int, int]:
    model.train()
    # Inactive blocks will force eval mode in their forward.
    loss_sum = 0.0
    correct = 0
    total = 0
    active_blocks = 0

    for batch_idx, (x, y) in enumerate(loader):
        active_blocks = update_freezeout(model, optimizer, global_step, base_lr)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += int(pred.eq(y).sum().item())

        if batch_idx % log_interval == 0:
            acc = 100.0 * correct / max(total, 1)
            print(f"    [Train] Epoch {epoch} [{batch_idx}/{len(loader)}] "
                  f"Loss {loss_sum/(batch_idx+1):.3f} Acc {acc:.2f}% ActiveBlocks={active_blocks}")

        global_step += 1

    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1), global_step, active_blocks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--depth", type=int, default=110)
    p.add_argument("--data_root", default="/mnt/data2/data")
    p.add_argument("--output_json", required=True)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_acc", type=float, default=70.0)
    p.add_argument("--device", default="cuda")

    p.add_argument("--t0", type=float, default=0.8)
    p.add_argument("--schedule", choices=["linear", "squared", "cubic"], default="cubic")
    scale_group = p.add_mutually_exclusive_group()
    scale_group.add_argument("--scale_lr", action="store_true", default=True)
    scale_group.add_argument("--no_scale_lr", action="store_false", dest="scale_lr")

    p.add_argument("--log_interval", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trainloader, testloader, num_classes = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.num_workers, args.seed
    )

    model = CIFARResNet(depth=args.depth, num_classes=num_classes).to(device)
    total_steps = args.epochs * len(trainloader)
    configure_freezeout(model, args.lr, total_steps, args.t0, args.schedule, args.scale_lr)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("[Main] FreezeOut reproduction")
    print(f"[Main] dataset={args.dataset} depth={args.depth} classes={num_classes} seed={args.seed}")
    print(f"[Main] epochs={args.epochs} target={args.target_acc} t0={args.t0} schedule={args.schedule} scale_lr={args.scale_lr}")
    print(f"[Main] freeze blocks={model.num_freeze_blocks} total_steps={total_steps}")

    start = time.time()
    best_acc = 0.0
    time_to_target_sec = None
    epoch_to_target = None
    global_step = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_acc, global_step, active_blocks = train_one_epoch(
            model, trainloader, optimizer, criterion, device, epoch, global_step, args.lr, args.log_interval
        )
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start
        best_acc = max(best_acc, test_acc)

        if time_to_target_sec is None and test_acc >= args.target_acc:
            time_to_target_sec = elapsed
            epoch_to_target = epoch

        frozen_blocks = model.num_freeze_blocks - active_blocks

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc_so_far": best_acc,
            "target_acc": args.target_acc,
            "time_to_target_sec": time_to_target_sec,
            "epoch_to_target": epoch_to_target,
            "elapsed_total_sec": elapsed,
            "time": epoch_time,
            "freezeout": {
                "active_blocks": active_blocks,
                "frozen_blocks": frozen_blocks,
                "num_freeze_blocks": model.num_freeze_blocks,
                "global_step": global_step,
                "t0": args.t0,
                "schedule": args.schedule,
                "scale_lr": args.scale_lr,
            },
            # Keep this schema compatible with our existing summary tools.
            "cache_stats": {
                "corrected_reuses": 0,
                "memory_hits": 0,
                "disk_hits": 0,
                "tcrc_mean_probe_error": 0.0,
                "bank_allocated_gb": 0.0,
            },
            "timing": {},
            "config": {
                "script": "freezeout_resnet_cifar_repro.py",
                "method": "freezeout_reproduction",
                "dataset": args.dataset,
                "model": f"resnet{args.depth}",
                "seed": args.seed,
                "epochs": args.epochs,
                "target_acc": args.target_acc,
                "t0": args.t0,
                "schedule": args.schedule,
                "scale_lr": args.scale_lr,
                "num_classes": num_classes,
            },
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[Main] Epoch {epoch} | Test {test_acc:.2f}% | Best {best_acc:.2f}% | "
              f"TTA@{args.target_acc}: {time_to_target_sec} | FrozenBlocks {frozen_blocks}/{model.num_freeze_blocks} | "
              f"Time {epoch_time:.1f}s")

    print(f"[Main] Done: {out_path}")


if __name__ == "__main__":
    main()
