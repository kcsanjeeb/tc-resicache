#!/usr/bin/env python3
"""
Tiny-ImageNet / TinyResNet-34 TC-ResiCache seed-sweep trainer.

This is a practical third-setting experiment for the TC-ResiCache paper.

Design choice:
- Uses Tiny-ImageNet-200 with 64x64 images.
- Uses a ResNet-34 topology with CIFAR/Tiny-friendly width=16 by default.
  This keeps the dense activation bank feasible on 32GB V100 GPUs.
- Uses random padded crop + horizontal flip, not RandomResizedCrop, so the
  crop/flip transform can be reconstructed consistently in feature space.

Configurations are controlled by command-line flags:
- baseline: do not pass --tc_mode
- freeze-only: --tc_mode freeze --boundary layer2.1
- TC cache: --tc_mode cache --boundary layer1 --cache_start_epoch 55

JSONL output is compatible with previous TC-ResiCache analysis scripts.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF


TINY_MEAN = (0.4802, 0.4481, 0.3975)
TINY_STD = (0.2302, 0.2265, 0.2262)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyImageNetAugDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, padding: int = 4):
        split = "train" if train else "val"
        self.root = Path(root) / split
        self.train = train
        self.padding = padding
        self.base = ImageFolder(str(self.root))
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx

    def __len__(self):
        return len(self.base)

    def _normalize(self, img: Image.Image) -> torch.Tensor:
        x = TF.to_tensor(img.convert("RGB"))
        x = TF.normalize(x, TINY_MEAN, TINY_STD)
        return x

    def __getitem__(self, index: int):
        path, target = self.base.samples[index]
        img = Image.open(path).convert("RGB")
        x = self._normalize(img)

        if not self.train:
            return x, target, index, 0, 0, 0, x

        p = self.padding
        canonical = F.pad(x, (p, p, p, p), mode="constant", value=0.0)
        max_off = 2 * p
        top = random.randint(0, max_off)
        left = random.randint(0, max_off)
        crop = canonical[:, top:top + 64, left:left + 64]
        flip = 1 if random.random() < 0.5 else 0
        if flip:
            crop = torch.flip(crop, dims=[2])
        return crop, target, index, top, left, flip, canonical


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + self.shortcut(x))
        return out


class TinyResNet34(nn.Module):
    def __init__(self, num_classes=200, width=16):
        super().__init__()
        self.width = width
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, 3, stride=1)
        self.layer2 = self._make_layer(width * 2, 4, stride=2)
        self.layer3 = self._make_layer(width * 4, 6, stride=2)
        self.layer4 = self._make_layer(width * 8, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 8, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def stem(self, x):
        return torch.relu(self.bn1(self.conv1(x)))

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)

    def forward_prefix(self, x, boundary: str):
        out = self.stem(x)
        if boundary == "stem":
            return out
        out = self.layer1(out)
        if boundary == "layer1":
            return out
        for i, blk in enumerate(self.layer2):
            out = blk(out)
            if boundary == f"layer2.{i}":
                return out
        if boundary == "layer2":
            return out
        for i, blk in enumerate(self.layer3):
            out = blk(out)
            if boundary == f"layer3.{i}":
                return out
        if boundary == "layer3":
            return out
        raise ValueError(f"Unsupported boundary: {boundary}")

    def forward_suffix(self, feat, boundary: str):
        out = feat
        if boundary == "stem":
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif boundary == "layer1":
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif boundary.startswith("layer2."):
            start = int(boundary.split(".")[1]) + 1
            for blk in self.layer2[start:]:
                out = blk(out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif boundary == "layer2":
            out = self.layer3(out)
            out = self.layer4(out)
        elif boundary.startswith("layer3."):
            start = int(boundary.split(".")[1]) + 1
            for blk in self.layer3[start:]:
                out = blk(out)
            out = self.layer4(out)
        elif boundary == "layer3":
            out = self.layer4(out)
        else:
            raise ValueError(f"Unsupported boundary: {boundary}")
        out = self.avgpool(out).flatten(1)
        return self.fc(out)

    def modules_until_boundary(self, boundary: str):
        mods = [self.conv1, self.bn1]
        if boundary == "stem":
            return mods
        mods.append(self.layer1)
        if boundary == "layer1":
            return mods
        if boundary.startswith("layer2."):
            k = int(boundary.split(".")[1])
            mods.extend([self.layer2[i] for i in range(k + 1)])
            return mods
        mods.append(self.layer2)
        if boundary == "layer2":
            return mods
        if boundary.startswith("layer3."):
            k = int(boundary.split(".")[1])
            mods.extend([self.layer3[i] for i in range(k + 1)])
            return mods
        mods.append(self.layer3)
        return mods


def freeze_prefix(model: TinyResNet34, boundary: str) -> None:
    for mod in model.modules_until_boundary(boundary):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad = False


def boundary_stride(boundary: str) -> int:
    if boundary in ("stem", "layer1"):
        return 1
    if boundary.startswith("layer2") or boundary == "layer2":
        return 2
    if boundary.startswith("layer3") or boundary == "layer3":
        return 4
    return 1


def crop_from_canonical(feat_canon: torch.Tensor, top: torch.Tensor, left: torch.Tensor,
                        flip: torch.Tensor, boundary: str, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear/sub-pixel crop from canonical padded feature map."""
    b, c, hc, wc = feat_canon.shape
    oh, ow = out_hw
    stride = boundary_stride(boundary)
    device = feat_canon.device
    dtype = feat_canon.dtype

    ys = torch.arange(oh, device=device, dtype=torch.float32).view(1, oh, 1).expand(b, oh, ow)
    xs = torch.arange(ow, device=device, dtype=torch.float32).view(1, 1, ow).expand(b, oh, ow)

    top_f = top.to(device=device, dtype=torch.float32).view(b, 1, 1) / float(stride)
    left_f = left.to(device=device, dtype=torch.float32).view(b, 1, 1) / float(stride)

    y = ys + top_f
    x = xs + left_f

    # align_corners=True normalization.
    yn = 2.0 * y / max(hc - 1, 1) - 1.0
    xn = 2.0 * x / max(wc - 1, 1) - 1.0
    grid = torch.stack([xn, yn], dim=-1)

    out = F.grid_sample(feat_canon.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    out = out.to(dtype=dtype)
    flip_mask = flip.to(device=device).bool()
    if flip_mask.any():
        out[flip_mask] = torch.flip(out[flip_mask], dims=[3])
    return out


class DenseFeatureBank:
    def __init__(self, num_samples: int, shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype):
        c, h, w = shape
        self.num_samples = num_samples
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.bank = torch.empty((num_samples, c, h, w), device=device, dtype=dtype)
        self.valid = torch.zeros((num_samples,), device=device, dtype=torch.bool)
        self.populates = 0
        self.hits = 0
        self.misses = 0

    def get_valid_mask(self, idx: torch.Tensor):
        idx = idx.to(self.device)
        return self.valid[idx]

    def fetch(self, idx: torch.Tensor):
        idx = idx.to(self.device)
        valid = self.valid[idx]
        self.hits += int(valid.sum().item())
        self.misses += int((~valid).sum().item())
        return self.bank[idx], valid

    def store(self, idx: torch.Tensor, feats: torch.Tensor):
        idx = idx.to(self.device)
        self.bank[idx] = feats.to(device=self.device, dtype=self.dtype)
        self.valid[idx] = True
        self.populates += int(idx.numel())

    @property
    def allocated_gb(self):
        return self.bank.numel() * self.bank.element_size() / (1024 ** 3)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets, *_ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            out = model(inputs)
            loss = criterion(out, targets)
            loss_sum += float(loss.item())
            pred = out.argmax(1)
            total += targets.size(0)
            correct += int(pred.eq(targets).sum().item())
    return loss_sum / len(loader), 100.0 * correct / total


def run_epoch(model, loader, optimizer, criterion, device, args, epoch, bank, frozen):
    model.train()
    if frozen:
        freeze_prefix(model, args.boundary)

    loss_sum, correct, total = 0.0, 0, 0
    corrected_reuses = 0
    memory_hits = 0
    transform_time = 0.0
    populate_time = 0.0
    probe_errors = []

    for batch_idx, batch in enumerate(loader):
        inputs, targets, indices, top, left, flip, canonical = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        left = left.to(device, non_blocking=True)
        flip = flip.to(device, non_blocking=True)
        canonical = canonical.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if args.tc_mode == "none" or not frozen:
            out = model(inputs)
        elif args.tc_mode == "freeze":
            with torch.no_grad():
                feat = model.forward_prefix(inputs, args.boundary)
            out = model.forward_suffix(feat.detach(), args.boundary)
        else:
            assert args.tc_mode == "cache"
            with torch.no_grad():
                # Create bank lazily after seeing canonical prefix shape.
                if bank[0] is None:
                    feat0 = model.forward_prefix(canonical[:1], args.boundary)
                    bank[0] = DenseFeatureBank(
                        num_samples=len(loader.dataset),
                        shape=tuple(feat0.shape[1:]),
                        device=device,
                        dtype=torch.float16 if args.bank_dtype == "fp16" else torch.float32,
                    )
                    print(f"[Bank] allocated shape={bank[0].bank.shape} gb={bank[0].allocated_gb:.2f}")

                bank_obj = bank[0]
                feats_canon, valid = bank_obj.fetch(indices)
                if (~valid).any():
                    t0 = time.time()
                    miss_idx = indices[~valid]
                    miss_canon = canonical[~valid]
                    miss_feat = model.forward_prefix(miss_canon, args.boundary)
                    bank_obj.store(miss_idx, miss_feat)
                    feats_canon[~valid] = miss_feat.to(feats_canon.dtype)
                    populate_time += time.time() - t0

                # Determine feature output size by current crop prefix once for shape.
                # Avoid full prefix for every batch unless probing.
                stride = boundary_stride(args.boundary)
                out_hw = (64 // stride, 64 // stride)

                t1 = time.time()
                feat = crop_from_canonical(feats_canon, top, left, flip, args.boundary, out_hw)
                transform_time += time.time() - t1

                if args.probe_every > 0 and batch_idx % args.probe_every == 0:
                    true_feat = model.forward_prefix(inputs, args.boundary)
                    err = (true_feat.float() - feat.float()).pow(2).mean().sqrt().item()
                    probe_errors.append(err)

            corrected_reuses += int(indices.numel())
            memory_hits += int(valid.sum().item())
            out = model.forward_suffix(feat.detach(), args.boundary)

        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        pred = out.argmax(1)
        total += targets.size(0)
        correct += int(pred.eq(targets).sum().item())

        if batch_idx % 50 == 0:
            print(f"    [Train] Epoch {epoch} [{batch_idx}/{len(loader)}] Loss {loss_sum/(batch_idx+1):.3f} Acc {100.0*correct/total:.2f}% Frozen={frozen}")

    stats = {
        "corrected_reuses": corrected_reuses,
        "memory_hits": memory_hits,
        "disk_hits": 0,
        "tcrc_mean_probe_error": float(np.mean(probe_errors)) if probe_errors else 0.0,
        "bank_allocated_gb": bank[0].allocated_gb if bank[0] is not None else 0.0,
        "bank_populates": bank[0].populates if bank[0] is not None else 0,
    }
    timing = {
        "tcrc_get_transform_batch": {"total_sec": transform_time},
        "tcrc_canonical_populate": {"total_sec": populate_time},
    }
    return loss_sum / len(loader), 100.0 * correct / total, stats, timing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/data2/data/tiny-imagenet-200")
    p.add_argument("--output_json", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--target_acc", type=float, default=40.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--padding", type=int, default=4)
    p.add_argument("--tc_mode", choices=["none", "freeze", "cache"], default="none")
    p.add_argument("--boundary", default="layer1")
    p.add_argument("--force_freeze_epoch", type=int, default=55)
    p.add_argument("--cache_start_epoch", type=int, default=55)
    p.add_argument("--bank_dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--probe_every", type=int, default=100)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Main] Tiny-ImageNet / TinyResNet34 width={args.width} mode={args.tc_mode} boundary={args.boundary} seed={args.seed} device={device}")

    trainset = TinyImageNetAugDataset(args.data_root, train=True, padding=args.padding)
    testset = TinyImageNetAugDataset(args.data_root, train=False, padding=args.padding)

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        generator=gen,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = TinyResNet34(num_classes=200, width=args.width).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    total_start = time.time()
    best_acc = 0.0
    time_to_target_sec = None
    epoch_to_target = None
    bank = [None]
    frozen = False

    for epoch in range(args.epochs):
        if args.tc_mode != "none" and (not frozen) and epoch >= args.force_freeze_epoch:
            frozen = True
            freeze_prefix(model, args.boundary)
            optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=optimizer.param_groups[0]["lr"], momentum=0.9, weight_decay=5e-4)
            # Keep scheduler simple after optimizer reset by mirroring LR manually through current LR.
            print(f"[Freeze] epoch={epoch} boundary={args.boundary}")

        cache_allowed = args.tc_mode == "cache" and epoch >= args.cache_start_epoch
        effective_mode = args.tc_mode if cache_allowed or args.tc_mode != "cache" else "freeze"
        old_mode = args.tc_mode
        args.tc_mode = effective_mode

        epoch_start = time.time()
        train_loss, train_acc, cache_stats, timing = run_epoch(model, trainloader, optimizer, criterion, device, args, epoch, bank, frozen)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        epoch_time = time.time() - epoch_start

        args.tc_mode = old_mode

        elapsed = time.time() - total_start
        best_acc = max(best_acc, test_acc)
        if time_to_target_sec is None and test_acc >= args.target_acc:
            time_to_target_sec = elapsed
            epoch_to_target = epoch

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
            "frozen_layers": [args.boundary] if frozen else [],
            "frontmost_frozen_layer": args.boundary if frozen else None,
            "cache_stats": cache_stats,
            "timing": timing,
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
            "config": {
                "script": "tiny_imagenet_resnet34_tc_resicache.py",
                "dataset": "tiny-imagenet-200",
                "model": f"tiny-resnet34-width{args.width}",
                "seed": args.seed,
                "epochs": args.epochs,
                "target_acc": args.target_acc,
                "tc_mode": old_mode,
                "boundary": args.boundary,
                "force_freeze_epoch": args.force_freeze_epoch,
                "cache_start_epoch": args.cache_start_epoch,
                "bank_dtype": args.bank_dtype,
                "padding": args.padding,
            },
        }

        with open(args.output_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[Main] Epoch {epoch} | Test {test_acc:.2f}% | Best {best_acc:.2f}% | TTA@{args.target_acc}: {time_to_target_sec} | Time {epoch_time:.1f}s | mode={effective_mode}")

        scheduler.step()

    print(f"[Main] Done: {args.output_json}")


if __name__ == "__main__":
    main()
