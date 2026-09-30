#!/usr/bin/env python3
"""
Plain CIFAR-10 / ResNet-56 baseline runner for ResiCache multi-seed validation.

This intentionally contains no freezing controller and no activation cache.
It matches the ResiCache training setup as closely as possible:
  - CIFAR-10
  - ResNet-56
  - random crop + random horizontal flip
  - SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
  - MultiStepLR milestones [100, 150], gamma=0.1
  - target accuracy logging
  - JSONL output compatible with the multi-seed summary script
"""

import os
import json
import time
import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
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

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + self.shortcut(x))
        return out


class ResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 9, stride=1)
        self.layer2 = self._make_layer(32, 9, stride=2)
        self.layer3_0 = self._make_layer(64, 3, stride=2)
        self.layer3_1 = self._make_layer(64, 2, stride=1)
        self.layer3_2 = self._make_layer(64, 2, stride=1)
        self.layer3_3 = self._make_layer(64, 2, stride=1)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3_0(out)
        out = self.layer3_1(out)
        out = self.layer3_2(out)
        out = self.layer3_3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(
                f"    [Train] Epoch: {epoch} [{batch_idx}/{len(loader)}] "
                f"Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100.0 * correct / total:.2f}%"
            )

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/data2/data")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--target_acc", type=float, default=92.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")
    if device.type == "cuda":
        print(f"[Main] GPU: {torch.cuda.get_device_name(0)}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transform_test)

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=gen,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ResNet56().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    total_start = time.time()
    time_to_target_sec = None
    epoch_to_target = None
    best_acc = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device, epoch)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - total_start
        best_acc = max(best_acc, test_acc)

        if time_to_target_sec is None and test_acc >= args.target_acc:
            time_to_target_sec = elapsed_total
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
            "elapsed_total_sec": elapsed_total,
            "frozen_layers": [],
            "frontmost_frozen_layer": None,
            "frontmost_active_layer": None,
            "cache_stats": {
                "hit_rate": 0.0,
                "memory_hits": 0,
                "disk_hits": 0,
                "misses": 0,
                "sample_accesses": 0,
                "sample_hits": 0,
                "corrected_reuses": 0,
                "delta_updates": 0,
                "refresh_batches": 0,
                "avg_delta_norm": 0.0,
            },
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
            "timing": {},
            "config": {
                "script": "baseline_resnet56_cifar10_jsonl.py",
                "method": "plain_baseline",
                "seed": args.seed,
                "target_acc": args.target_acc,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
        }

        with open(args.output_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[Main] Epoch {epoch} | Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | "
            f"TTA@{args.target_acc}: {time_to_target_sec} | Time: {epoch_time:.1f}s"
        )

        scheduler.step()

    print(f"[Main] Done. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
