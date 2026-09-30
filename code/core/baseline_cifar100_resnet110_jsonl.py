#!/usr/bin/env python3
"""
Plain CIFAR-100 / ResNet-110 baseline runner.

No freezing and no cache.
JSONL schema is compatible with the TC-ResiCache summary scripts.
"""

import os, json, time, argparse, random, warnings
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


class ResNet110CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 18, stride=1)
        self.layer2 = self._make_layer(32, 18, stride=2)
        # Split layer3 into four modules so naming matches TC-ResiCache scripts.
        self.layer3_0 = self._make_layer(64, 5, stride=2)
        self.layer3_1 = self._make_layer(64, 5, stride=1)
        self.layer3_2 = self._make_layer(64, 4, stride=1)
        self.layer3_3 = self._make_layer(64, 4, stride=1)
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
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
        if batch_idx % 50 == 0:
            print(f"    [Train] Epoch: {epoch} [{batch_idx}/{len(loader)}] Loss: {total_loss/(batch_idx+1):.3f} | Acc: {100.0*correct/total:.2f}%")
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += float(loss.item())
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/data2/data")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--target_acc", type=float, default=70.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Main] CIFAR-100 / ResNet-110 baseline | device={device} seed={args.seed}")

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=False, transform=transform_test)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker, generator=gen,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = ResNet110CIFAR(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    total_start = time.time()
    best_acc = 0.0
    time_to_target_sec = None
    epoch_to_target = None

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device, epoch)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        epoch_time = time.time() - epoch_start
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
            "frozen_layers": [],
            "cache_stats": {
                "hit_rate": 0.0,
                "memory_hits": 0,
                "disk_hits": 0,
                "corrected_reuses": 0,
                "delta_updates": 0,
            },
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
            "timing": {},
            "config": {
                "script": "baseline_cifar100_resnet110_jsonl.py",
                "dataset": "cifar100",
                "model": "resnet110",
                "method": "plain_baseline",
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "target_acc": args.target_acc,
            },
        }
        with open(args.output_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[Main] Epoch {epoch} | Test={test_acc:.2f}% | Best={best_acc:.2f}% | TTA@{args.target_acc}={time_to_target_sec} | Time={epoch_time:.1f}s")
        scheduler.step()

    print(f"[Main] Done: {args.output_json}")


if __name__ == "__main__":
    main()
