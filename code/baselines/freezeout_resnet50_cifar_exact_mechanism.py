#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

SMARTFRZ_FULL_TARGETS = {"cifar10": 96.10, "cifar100": 81.68}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = os.environ.get("STRICT_DETERMINISTIC", "0") != "1"
    torch.backends.cudnn.deterministic = os.environ.get("STRICT_DETERMINISTIC", "0") == "1"

def schedule_power(schedule: str) -> int:
    return {"linear": 1, "squared": 2, "cubic": 3}[schedule]

def schedule_ratio(i: int, n: int, t0: float, schedule: str) -> float:
    raw = 1.0 if n <= 1 else t0 + (1.0 - t0) * float(i) / float(n - 1)
    raw = max(min(raw, 1.0), 1e-6)
    return raw ** schedule_power(schedule)

def make_data(dataset: str, data_root: str, input_size: int, batch_size: int, num_workers: int, seed: int):
    train_tf = T.Compose([T.RandomResizedCrop(input_size), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_tf = T.Compose([T.Resize(input_size), T.CenterCrop(input_size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if dataset == "cifar10":
        cls, num_classes = torchvision.datasets.CIFAR10, 10
    elif dataset == "cifar100":
        cls, num_classes = torchvision.datasets.CIFAR100, 100
    else:
        raise ValueError(dataset)
    train = cls(root=data_root, train=True, download=False, transform=train_tf)
    val = cls(root=data_root, train=False, download=False, transform=val_tf)
    gen = torch.Generator(); gen.manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=gen, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return train_loader, val_loader, num_classes

def build_model(model_name: str, num_classes: int, weights: str):
    model_name = model_name.lower()
    builder = getattr(torchvision.models, model_name, None)
    if builder is None:
        raise ValueError(f"unknown model {model_name}")
    if weights == "default":
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
    else:
        try:
            model = builder(weights=None)
        except Exception:
            model = builder(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class FreezeOutController:
    """
    Exact FreezeOut mechanism adapted to torchvision ResNet50:
    per-unit cosine LR to zero, freeze, eval, and detach output.
    Units are stem + each Bottleneck block; classifier head remains trainable.
    """
    def __init__(self, model: nn.Module, total_steps: int, base_lr: float, t0: float, schedule: str, scale_lr: bool):
        self.model, self.total_steps, self.base_lr = model, total_steps, base_lr
        self.t0, self.schedule, self.scale_lr = t0, schedule, scale_lr
        self.units: List[Tuple[str, nn.Module]] = [("stem", nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool))]
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, lname)
            for bi, block in enumerate(layer):
                self.units.append((f"{lname}.{bi}", block))
        n = len(self.units)
        self.freeze_steps: Dict[str, int] = {}
        self.base_lrs: Dict[str, float] = {}
        self.frozen: Dict[str, bool] = {}
        for i, (name, _) in enumerate(self.units):
            r = schedule_ratio(i, n, t0, schedule)
            self.freeze_steps[name] = int(round(total_steps * r))
            self.base_lrs[name] = base_lr / r if scale_lr else base_lr
            self.frozen[name] = False
        self.hooks = []
        self._install_hooks()

    def _install_hooks(self):
        def make_hook(name):
            def hook(module, inputs, output):
                if self.frozen.get(name, False) and isinstance(output, torch.Tensor):
                    return output.detach()
                return output
            return hook
        for name, mod in self.units:
            self.hooks.append(mod.register_forward_hook(make_hook(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def param_groups(self):
        groups, used = [], set()
        for name, mod in self.units:
            params = list(mod.parameters())
            for p in params:
                used.add(id(p))
            groups.append({"params": params, "lr": self.base_lrs[name], "freezeout_name": name})
        rest = [p for p in self.model.parameters() if id(p) not in used]
        if rest:
            groups.append({"params": rest, "lr": self.base_lr, "freezeout_name": "head"})
        return groups

    def update(self, optimizer: optim.Optimizer, step: int):
        active = frozen_now = 0
        for name, mod in self.units:
            if step >= self.freeze_steps[name]:
                if not self.frozen[name]:
                    self.frozen[name] = True
                    mod.eval()
                    for p in mod.parameters():
                        p.requires_grad = False
                frozen_now += 1
            else:
                active += 1
        for group in optimizer.param_groups:
            name = group.get("freezeout_name", "head")
            if name == "head":
                group["lr"] = self.base_lr
            elif self.frozen[name]:
                group["lr"] = 0.0
            else:
                fs = max(float(self.freeze_steps[name]), 1.0)
                progress = min(max(float(step) / fs, 0.0), 1.0)
                group["lr"] = 0.5 * self.base_lrs[name] * (1.0 + math.cos(math.pi * progress))
        return active, frozen_now

@torch.no_grad()
def evaluate(model, loader, criterion, device, amp):
    model.eval(); correct = total = 0; loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x); loss = criterion(logits, y)
        loss_sum += float(loss.item()); pred = logits.argmax(1)
        total += y.numel(); correct += int(pred.eq(y).sum().item())
    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1)

def train_epoch(model, loader, optimizer, criterion, device, controller, global_step, args, scaler):
    model.train(); loss_sum = 0.0; correct = total = 0; last_active = last_frozen = 0
    for bi, (x, y) in enumerate(loader):
        last_active, last_frozen = controller.update(optimizer, global_step)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(x); loss = criterion(logits, y)
        if args.amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        loss_sum += float(loss.item()); pred = logits.detach().argmax(1)
        total += y.numel(); correct += int(pred.eq(y).sum().item())
        if bi % args.log_interval == 0:
            print(f"    [Train] step={global_step} batch={bi}/{len(loader)} loss={loss_sum/(bi+1):.4f} acc={100.0*correct/max(total,1):.2f} active={last_active} frozen={last_frozen}", flush=True)
        global_step += 1
    return loss_sum / max(len(loader), 1), 100.0 * correct / max(total, 1), global_step, last_active, last_frozen

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--data_root", default="/mnt/data2/data")
    p.add_argument("--output_json", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_acc", type=float, default=None)
    p.add_argument("--model", choices=["resnet18", "resnet34", "resnet50", "resnet101"], default="resnet50")
    p.add_argument("--weights", choices=["default", "none"], default="default")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--schedule", choices=["linear", "squared", "cubic"], default="cubic")
    p.add_argument("--t0", type=float, default=0.8)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scale_lr", action="store_true", default=True)
    g.add_argument("--no_scale_lr", dest="scale_lr", action="store_false")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_interval", type=int, default=100)
    return p.parse_args()

def main():
    args = parse_args(); set_seed(args.seed)
    if args.target_acc is None:
        args.target_acc = SMARTFRZ_FULL_TARGETS[args.dataset]
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = make_data(args.dataset, args.data_root, args.input_size, args.batch_size, args.num_workers, args.seed)
    model = build_model(args.model, num_classes, args.weights).to(device)
    total_steps = args.epochs * len(train_loader)
    controller = FreezeOutController(model, total_steps, args.lr, args.t0, args.schedule, args.scale_lr)
    optimizer = optim.SGD(controller.param_groups(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(); scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print("[Main] FreezeOut exact-mechanism ResNet50 CIFAR comparison")
    print(f"[Main] dataset={args.dataset} target={args.target_acc} schedule={args.schedule} t0={args.t0} scale_lr={args.scale_lr}")
    print(f"[Main] units={len(controller.units)} total_steps={total_steps}")
    print(f"[Main] freeze_steps={controller.freeze_steps}")
    start = time.time(); best_acc = 0.0; time_to_target = None; epoch_to_target = None; global_step = 0
    for epoch in range(args.epochs):
        ep_start = time.time()
        train_loss, train_acc, global_step, active_units, frozen_units = train_epoch(model, train_loader, optimizer, criterion, device, controller, global_step, args, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.amp)
        ep_time = time.time() - ep_start; elapsed = time.time() - start
        best_acc = max(best_acc, val_acc)
        if time_to_target is None and val_acc >= args.target_acc:
            time_to_target = elapsed; epoch_to_target = epoch
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": val_loss, "test_acc": val_acc, "best_test_acc_so_far": best_acc,
            "target_acc": args.target_acc, "time_to_target_sec": time_to_target,
            "epoch_to_target": epoch_to_target, "elapsed_total_sec": elapsed, "time": ep_time,
            "freezeout": {"active_units": active_units, "frozen_units": frozen_units, "num_units": len(controller.units), "schedule": args.schedule, "t0": args.t0, "scale_lr": args.scale_lr, "global_step": global_step},
            "cache_stats": {"corrected_reuses": 0, "memory_hits": 0, "disk_hits": 0, "bank_allocated_gb": 0.0},
            "config": vars(args) | {"script": "freezeout_resnet50_cifar_exact_mechanism.py", "method": "freezeout_exact_mechanism_resnet_adapter", "num_classes": num_classes, "augmentation": "RandomResizedCrop(224)+HorizontalFlip, ImageNet normalization"},
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[Main] Epoch {epoch} | Test {val_acc:.2f}% | Best {best_acc:.2f}% | TTA@{args.target_acc}: {time_to_target} | Frozen {frozen_units}/{len(controller.units)} | Time {ep_time:.1f}s", flush=True)
    controller.remove_hooks()
    print(f"[Done] wrote {out}")

if __name__ == "__main__":
    main()
