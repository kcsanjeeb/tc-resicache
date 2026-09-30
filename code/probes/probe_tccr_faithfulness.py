#!/usr/bin/env python3
"""TC-CR faithfulness probe at the full-training setting.

Setting: CIFAR-10, ResNet-56, RandomCrop(32, pad=4) + horizontal flip,
the regime where the paper claims TC-CR works.

For each test sample we build the canonical bank entry exactly as TC-CR
does: frozen prefix applied to the padded (40x40), unflipped full image,
at boundaries layer2 (stride 4) and layer3 (stride 8). For random crops
(u, v, flip) we compare, against the TRUE prefix output on the crop:
  TC-CR: window at (u,v)/stride of size 32/stride, bilinear for
         non-integer offsets, feature-flipped when flip=1;
  TC-SI: the stored activation of a fixed first crop (stale reuse);
  ref:   the activation of a different image (calibration floor).
Metrics: relative L2 error and cosine similarity, mean +/- std.
"""
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# ---------------- CIFAR ResNet-56 ----------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.short(x) if self.short is not None else x
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, depth=56, num_classes=10):
        super().__init__()
        n = (depth - 2) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._stage(16, 16, n, 1)
        self.layer2 = self._stage(16, 32, n, 2)
        self.layer3 = self._stage(32, 64, n, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _stage(self, cin, cout, n, stride):
        blocks = [BasicBlock(cin, cout, stride)]
        blocks += [BasicBlock(cout, cout, 1) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def prefix(self, x, boundary):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        if boundary == "layer1":
            return x
        x = self.layer2(x)
        if boundary == "layer2":
            return x
        return self.layer3(x)

    def forward(self, x):
        x = self.prefix(x, "layer3")
        return self.fc(self.avgpool(x).flatten(1))


# ---------------- window extraction (the TC-CR mechanism) ----------------
def extract_window(canon, u, v, stride, crop=32):
    """Bilinear sampling of canon at offset (u/stride, v/stride), window
    crop/stride x crop/stride. canon: (C, H, W)."""
    C, H, W = canon.shape
    w = crop // stride
    # grid of target positions in canonical coordinates
    ys = v / stride + torch.arange(w, device=canon.device, dtype=torch.float32)
    xs = u / stride + torch.arange(w, device=canon.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # normalize to [-1, 1] for grid_sample
    gx = (xx / max(W - 1, 1)) * 2 - 1
    gy = (yy / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None]  # (1, w, w, 2)
    out = F.grid_sample(canon[None], grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out[0]


def metrics(a, b, eps=1e-8):
    a = a.flatten().float()
    b = b.flatten().float()
    rel = (a - b).norm() / (b.norm() + eps)
    cos = F.cosine_similarity(a, b, dim=0, eps=eps)
    return rel.item(), cos.item()


def main():
    device = "cuda"
    torch.manual_seed(0)
    model = CifarResNet(56).to(device)

    ckpt = "/mnt/data2/tccr_probe/r56_e30.pt"
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("[probe] loaded checkpoint", ckpt)
    except FileNotFoundError:
        print("[probe] no checkpoint, training 30 epochs first")
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ds = torchvision.datasets.CIFAR10("/mnt/data2/data", train=True, transform=tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True,
                                         num_workers=4, drop_last=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        ce = nn.CrossEntropyLoss()
        for ep in range(30):
            model.train()
            t0 = time.time()
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = ce(model(x), y)
                loss.backward()
                opt.step()
            sch.step()
            print(f"[train] epoch {ep} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), ckpt)
    model.eval()  # frozen prefix convention: BN uses running statistics

    # probe data: raw test images (untransformed, unnormalized handled below)
    base = torchvision.datasets.CIFAR10("/mnt/data2/data", train=False, transform=None)
    N, CROPS, PAD = 500, 8, 4
    torch.manual_seed(123)

    results = {}
    for boundary, stride in [("layer2", 2), ("layer3", 4)]:
        agg = {k: [] for k in ["tccr_rel", "tccr_cos", "tcsi_rel", "tcsi_cos",
                               "ref_rel", "ref_cos"]}
        with torch.no_grad():
            for i in range(N):
                img = T.ToTensor()(base[i][0])  # (3,32,32) in [0,1]
                img_pad = F.pad(img, [PAD] * 4)  # 40x40
                canon = model.prefix(img_pad[None].to(device), boundary)[0]
                # TC-SI entry: activation of one fixed but RANDOM crop per
                # sample (what the populate epoch actually stores)
                u0 = int(torch.randint(0, 2 * PAD + 1, (1,)))
                v0 = int(torch.randint(0, 2 * PAD + 1, (1,)))
                stored = model.prefix(img_pad[:, v0:v0+32, u0:u0+32][None].to(device),
                                      boundary)[0]
                ref = model.prefix(T.ToTensor()(base[(i + 1) % len(base)][0])[None]
                                   .to(device), boundary)[0]
                for _ in range(CROPS):
                    u = int(torch.randint(0, 2 * PAD + 1, (1,)))
                    v = int(torch.randint(0, 2 * PAD + 1, (1,)))
                    flip = int(torch.randint(0, 2, (1,)))
                    crop = img_pad[:, v:v+32, u:u+32]
                    if flip:
                        crop = torch.flip(crop, dims=[2])
                    true = model.prefix(crop[None].to(device), boundary)[0]
                    # TC-CR: map flipped crop back to unflipped canonical coords
                    uc = 2 * PAD - u if flip else u
                    rec = extract_window(canon, uc, v, stride)
                    if flip:
                        rec = torch.flip(rec, dims=[2])
                    for name, cand in [("tccr", rec), ("tcsi", stored), ("ref", ref)]:
                        if cand.shape != true.shape:
                            print(f"[shape-mismatch] boundary={boundary} i={i} "
                                  f"name={name} cand={tuple(cand.shape)} true={tuple(true.shape)} "
                                  f"canon={tuple(canon.shape)} u={u} v={v} flip={flip}", flush=True)
                            raise SystemExit(1)
                        r, c = metrics(cand, true)
                        agg[f"{name}_rel"].append(r)
                        agg[f"{name}_cos"].append(c)
                if (i + 1) % 100 == 0:
                    print(f"[probe] {boundary} {i+1}/{N}", flush=True)
        import numpy as np
        results[boundary] = {k: [float(np.mean(v)), float(np.std(v))]
                             for k, v in agg.items()}
        print(boundary, json.dumps({k: [round(m, 3), round(s, 3)]
                                    for k, (m, s) in results[boundary].items()}))
    with open("/mnt/data2/tccr_probe/probe_tccr_faithfulness.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
