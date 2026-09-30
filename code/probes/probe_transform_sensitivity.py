#!/usr/bin/env python3
"""Reviewer probe: transform sensitivity of boundary activations, across many
samples and several freezing boundaries, in interpretable units.

Answers the review point that Fig. 1(a) rests on one sample and reports raw L2:
  - N samples (not 1), K augmented views each
  - boundaries layer1..layer4 (not one)
  - relative L2 error and cosine similarity (not raw L2)
  - a different-image reference, so the numbers can be calibrated

Setting matches the fine-tuning experiments: CIFAR-10, ImageNet-pretrained
ResNet-50, RandomResizedCrop(224) + RandomHorizontalFlip.

Run from repo root:  python3 experiments/paper_requirements/code/probe_transform_sensitivity.py
"""
import argparse, json, itertools
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50, ResNet50_Weights

p = argparse.ArgumentParser()
p.add_argument("--samples", type=int, default=64)
p.add_argument("--views", type=int, default=16)
p.add_argument("--data", default="experiments/data")
p.add_argument("--device", default="mps")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="experiments/paper_requirements/probe_transform_sensitivity.json")
a = p.parse_args()

torch.manual_seed(a.seed)
dev = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")

net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(dev)
for q in net.parameters(): q.requires_grad_(False)

BOUNDS = {"layer1": net.layer1, "layer2": net.layer2, "layer3": net.layer3, "layer4": net.layer4}
feat = {}
for nm, mod in BOUNDS.items():
    mod.register_forward_hook(lambda m, i, o, nm=nm: feat.__setitem__(nm, o.detach().flatten(1)))

aug = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds = CIFAR10(a.data, train=True, download=False)

def views_of(img, k):
    return torch.stack([aug(img) for _ in range(k)]).to(dev)

acc = {b: {"rel_l2": [], "cos": []} for b in BOUNDS}
xacc = {b: {"rel_l2": [], "cos": []} for b in BOUNDS}
prev = None
for s in range(a.samples):
    img, _ = ds[s]
    with torch.no_grad():
        net(views_of(img, a.views))
    cur = {b: feat[b].clone() for b in BOUNDS}
    for b, Z in cur.items():
        i, j = zip(*itertools.permutations(range(a.views), 2))
        i, j = list(i), list(j)
        d = (Z[i] - Z[j]).norm(dim=1) / Z[j].norm(dim=1)        # relative L2 error
        c = nn.functional.cosine_similarity(Z[i], Z[j], dim=1)  # cosine similarity
        acc[b]["rel_l2"].append(d.mean().item()); acc[b]["cos"].append(c.mean().item())
        if prev is not None:                                     # different-image reference
            P = prev[b]
            d2 = (Z - P).norm(dim=1) / P.norm(dim=1)
            c2 = nn.functional.cosine_similarity(Z, P, dim=1)
            xacc[b]["rel_l2"].append(d2.mean().item()); xacc[b]["cos"].append(c2.mean().item())
    prev = cur
    if (s + 1) % 16 == 0: print("  ...%d/%d samples" % (s + 1, a.samples), flush=True)

def stat(v):
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
    return m, sd

print("\nCIFAR-10 / ResNet-50 (ImageNet-pretrained), RandomResizedCrop(224)+flip")
print("%d samples x %d views; mean +- sd over samples\n" % (a.samples, a.views))
print("%-8s | %-24s | %-24s | %s" % ("boundary", "same sample, diff crop", "", "different image"))
print("%-8s | %-11s %-12s | %-11s %-12s" % ("", "rel L2", "cosine", "rel L2", "cosine"))
print("-" * 74)
res = {}
for b in BOUNDS:
    dm, dsd = stat(acc[b]["rel_l2"]); cm, csd = stat(acc[b]["cos"])
    xdm, xdsd = stat(xacc[b]["rel_l2"]); xcm, xcsd = stat(xacc[b]["cos"])
    print("%-8s | %.3f+-%.3f  %.3f+-%.3f | %.3f+-%.3f  %.3f+-%.3f"
          % (b, dm, dsd, cm, csd, xdm, xdsd, xcm, xcsd))
    res[b] = {"same_sample": {"rel_l2": [dm, dsd], "cos": [cm, csd]},
              "diff_image": {"rel_l2": [xdm, xdsd], "cos": [xcm, xcsd]}}
json.dump({"config": vars(a), "results": res}, open(a.out, "w"), indent=2)
print("\nwrote", a.out)
