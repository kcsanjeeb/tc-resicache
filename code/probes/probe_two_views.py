#!/usr/bin/env python3
"""Emit the raw material for Figure 3.x (two views of one sample).

Uses the same deterministic transform pool as
probe_cache_correctness_calibrated.py, so this figure and the calibrated
heatmap describe the same experiment: resize-256 / crop-224 plus flip,
seed 42, CIFAR-10 sample index 42, ImageNet-pretrained ResNet-50 layer4.

Saves one .npz holding the resized base image, the two crop specs, the two
crops as uint8, the two channel-mean activation maps, and the measured
relative L2 and cosine between the two full activation vectors.
"""
import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from torchvision import models
from torchvision.datasets import CIFAR10

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TransformPool:
    def __init__(self, input_size=224, resize_size=256, n_transforms=16, seed=42):
        self.input_size, self.resize_size = input_size, resize_size
        self.n_transforms = n_transforms
        rng = random.Random(seed)
        max_off = resize_size - input_size
        self.specs = []
        for tid in range(n_transforms):
            top = int(round(max_off * rng.random()))
            left = int(round(max_off * rng.random()))
            flip = bool((tid + rng.randint(0, 1000)) % 2)
            self.specs.append(dict(top=top, left=left, hflip=flip))

    def crop_uint8(self, img):
        return F.resize(img, [self.resize_size, self.resize_size])

    def apply(self, img, tid):
        s = self.specs[tid]
        im = F.resize(img, [self.resize_size, self.resize_size])
        im = F.crop(im, s["top"], s["left"], self.input_size, self.input_size)
        if s["hflip"]:
            im = F.hflip(im)
        raw = F.to_tensor(im)
        return F.normalize(raw.clone(), MEAN, STD), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=42)
    ap.add_argument("--data-root", default="/mnt/data2/data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CIFAR10(a.data_root, train=True, download=False)
    img = ds[a.sample][0].convert("RGB")
    pool = TransformPool()

    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval().to(dev)
    for p in net.parameters():
        p.requires_grad_(False)

    # pick two transforms that differ in flip, for a visible contrast
    f0 = next(t for t in range(16) if not pool.specs[t]["hflip"])
    f1 = next(t for t in range(16) if pool.specs[t]["hflip"])

    store = {}
    h = net.layer4.register_forward_hook(
        lambda m, i, o: store.__setitem__("z", o.detach().float().cpu()))

    crops, acts = [], []
    for tid in (f0, f1):
        x, raw = pool.apply(img, tid)
        with torch.no_grad():
            net(x.unsqueeze(0).to(dev))
        crops.append((raw.numpy() * 255).astype(np.uint8))
        acts.append(store["z"][0].numpy())
    h.remove()

    A, B = torch.tensor(acts[0]).flatten(), torch.tensor(acts[1]).flatten()
    rel_l2 = float((A - B).norm() / B.norm())
    cos = float(NF.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0)).item())

    base = np.array(pool.crop_uint8(img))
    np.savez_compressed(
        a.out,
        base=base,
        crop0=crops[0], crop1=crops[1],
        act0=acts[0].mean(0), act1=acts[1].mean(0),
        spec0=json.dumps(pool.specs[f0]), spec1=json.dumps(pool.specs[f1]),
        tid0=f0, tid1=f1, rel_l2=rel_l2, cos=cos,
        sample=a.sample, boundary="layer4", arch="resnet50",
        act_shape=np.array(acts[0].shape),
    )
    print(json.dumps(dict(tid0=f0, tid1=f1, spec0=pool.specs[f0],
                          spec1=pool.specs[f1], rel_l2=round(rel_l2, 4),
                          cos=round(cos, 4), act_shape=list(acts[0].shape)),
                     indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
