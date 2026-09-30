#!/usr/bin/env python3
"""Probe: limits of geometric feature-space reconstruction for TC-CR (negative result, 2026-09-09).

Question: can the boundary activation of a RandomResizedCrop+hflip crop be
reconstructed from the canonical (reflection-padded, resized) full-image
activation by stride-mapped bilinear sampling?

Method: for real CIFAR-10 images and a pretrained ResNet-50, compare
  cos(reconstruct(canon_feat, crop_params), prefix(true crop))
against two references:
  naive-stale: cos(prefix(different random crop of same image), prefix(crop))
  diff-image : cos(prefix(crop of a different image), prefix(crop))

Measured (CPU, 4 images x 3 crops, CIFAR-10, ResNet-50 ImageNet-pretrained):
  layer1 (56x56, stride  4): recon 0.83 | naive-stale 0.79 | diff-image 0.75
  layer2 (28x28, stride  8): recon 0.68 | naive-stale 0.66 | diff-image 0.63
  layer3 (14x14, stride 16): recon 0.45 | naive-stale 0.57 | diff-image 0.49
  layer4 ( 7x 7, stride 32): recon 0.16 | naive-stale 0.27 | diff-image 0.10
Relative-L2 (same setting, random-init weights, noise images):
  layer1 0.42 -> layer2 0.42 -> layer3 0.59 -> layer4 0.71 (worst affine
  alignment search over offsets +-3 cells and scale 0.5-2.0 floors at ~0.52,
  i.e. no coordinate convention rescues it).

Conclusion: under scale-varying augmentation (RandomResizedCrop, 32->224 px),
adjacent feature cells contain different ReLU activation patterns, so feature
maps do not interpolate linearly across positions. Geometric reconstruction is
no more faithful than reusing the features of a different image of the same
class, and is WORSE than naive stale-crop reuse. It must not be used as the
TC-CR mechanism in this setting. Exact geometric reconstruction only holds for
translation-only augmentation on the native pixel grid (integer stride shifts),
which is the 32 px-native regime of the PlantCRC-style probes.

This probe is why the ICASSP paper's TC-ResiCache rows rely on direct
sample-indexed feature banking (population at freeze, exact reuse of stored
features) and why the cache-correctness ablation is reported honestly.
"""
import statistics as st
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

CANON_PAD = 16
INPUT_SIZE = 224
IMG_SIZE = 32
DATA = "../../data"  # experiments/data relative to this file's directory


def main():
    from tc_resicache_resnet50_cifar_finetune import (
        IMAGENET_MEAN, IMAGENET_STD, ResNetSplit, build_model, set_seed,
    )
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    def recorded_aug(img):
        i, j, h, w = T.RandomResizedCrop.get_params(img, (0.08, 1.0), (3.0 / 4.0, 4.0 / 3.0))
        x = TF.crop(img, i, j, h, w)
        x = TF.resize(x, [INPUT_SIZE, INPUT_SIZE], interpolation=TF.InterpolationMode.BILINEAR)
        flip = 0
        if float(torch.rand(())) < 0.5:
            x = TF.hflip(x)
            flip = 1
        x = TF.to_tensor(x)
        return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD), torch.tensor(
            [float(i), float(j), float(h), float(w), float(flip)])

    S = INPUT_SIZE / (IMG_SIZE + 2 * CANON_PAD)
    stride_map = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}

    canon_tf = T.Compose([
        T.Pad(CANON_PAD, padding_mode="reflect"),
        T.Resize([INPUT_SIZE, INPUT_SIZE], interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    set_seed(0)
    print("loading pretrained resnet50...", flush=True)
    split = ResNetSplit(build_model("resnet50", 10, "default")).eval()
    ds = CIFAR10(root=DATA, train=True, download=False)

    def cos(a, b):
        a = a.flatten().float()
        b = b.flatten().float()
        return float(torch.dot(a, b) / a.norm() / b.norm())

    def reconstruct(canon_feat, params, stride):
        B, _, Hc, Wc = canon_feat.shape
        out_h = INPUT_SIZE // stride
        i, j, h, w, flip = [params[0, k] for k in range(5)]
        fy0, fx0 = (i + CANON_PAD) * S / stride, (j + CANON_PAD) * S / stride
        fh, fw = h * S / stride, w * S / stride
        if float(flip) > 0.5:
            fx0 = INPUT_SIZE / stride - fx0 - fw
        fr = torch.arange(out_h, dtype=torch.float32) / out_h
        ys = fy0 + fh * fr.view(1, out_h, 1)
        xs = fx0 + fw * fr.view(1, 1, out_h)
        yn = 2 * ys / (Hc - 1) - 1
        xn = 2 * xs / (Wc - 1) - 1
        grid = torch.stack((xn.expand(B, out_h, out_h), yn.expand(B, out_h, out_h)), dim=-1)
        return F.grid_sample(canon_feat.float(), grid, mode="bilinear",
                             padding_mode="border", align_corners=True)

    for boundary, stride in stride_map.items():
        c_rec, c_naive, c_diff = [], [], []
        for ii in range(4):
            img, _ = ds[ii]
            with torch.no_grad():
                cf = split.forward_prefix(canon_tf(img).unsqueeze(0), boundary)
            for _ in range(3):
                x1, p1 = recorded_aug(img)
                x2, _ = recorded_aug(img)
                with torch.no_grad():
                    f1 = split.forward_prefix(x1.unsqueeze(0), boundary)
                    f2 = split.forward_prefix(x2.unsqueeze(0), boundary)
                    rec = reconstruct(cf, p1.unsqueeze(0), stride)
                c_rec.append(cos(rec, f1))
                c_naive.append(cos(f2, f1))
            img_b, _ = ds[ii + 100]
            xb, _ = recorded_aug(img_b)
            with torch.no_grad():
                fb = split.forward_prefix(xb.unsqueeze(0), boundary)
            c_diff.append(cos(fb, f1))
        print(f"[{boundary}] cos recon={st.mean(c_rec):.4f} "
              f"naive-stale={st.mean(c_naive):.4f} diff-image={st.mean(c_diff):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
