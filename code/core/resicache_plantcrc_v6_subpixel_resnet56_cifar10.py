#!/usr/bin/env python3
"""
ResiCache PlanTCRC-v6 sub-pixel reconstruction experiment.

v6 keeps the selected final system:
  Freezing: no-reference blockwise ResiCache
  Caching: dense-bank PlanTCRC / plantcrc_v2
  Operating region: layer2.0 and layer2.3, direct activation, not staged

New v6 focus:
  1. Bilinear/sub-pixel crop reconstruction:
       --tcrc_subpixel bilinear
     At layer2, CIFAR random-crop offsets divided by the feature stride can be
     fractional, e.g. 1/2, 3/2. Previous PlanTCRC used floor/round/ceil integer
     mapping. v6 can reconstruct the feature window with torch.grid_sample at
     fractional feature offsets, targeting layer2 transform mismatch.

  2. Softer optional gate:
       --tcrc_gate_threshold 0.55 or 0.60
     v5 showed gate errors around 0.49-0.54; threshold 0.4 rejected most cached
     batches. v6 tests a softer gate that preserves more cache benefit while
     still rejecting worst-case batches.

Goal:
  Reduce the learning-trajectory delay that prevented PlanTCRC-v3 from beating
  baseline TTA, while preserving dense-bank memory-resident cache speed.
"""

import os
import json
import time
import argparse
import warnings
import random
from collections import OrderedDict, defaultdict
import hashlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

DATA_ROOT = '/mnt/data2/data'

# -----------------------------------------------------------------------------
# Blockwise freeze-unit registry
# -----------------------------------------------------------------------------
# ResNet-56 has 27 BasicBlocks.  Earlier ResiCache experiments froze coarse
# modules (layer1, layer2, layer3_0, ...).  For accuracy-vs-freeze-depth
# studies, we expose every BasicBlock as an individual freeze unit.
#
# Backward-compatible aliases are still accepted:
#   layer1   -> layer1.8
#   layer2   -> layer2.8
#   layer3_0 -> layer3_0.2
#   layer3_1 -> layer3_1.1
#   layer3_2 -> layer3_2.1
#   layer3_3 -> layer3_3.1
STAGE_BLOCK_COUNTS = OrderedDict([
    ('layer1', 9),
    ('layer2', 9),
    ('layer3_0', 3),
    ('layer3_1', 2),
    ('layer3_2', 2),
    ('layer3_3', 2),
])

FREEZE_UNITS = ['conv1'] + [
    f'{stage}.{i}'
    for stage, count in STAGE_BLOCK_COUNTS.items()
    for i in range(count)
]

MODULE_FREEZE_ALIASES = {
    stage: f'{stage}.{count - 1}'
    for stage, count in STAGE_BLOCK_COUNTS.items()
}
MODULE_FREEZE_ALIASES['conv1'] = 'conv1'
FREEZE_CHOICES = FREEZE_UNITS + [k for k in MODULE_FREEZE_ALIASES if k != 'conv1']

# Approximate cumulative compute saved when caching after a given freeze unit.
# These preserve the old coarse estimates while interpolating within each group.
COARSE_COMPUTE_FRACTION = OrderedDict([
    ('conv1', 0.02),
    ('layer1', 0.12),
    ('layer2', 0.35),
    ('layer3_0', 0.55),
    ('layer3_1', 0.70),
    ('layer3_2', 0.85),
    ('layer3_3', 0.97),
])

def normalize_freeze_name(name):
    """Map a coarse module alias to the corresponding last BasicBlock."""
    return MODULE_FREEZE_ALIASES.get(str(name), str(name))

def freeze_unit_index(name):
    return FREEZE_UNITS.index(normalize_freeze_name(name))

def parse_block_unit(name):
    name = normalize_freeze_name(name)
    if name == 'conv1':
        return 'conv1', None
    if '.' not in name:
        raise ValueError(f'Unknown freeze unit: {name}')
    stage, idx = name.rsplit('.', 1)
    return stage, int(idx)

def build_blockwise_compute_fractions():
    out = {'conv1': COARSE_COMPUTE_FRACTION['conv1']}
    prev = COARSE_COMPUTE_FRACTION['conv1']
    for stage, count in STAGE_BLOCK_COUNTS.items():
        end = COARSE_COMPUTE_FRACTION[stage]
        for i in range(count):
            out[f'{stage}.{i}'] = prev + (end - prev) * float(i + 1) / float(count)
        prev = end
    return out

BLOCKWISE_COMPUTE_FRACTION = build_blockwise_compute_fractions()

def ordered_frozen_layers(frozen):
    frozen = {normalize_freeze_name(x) for x in frozen}
    return [u for u in FREEZE_UNITS if u in frozen]

# =============================================================================
# RESNET-56
# =============================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet56, self).__init__()
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
        self.frozen_layers = set()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
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
        out = self.fc(out)
        return out

    def _iter_freeze_units_from(self, out, start_after=None):
        """Yield (unit_name, output) while forwarding through blockwise units."""
        seen_start = (start_after is None)
        if start_after == 'conv1':
            seen_start = True
        for stage, count in STAGE_BLOCK_COUNTS.items():
            seq = getattr(self, stage)
            for i in range(count):
                unit = f'{stage}.{i}'
                if not seen_start:
                    if unit == start_after:
                        seen_start = True
                    continue
                out = seq[i](out)
                yield unit, out

    def forward_until_layer(self, x, layer_name):
        """Forward to a blockwise freeze unit and return its activation.

        Accepts both exact BasicBlock names (e.g. layer2.3) and old coarse
        aliases (e.g. layer2 -> layer2.8).
        """
        layer_name = normalize_freeze_name(layer_name)
        out = torch.relu(self.bn1(self.conv1(x)))
        if layer_name == 'conv1':
            return out
        for unit, out in self._iter_freeze_units_from(out):
            if unit == layer_name:
                return out
        raise ValueError(f"Unknown freeze unit: {layer_name}")

    def forward_from_layer(self, x, layer_name):
        """Continue the forward pass after a blockwise freeze unit."""
        layer_name = normalize_freeze_name(layer_name)
        out = x
        if layer_name not in FREEZE_UNITS:
            raise ValueError(f"Unknown freeze unit: {layer_name}")
        for _, out in self._iter_freeze_units_from(out, start_after=layer_name):
            pass
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _module_for_freeze_unit(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        if layer_name == 'conv1':
            return nn.Sequential(self.conv1, self.bn1)
        stage, idx = parse_block_unit(layer_name)
        return getattr(self, stage)[idx]

    def freeze_layer(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        self.frozen_layers.add(layer_name)
        if layer_name == 'conv1':
            for p in [self.conv1, self.bn1]:
                for param in p.parameters():
                    param.requires_grad = False
        else:
            mod = self._module_for_freeze_unit(layer_name)
            for param in mod.parameters():
                param.requires_grad = False

    def set_frozen_eval(self):
        """Put frozen BN/block modules in eval mode during training."""
        for lname in self.frozen_layers:
            lname = normalize_freeze_name(lname)
            if lname == 'conv1':
                self.bn1.eval()
            else:
                self._module_for_freeze_unit(lname).eval()

    def unfreeze_all(self):
        self.frozen_layers.clear()
        for p in self.parameters(): p.requires_grad = True


# =============================================================================
# DELTA-RESIDUAL SAMPLE-LEVEL CACHE (shuffle=True compatible)
# =============================================================================
class DeltaResidualSampleCache:
    """Sample-level activation cache with delta-residual correction.

    Each entry stores base and delta. On hit, returns base + delta.
    Refresh/probe batches recompute the true current activation and update delta
    as an EMA of current - base. Statistics are counted per sample.
    """

    def __init__(self, cache_dir='./resicache_delta_samples', gpu_cache_size=5000,
                 device='cuda', residual_momentum=0.90, residual_clip=5.0, disk_cache=True):
        self.cache_dir = cache_dir
        self.device = device
        self.gpu_cache_size = gpu_cache_size
        self.residual_momentum = residual_momentum
        self.residual_clip = residual_clip
        self.disk_cache = disk_cache
        self.gpu_cache = OrderedDict()
        os.makedirs(cache_dir, exist_ok=True)
        self.sample_accesses = 0
        self.sample_hits = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.sample_misses = 0
        self.corrected_reuses = 0
        self.delta_updates = 0
        self.refresh_batches = 0
        self.delta_norm_sum = 0.0

    def _get_path(self, sample_id, layer_name):
        subdir = str(int(sample_id) // 1000)
        dir_path = os.path.join(self.cache_dir, layer_name, subdir)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"{int(sample_id)}.pt")

    def _new_entry(self, base_activation):
        base = base_activation.detach().to(self.device)
        return {'base': base, 'delta': torch.zeros_like(base, device=self.device), 'updates': 0}

    def _entry_to_device(self, entry):
        entry['base'] = entry['base'].to(self.device, non_blocking=True)
        entry['delta'] = entry['delta'].to(self.device, non_blocking=True)
        return entry

    def _add_to_gpu_cache(self, key, entry):
        self.gpu_cache[key] = self._entry_to_device(entry)
        self.gpu_cache.move_to_end(key)
        while len(self.gpu_cache) > self.gpu_cache_size:
            del self.gpu_cache[next(iter(self.gpu_cache))]

    def _load_entry(self, sample_id, layer_name):
        key = (int(sample_id), layer_name)
        if key in self.gpu_cache:
            self.memory_hits += 1
            self.gpu_cache.move_to_end(key)
            return self.gpu_cache[key]
        if self.disk_cache:
            path = self._get_path(sample_id, layer_name)
            if os.path.exists(path):
                entry = torch.load(path, map_location=self.device)
                if isinstance(entry, torch.Tensor):
                    entry = self._new_entry(entry)
                entry = self._entry_to_device(entry)
                self.disk_hits += 1
                self._add_to_gpu_cache(key, entry)
                return entry
        return None

    def _save_entry(self, sample_id, layer_name, entry):
        key = (int(sample_id), layer_name)
        self._add_to_gpu_cache(key, entry)
        if self.disk_cache:
            path = self._get_path(sample_id, layer_name)
            torch.save({
                'base': entry['base'].detach().cpu(),
                'delta': entry['delta'].detach().cpu(),
                'updates': int(entry.get('updates', 0)),
            }, path)

    def get_batch(self, sample_ids, layer_name):
        entries = []
        for sid in sample_ids:
            self.sample_accesses += 1
            entry = self._load_entry(sid, layer_name)
            if entry is None:
                self.sample_misses += 1
                return None
            self.sample_hits += 1
            entries.append(entry)
        corrected = [entry['base'] + entry['delta'] for entry in entries]
        self.corrected_reuses += len(entries)
        return torch.stack(corrected)

    def put_batch(self, sample_ids, layer_name, activation_batch):
        for i, sid in enumerate(sample_ids):
            self._save_entry(sid, layer_name, self._new_entry(activation_batch[i]))

    def refresh_batch(self, sample_ids, layer_name, current_activation_batch):
        self.refresh_batches += 1
        for i, sid in enumerate(sample_ids):
            current = current_activation_batch[i].detach().to(self.device)
            entry = self._load_entry(sid, layer_name)
            if entry is None:
                entry = self._new_entry(current)
            else:
                observed_delta = current - entry['base']
                if self.residual_clip is not None and self.residual_clip > 0:
                    base_norm = entry['base'].norm().clamp_min(1e-8)
                    delta_norm = observed_delta.norm()
                    max_norm = self.residual_clip * base_norm
                    if delta_norm > max_norm:
                        observed_delta = observed_delta * (max_norm / delta_norm)
                entry['delta'] = (
                    self.residual_momentum * entry['delta']
                    + (1.0 - self.residual_momentum) * observed_delta
                ).detach()
                entry['updates'] = int(entry.get('updates', 0)) + 1
                self.delta_updates += 1
                self.delta_norm_sum += float(entry['delta'].norm().item())
            self._save_entry(sid, layer_name, entry)

    def clear(self):
        self.gpu_cache.clear()
        import glob
        files_removed = 0
        for f in glob.glob(os.path.join(self.cache_dir, "**/*.pt"), recursive=True):
            try:
                os.remove(f)
                files_removed += 1
            except Exception:
                pass
        self.sample_accesses = 0
        self.sample_hits = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.sample_misses = 0
        self.corrected_reuses = 0
        self.delta_updates = 0
        self.refresh_batches = 0
        self.delta_norm_sum = 0.0
        print(f"  [Cache] Cleared GPU cache and {files_removed} disk files")

    def get_stats(self):
        hit_rate = self.sample_hits / max(self.sample_accesses, 1)
        return {
            'hit_rate': hit_rate,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'misses': self.sample_misses,
            'sample_accesses': self.sample_accesses,
            'sample_hits': self.sample_hits,
            'corrected_reuses': self.corrected_reuses,
            'delta_updates': self.delta_updates,
            'refresh_batches': self.refresh_batches,
            'avg_delta_norm': self.delta_norm_sum / max(self.delta_updates, 1),
        }



class TransformConsistentResidualCache:
    """Transform-consistent residual cache (TCRC).

    TCRC addresses the random-augmentation cache mismatch.  Instead of caching an
    activation for each random crop, it caches the frozen-prefix activation of a
    canonical padded image.  At reuse time, the cache derives the activation for
    the current crop by cropping the cached feature map in feature space.  For
    horizontal flip, it caches a separate canonical activation of the horizontally
    flipped padded image, because arbitrary CNN filters are not exactly flip
    equivariant.

    This is an experimental prototype intended to test the hypothesis:

        high reuse of sample-id cache + correctness of augmentation-aware cache
        can be approached by canonical feature caching.

    Notes:
      - Crop equivariance is approximate because of padding and boundary effects.
      - A global residual per layer is updated from probe/refresh batches to
        correct systematic transform error.
      - No disk cache is used by default; this is a memory-centric cache.
    """

    def __init__(self, gpu_cache_size=50000, device='cuda', residual_momentum=0.90,
                 residual_clip=5.0, crop_rounding='floor', probe_error_threshold=0.20,
                 subpixel_mode='none'):
        self.device = device
        self.gpu_cache_size = int(gpu_cache_size)
        self.residual_momentum = float(residual_momentum)
        self.residual_clip = residual_clip
        self.crop_rounding = crop_rounding
        self.subpixel_mode = str(subpixel_mode)
        self.probe_error_threshold = float(probe_error_threshold)
        self.gpu_cache = OrderedDict()
        self.global_delta = {}
        # Optional state-conditioned residual correction.  When enabled, this
        # keeps a residual table keyed by (layer, feature-y, feature-x, flip).
        # It targets crop/flip-dependent TCRC transform error, especially in
        # stride-2 layer2.* boundaries.
        self.use_state_delta = False
        self.state_delta = {}

        self.sample_accesses = 0
        self.sample_hits = 0
        self.sample_misses = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.corrected_reuses = 0
        self.delta_updates = 0
        self.refresh_batches = 0
        self.delta_norm_sum = 0.0
        self.tcrc_populates = 0
        self.tcrc_transforms = 0
        self.tcrc_probe_updates = 0
        self.tcrc_probe_error_sum = 0.0
        self.tcrc_probe_error_max = 0.0
        self.tcrc_disabled_due_error = False
        self.tcrc_gate_accepts = 0
        self.tcrc_gate_fallbacks = 0
        self.tcrc_gate_error_sum = 0.0

    def clear(self):
        self.gpu_cache.clear()
        self.global_delta.clear()
        self.state_delta.clear()
        self.sample_accesses = 0
        self.sample_hits = 0
        self.sample_misses = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.corrected_reuses = 0
        self.delta_updates = 0
        self.refresh_batches = 0
        self.delta_norm_sum = 0.0
        self.tcrc_populates = 0
        self.tcrc_transforms = 0
        self.tcrc_probe_updates = 0
        self.tcrc_probe_error_sum = 0.0
        self.tcrc_probe_error_max = 0.0
        self.tcrc_disabled_due_error = False
        self.tcrc_gate_accepts = 0
        self.tcrc_gate_fallbacks = 0
        self.tcrc_gate_error_sum = 0.0
        print("  [TCRC] Cleared canonical activation cache")

    def _cache_key(self, sample_id, layer_name, flip):
        # flip-specific canonical activations are necessary because CNNs are not
        # strictly equivariant to horizontal reflection with arbitrary filters.
        return (int(sample_id), normalize_freeze_name(layer_name), int(flip))

    def _add(self, key, activation):
        self.gpu_cache[key] = activation.detach().to(self.device)
        self.gpu_cache.move_to_end(key)
        while len(self.gpu_cache) > self.gpu_cache_size:
            del self.gpu_cache[next(iter(self.gpu_cache))]

    def _layer_stride(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        if layer_name == 'conv1' or layer_name.startswith('layer1.'):
            return 1
        if layer_name.startswith('layer2.'):
            return 2
        return 4

    def _target_hw(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        if layer_name == 'conv1' or layer_name.startswith('layer1.'):
            return 32, 32
        if layer_name.startswith('layer2.'):
            return 16, 16
        return 8, 8

    def _feature_offset(self, v, stride):
        x = float(v) / float(stride)
        if self.crop_rounding == 'round':
            return int(round(x))
        if self.crop_rounding == 'ceil':
            return int(np.ceil(x))
        return int(np.floor(x))

    def _feature_offset_float(self, v, stride):
        # v6: for bilinear/sub-pixel reconstruction preserve fractional feature
        # offsets instead of forcing floor/round/ceil integer positions.
        return float(v) / float(stride)

    def _use_bilinear_subpixel(self):
        return getattr(self, 'subpixel_mode', 'none') == 'bilinear'

    def _bilinear_window_from_canonical(self, canonical_batch, layer_name, crop_i, crop_j, flips):
        """Sub-pixel crop reconstruction using grid_sample.

        Previous PlanTCRC mapped an image crop offset to an integer feature offset
        with floor/round/ceil.  For layer2.* the stride is 2, so CIFAR crop
        offsets 1,3,5,7 correspond to half-pixel feature offsets.  This method
        samples the canonical feature map at fractional offsets using bilinear
        interpolation.  It is intended to reduce transform mismatch at layer2.
        """
        if canonical_batch.dim() != 4:
            raise RuntimeError(f'Expected 4D canonical activation batch, got {tuple(canonical_batch.shape)}')
        B, C, Hc, Wc = canonical_batch.shape
        h, w = self._target_hw(layer_name)
        stride = self._layer_stride(layer_name)

        ci = crop_i.to(self.device, non_blocking=True).float()
        cj = crop_j.to(self.device, non_blocking=True).float()
        fl = flips.to(self.device, non_blocking=True).long()
        cj_eff = torch.where(fl == 1, torch.full_like(cj, 8.0) - cj, cj)

        y0 = (ci / float(stride)).clamp(0.0, max(float(Hc - h), 0.0))
        x0 = (cj_eff / float(stride)).clamp(0.0, max(float(Wc - w), 0.0))

        yy = torch.arange(h, device=self.device, dtype=canonical_batch.dtype).view(1, h, 1)
        xx = torch.arange(w, device=self.device, dtype=canonical_batch.dtype).view(1, 1, w)
        y = y0.to(dtype=canonical_batch.dtype).view(B, 1, 1) + yy
        x = x0.to(dtype=canonical_batch.dtype).view(B, 1, 1) + xx

        if Hc > 1:
            yn = 2.0 * y / float(Hc - 1) - 1.0
        else:
            yn = torch.zeros_like(y)
        if Wc > 1:
            xn = 2.0 * x / float(Wc - 1) - 1.0
        else:
            xn = torch.zeros_like(x)

        grid = torch.stack((xn.expand(B, h, w), yn.expand(B, h, w)), dim=-1)
        out = F.grid_sample(canonical_batch, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return out

    def _state_key(self, layer_name, crop_i, crop_j, flip):
        """Residual key using feature-space crop offset and flip state."""
        layer_name = normalize_freeze_name(layer_name)
        stride = self._layer_stride(layer_name)
        ci = int(crop_i)
        cj = int(crop_j)
        fl = int(flip)
        if fl == 1:
            cj = 8 - cj
        y0 = self._feature_offset(ci, stride)
        x0 = self._feature_offset(cj, stride)
        return (layer_name, int(y0), int(x0), fl)

    def _state_delta_batch(self, layer_name, crop_i, crop_j, flips, ref_tensor):
        if not self.use_state_delta or not self.state_delta:
            return None
        outs = []
        any_found = False
        for pos in range(ref_tensor.size(0)):
            key = self._state_key(layer_name, int(crop_i[pos]), int(crop_j[pos]), int(flips[pos]))
            delta = self.state_delta.get(key, None)
            if delta is None:
                outs.append(torch.zeros_like(ref_tensor[pos]))
            else:
                any_found = True
                outs.append(delta.to(ref_tensor.device, dtype=ref_tensor.dtype))
        if not any_found:
            return None
        return torch.stack(outs, dim=0)

    def _update_state_delta(self, layer_name, crop_i, crop_j, flips, residual):
        if not self.use_state_delta:
            return
        # Update per-state residual using EMA.  Number of crop/flip states is
        # small for CIFAR random crop: 9 x 9 x 2, and fewer after stride mapping.
        groups = defaultdict(list)
        for pos in range(residual.size(0)):
            key = self._state_key(layer_name, int(crop_i[pos]), int(crop_j[pos]), int(flips[pos]))
            groups[key].append(pos)
        for key, positions in groups.items():
            idx = torch.as_tensor(positions, device=residual.device, dtype=torch.long)
            upd = residual.index_select(0, idx).mean(dim=0).detach()
            if self.residual_clip is not None and self.residual_clip > 0:
                upd_norm = upd.norm()
                # Local residuals should be conservative; use self.residual_clip
                # as an absolute multiplier over the residual mean norm.
                max_norm = self.residual_clip * residual.mean(dim=0).norm().clamp_min(1e-8)
                if upd_norm > max_norm:
                    upd = upd * (max_norm / upd_norm)
            old = self.state_delta.get(key, None)
            if old is None:
                new_delta = (1.0 - self.residual_momentum) * upd
            else:
                new_delta = self.residual_momentum * old.to(residual.device) + (1.0 - self.residual_momentum) * upd
            self.state_delta[key] = new_delta.detach()

    def _transform_one(self, canonical_activation, layer_name, crop_i, crop_j, flip):
        """Crop the canonical feature map to match current random crop.

        Augmentation order is pad -> crop -> hflip.  If flip=1 and we use a
        canonical activation of the flipped padded image, the equivalent crop
        start in the flipped padded image is 8 - crop_j for CIFAR-10 padding=4.
        """
        stride = self._layer_stride(layer_name)
        h, w = self._target_hw(layer_name)
        ci = int(crop_i)
        cj = int(crop_j)
        if int(flip) == 1:
            cj = 8 - cj
        y0 = self._feature_offset(ci, stride)
        x0 = self._feature_offset(cj, stride)
        y0 = max(0, min(y0, canonical_activation.shape[-2] - h))
        x0 = max(0, min(x0, canonical_activation.shape[-1] - w))
        return canonical_activation[..., y0:y0 + h, x0:x0 + w]

    def _populate_missing(self, sample_ids, layer_name, canonical_inputs, flips, model):
        """Ensure canonical activations for the current batch exist in the LRU cache.

        Important: the TCRC cache can store two canonical variants per sample
        because flip=0 and flip=1 are cached separately.  With a finite LRU
        budget, adding new entries can evict old entries that are also needed by
        the same batch.  We therefore first mark existing entries from the
        current batch as recently used, and _raw_transform_batch has an extra
        per-sample fallback for entries evicted under tight cache capacity.
        """
        missing_positions = []
        for pos, sid in enumerate(sample_ids):
            key = self._cache_key(sid, layer_name, int(flips[pos]))
            if key in self.gpu_cache:
                self.gpu_cache.move_to_end(key)
            else:
                missing_positions.append(pos)
        if not missing_positions:
            return
        # Populate missing canonical activations as a batch.  Frozen prefix is not
        # trained, so no gradient is needed for canonical cache construction.
        canon = canonical_inputs[missing_positions].to(self.device, non_blocking=True)
        flip_tensor = flips[missing_positions].to(self.device)
        # Flip the full padded canonical input before forwarding when the current
        # augmentation requires horizontal flip.
        if flip_tensor.numel() > 0:
            flip_mask = flip_tensor.bool()
            if flip_mask.any():
                canon = canon.clone()
                canon[flip_mask] = torch.flip(canon[flip_mask], dims=[-1])
        with torch.no_grad():
            acts = model.forward_until_layer(canon, layer_name).detach()
        for local_i, pos in enumerate(missing_positions):
            key = self._cache_key(sample_ids[pos], layer_name, int(flips[pos]))
            self._add(key, acts[local_i])
            self.sample_misses += 1
            self.tcrc_populates += 1

    def _populate_one_missing(self, pos, sample_ids, layer_name, canonical_inputs, flips, model):
        """Robust fallback for a single entry if LRU eviction removed it."""
        canon = canonical_inputs[pos:pos + 1].to(self.device, non_blocking=True)
        flip_val = int(flips[pos])
        if flip_val == 1:
            canon = torch.flip(canon, dims=[-1])
        with torch.no_grad():
            act = model.forward_until_layer(canon, layer_name).detach()[0]
        key = self._cache_key(sample_ids[pos], layer_name, flip_val)
        self._add(key, act)
        self.sample_misses += 1
        self.tcrc_populates += 1
        return self.gpu_cache[key]

    def _raw_transform_batch(self, sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model):
        self._populate_missing(sample_ids, layer_name, canonical_inputs, flips, model)
        outs = []
        for pos, sid in enumerate(sample_ids):
            key = self._cache_key(sid, layer_name, int(flips[pos]))
            can = self.gpu_cache.get(key, None)
            if can is None:
                # This can happen when gpu_cache_size is smaller than the active
                # canonical working set and LRU eviction removes an entry between
                # batch population and use.  Recompute only this canonical entry
                # instead of failing the run.
                can = self._populate_one_missing(pos, sample_ids, layer_name, canonical_inputs, flips, model)
            else:
                self.gpu_cache.move_to_end(key)
                self.memory_hits += 1
                self.sample_hits += 1
            out = self._transform_one(can, layer_name, int(crop_i[pos]), int(crop_j[pos]), int(flips[pos]))
            outs.append(out)
        self.tcrc_transforms += len(outs)
        return torch.stack(outs, dim=0)

    def get_batch(self, sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model):
        self.sample_accesses += len(sample_ids)
        if self.tcrc_disabled_due_error:
            return None
        raw = self._raw_transform_batch(sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model)
        delta = self.global_delta.get(layer_name, None)
        if delta is not None:
            raw = raw + delta.to(raw.device).unsqueeze(0)
        state_delta = self._state_delta_batch(layer_name, crop_i, crop_j, flips, raw)
        if state_delta is not None:
            raw = raw + state_delta
        self.corrected_reuses += raw.size(0)
        return raw

    def put_batch(self, sample_ids, layer_name, canonical_inputs, crop_i=None, crop_j=None, flips=None, model=None):
        # Warmup population path.  No transformed output is returned.
        if flips is None or model is None:
            return
        self._populate_missing(sample_ids, layer_name, canonical_inputs, flips, model)

    def refresh_batch(self, sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model, true_activation_batch):
        """Probe transform error and update a global residual correction."""
        self.refresh_batches += 1
        with torch.no_grad():
            raw = self._raw_transform_batch(sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model)
            true = true_activation_batch.detach().to(self.device)
            if true.shape != raw.shape:
                return
            residual = true - raw
            # Relative error before applying/update residual.
            rel = residual.flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1).clamp_min(1e-8)
            mean_err = float(rel.mean().item())
            max_err = float(rel.max().item())
            self.tcrc_probe_updates += 1
            self.tcrc_probe_error_sum += mean_err
            self.tcrc_probe_error_max = max(self.tcrc_probe_error_max, max_err)
            # Update global residual.  This corrects systematic feature-crop
            # mismatch, especially boundary error.
            update = residual.mean(dim=0).detach()
            if self.residual_clip is not None and self.residual_clip > 0:
                base_norm = raw.mean(dim=0).norm().clamp_min(1e-8)
                max_norm = self.residual_clip * base_norm
                upd_norm = update.norm()
                if upd_norm > max_norm:
                    update = update * (max_norm / upd_norm)
            old = self.global_delta.get(layer_name, None)
            if old is None:
                new_delta = (1.0 - self.residual_momentum) * update
            else:
                new_delta = self.residual_momentum * old.to(self.device) + (1.0 - self.residual_momentum) * update
            self.global_delta[layer_name] = new_delta.detach()
            self._update_state_delta(layer_name, crop_i, crop_j, flips, residual)
            self.delta_updates += 1
            self.delta_norm_sum += float(new_delta.norm().item())
            # Safety: if transform error is very large, cache reuse can be disabled
            # for the rest of the run.  Set threshold <=0 to disable this guard.
            if self.probe_error_threshold > 0 and mean_err > self.probe_error_threshold:
                print(f"  [TCRC] Warning: mean transform error {mean_err:.3f} exceeds threshold {self.probe_error_threshold:.3f}; disabling TCRC reuse")
                self.tcrc_disabled_due_error = True

    def get_stats(self):
        hit_rate = self.sample_hits / max(self.sample_accesses, 1)
        return {
            'hit_rate': hit_rate,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'misses': self.sample_misses,
            'sample_accesses': self.sample_accesses,
            'sample_hits': self.sample_hits,
            'corrected_reuses': self.corrected_reuses,
            'delta_updates': self.delta_updates,
            'refresh_batches': self.refresh_batches,
            'avg_delta_norm': self.delta_norm_sum / max(self.delta_updates, 1),
            'tcrc_populates': self.tcrc_populates,
            'tcrc_transforms': self.tcrc_transforms,
            'tcrc_probe_updates': self.tcrc_probe_updates,
            'tcrc_mean_probe_error': self.tcrc_probe_error_sum / max(self.tcrc_probe_updates, 1),
            'tcrc_max_probe_error': self.tcrc_probe_error_max,
            'tcrc_disabled_due_error': self.tcrc_disabled_due_error,
            'tcrc_global_delta_layers': sorted(list(self.global_delta.keys())),
            'tcrc_state_delta_enabled': bool(self.use_state_delta),
            'tcrc_state_delta_entries': len(self.state_delta),
            'tcrc_gate_accepts': self.tcrc_gate_accepts,
            'tcrc_gate_fallbacks': self.tcrc_gate_fallbacks,
            'tcrc_gate_mean_error': self.tcrc_gate_error_sum / max(self.tcrc_gate_accepts + self.tcrc_gate_fallbacks, 1),
            'tcrc_subpixel_mode': getattr(self, 'subpixel_mode', 'none'),
        }



class PlanTransformConsistentResidualCache(TransformConsistentResidualCache):
    """PlanTCRC: vectorized transform-plan version of TCRC.

    The original TCRC prototype reconstructs each cached activation by looping
    over samples in Python and slicing one canonical feature map at a time.  This
    class keeps the same cache semantics and residual/probe logic, but replaces
    the per-sample crop loop with a batched gather operation on the GPU.

    Core idea:
      1. Cache canonical padded-image activations as before.
      2. Stack canonical activations for the current batch.
      3. Convert every crop/flip state into feature-space offsets as tensors.
      4. Use one torch.gather over flattened spatial dimensions to extract all
         crop windows in a single batched operation.

    This is intended as a drop-in faster implementation for --cache_impl plantcrc.
    It should produce the same crop-window result as the old TCRC transform path,
    up to numerical equivalence, while reducing tcrc_get_transform_batch time.
    """

    def __init__(self, *args, plan_dtype='same', **kwargs):
        super().__init__(*args, **kwargs)
        self.plan_dtype = plan_dtype
        self.plan_vectorized_batches = 0
        self.plan_vectorized_samples = 0
        self.plan_fallback_single_populates = 0

    def clear(self):
        super().clear()
        self.plan_vectorized_batches = 0
        self.plan_vectorized_samples = 0
        self.plan_fallback_single_populates = 0
        print("  [PlanTCRC] Cleared vectorized transform-plan cache")

    def _vector_offsets(self, values, stride):
        vals = values.to(self.device, non_blocking=True).float() / float(stride)
        if self.crop_rounding == 'round':
            return torch.round(vals).long()
        if self.crop_rounding == 'ceil':
            return torch.ceil(vals).long()
        return torch.floor(vals).long()

    def _stack_canonical_batch(self, sample_ids, layer_name, canonical_inputs, flips, model):
        """Populate missing entries and stack canonical feature maps for a batch.

        We still need a small Python loop for dictionary lookup because the
        current cache is an OrderedDict/LRU.  The expensive crop reconstruction is
        no longer done in Python; it is vectorized after stacking.
        """
        self._populate_missing(sample_ids, layer_name, canonical_inputs, flips, model)
        cans = []
        for pos, sid in enumerate(sample_ids):
            key = self._cache_key(sid, layer_name, int(flips[pos]))
            can = self.gpu_cache.get(key, None)
            if can is None:
                can = self._populate_one_missing(pos, sample_ids, layer_name, canonical_inputs, flips, model)
                self.plan_fallback_single_populates += 1
            else:
                self.gpu_cache.move_to_end(key)
                self.memory_hits += 1
                self.sample_hits += 1
            cans.append(can)
        return torch.stack(cans, dim=0)

    def _raw_transform_batch(self, sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model):
        """Vectorized crop/flip feature-space reconstruction.

        Equivalent target to the original _raw_transform_batch:
          output[b] = _transform_one(canonical[b], layer_name, crop_i[b], crop_j[b], flip[b])

        But instead of slicing one sample at a time, this builds the spatial
        gather indices for the whole batch and performs a single torch.gather.
        """
        canonical_batch = self._stack_canonical_batch(sample_ids, layer_name, canonical_inputs, flips, model)
        if canonical_batch.dim() != 4:
            raise RuntimeError(f"PlanTCRC expected 4D canonical activation batch, got {tuple(canonical_batch.shape)}")

        B, C, Hc, Wc = canonical_batch.shape
        h, w = self._target_hw(layer_name)
        stride = self._layer_stride(layer_name)

        if self._use_bilinear_subpixel():
            out = self._bilinear_window_from_canonical(canonical_batch, layer_name, crop_i, crop_j, flips)
            self.tcrc_transforms += B
            self.plan_vectorized_batches += 1
            self.plan_vectorized_samples += B
            return out

        ci = crop_i.to(self.device, non_blocking=True).long()
        cj = crop_j.to(self.device, non_blocking=True).long()
        fl = flips.to(self.device, non_blocking=True).long()

        # If the current augmented sample is horizontally flipped, TCRC uses a
        # flipped canonical input.  In the flipped padded coordinate system the
        # equivalent crop column is 8 - crop_j for CIFAR-10 padding=4.
        cj_eff = torch.where(fl == 1, torch.full_like(cj, 8) - cj, cj)

        y0 = self._vector_offsets(ci, stride).clamp(0, max(int(Hc) - int(h), 0))
        x0 = self._vector_offsets(cj_eff, stride).clamp(0, max(int(Wc) - int(w), 0))

        yy = torch.arange(h, device=self.device, dtype=torch.long).view(1, h, 1)
        xx = torch.arange(w, device=self.device, dtype=torch.long).view(1, 1, w)
        spatial_idx = (y0.view(B, 1, 1) + yy) * Wc + (x0.view(B, 1, 1) + xx)
        spatial_idx = spatial_idx.reshape(B, 1, h * w).expand(B, C, h * w)

        flat = canonical_batch.reshape(B, C, Hc * Wc)
        out = torch.gather(flat, dim=2, index=spatial_idx).reshape(B, C, h, w)

        self.tcrc_transforms += B
        self.plan_vectorized_batches += 1
        self.plan_vectorized_samples += B
        return out

    def get_stats(self):
        stats = super().get_stats()
        stats.update({
            'plantcrc_vectorized_batches': self.plan_vectorized_batches,
            'plantcrc_vectorized_samples': self.plan_vectorized_samples,
            'plantcrc_fallback_single_populates': self.plan_fallback_single_populates,
        })
        return stats


class BankPlanTransformConsistentResidualCache(PlanTransformConsistentResidualCache):
    """PlanTCRC-v2: dense GPU-bank + bucketed transform plan.

    Motivation:
      PlanTCRC-v1 removes the per-sample crop loop, but it still performs
      OrderedDict/LRU lookup for every sample and builds a large gather index on
      every batch.  For CIFAR-style datasets the working set is small enough to
      store the currently active canonical activations in a dense GPU bank.

    Key differences from PlanTCRC-v1:
      1. Dense bank indexed by (flip, sample_id), avoiding per-sample dict lookup.
      2. Valid mask records which canonical activations have been populated.
      3. Crop reconstruction is bucketed by feature-space crop offset.  Samples
         sharing the same crop window are copied using one sliced tensor view,
         avoiding a large B*C*H*W LongTensor gather index.

    This trades GPU memory for lower transform overhead.  For CIFAR-10 layer1
    the fp32 dense bank is roughly 10GB for two flip states; fp16 is roughly 5GB.
    Use --plantcrc_bank_dtype fp16 if memory is tight.  Outputs are cast back to
    fp32 before being passed into the fp32 model.
    """

    def __init__(self, *args, num_samples=50000, bank_dtype='fp32', use_bucket_transform=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = int(num_samples)
        self.bank_dtype = str(bank_dtype)
        self.use_bucket_transform = bool(use_bucket_transform)
        self.banks = {}
        self.bank_valid = {}
        self.bank_populated = {}
        self.bank_allocated_bytes = 0
        self.bank_hits = 0
        self.bank_misses = 0
        self.bank_bucket_batches = 0
        self.bank_bucket_groups = 0
        self.bank_bucket_samples = 0

    def clear(self):
        super().clear()
        self.banks.clear()
        self.bank_valid.clear()
        self.bank_populated.clear()
        self.bank_allocated_bytes = 0
        self.bank_hits = 0
        self.bank_misses = 0
        self.bank_bucket_batches = 0
        self.bank_bucket_groups = 0
        self.bank_bucket_samples = 0
        print("  [PlanTCRC-v2] Cleared dense GPU-bank transform cache")

    def _storage_dtype(self, acts):
        if self.bank_dtype == 'fp16':
            return torch.float16
        if self.bank_dtype == 'bf16':
            return torch.bfloat16
        if self.bank_dtype == 'same':
            return acts.dtype
        return torch.float32

    def _sid_flip_tensors(self, sample_ids, flips):
        if torch.is_tensor(sample_ids):
            sid = sample_ids.to(self.device, non_blocking=True).long()
        else:
            sid = torch.as_tensor(sample_ids, device=self.device, dtype=torch.long)
        fl = flips.to(self.device, non_blocking=True).long()
        return sid, fl

    def _ensure_bank(self, layer_name, acts):
        layer_name = normalize_freeze_name(layer_name)
        if layer_name in self.banks:
            return
        dtype = self._storage_dtype(acts)
        shape = (2, self.num_samples) + tuple(acts.shape[1:])
        bank = torch.empty(shape, device=self.device, dtype=dtype)
        valid = torch.zeros((2, self.num_samples), device=self.device, dtype=torch.bool)
        self.banks[layer_name] = bank
        self.bank_valid[layer_name] = valid
        self.bank_populated[layer_name] = 0
        bytes_per = torch.empty((), dtype=dtype).element_size()
        alloc_bytes = int(torch.tensor(shape).prod().item()) * bytes_per
        self.bank_allocated_bytes += alloc_bytes
        print(f"  [PlanTCRC-v2] Allocated dense bank for {layer_name}: shape={shape}, dtype={dtype}, approx={alloc_bytes/1e9:.2f} GB")

    def _populate_missing(self, sample_ids, layer_name, canonical_inputs, flips, model):
        layer_name = normalize_freeze_name(layer_name)
        sid, fl = self._sid_flip_tensors(sample_ids, flips)

        if layer_name in self.bank_valid:
            valid_now = self.bank_valid[layer_name][fl, sid]
            missing_positions = torch.nonzero(~valid_now, as_tuple=False).flatten()
        else:
            missing_positions = torch.arange(sid.numel(), device=self.device, dtype=torch.long)

        if missing_positions.numel() == 0:
            return

        # Forward the missing canonical padded inputs as one batch.
        miss_cpu = missing_positions.detach().cpu().tolist()
        canon = canonical_inputs[miss_cpu].to(self.device, non_blocking=True)
        miss_fl = fl.index_select(0, missing_positions)
        if miss_fl.numel() > 0:
            flip_mask = miss_fl.bool()
            if flip_mask.any():
                canon = canon.clone()
                canon[flip_mask] = torch.flip(canon[flip_mask], dims=[-1])

        with torch.no_grad():
            acts = model.forward_until_layer(canon, layer_name).detach()

        self._ensure_bank(layer_name, acts)
        bank = self.banks[layer_name]
        valid = self.bank_valid[layer_name]
        miss_sid = sid.index_select(0, missing_positions)
        # Store in requested dtype.  Duplicated sample ids in a batch are harmless.
        bank[miss_fl, miss_sid] = acts.to(dtype=bank.dtype)
        valid[miss_fl, miss_sid] = True

        nmiss = int(missing_positions.numel())
        self.sample_misses += nmiss
        self.tcrc_populates += nmiss
        self.bank_misses += nmiss
        self.bank_populated[layer_name] = int(valid.sum().item())

    def _stack_canonical_batch(self, sample_ids, layer_name, canonical_inputs, flips, model):
        layer_name = normalize_freeze_name(layer_name)
        self._populate_missing(sample_ids, layer_name, canonical_inputs, flips, model)
        sid, fl = self._sid_flip_tensors(sample_ids, flips)
        bank = self.banks[layer_name]
        valid = self.bank_valid[layer_name]
        # A populated batch should now be valid.  Keep a safety fallback to avoid
        # silent invalid reads if sample ids are outside the expected range.
        valid_now = valid[fl, sid]
        if not bool(valid_now.all().item()):
            bad = torch.nonzero(~valid_now, as_tuple=False).flatten().detach().cpu().tolist()
            for pos in bad:
                # Fall back to the parent single-entry path for unusual cases.
                super()._populate_one_missing(pos, sample_ids, layer_name, canonical_inputs, flips, model)
            valid_now = valid[fl, sid]
            if not bool(valid_now.all().item()):
                raise RuntimeError("PlanTCRC-v2 dense bank has invalid entries after populate")

        cans = bank[fl, sid]
        # Count all returned canonical activations as memory hits, matching old
        # TCRC accounting after populate.
        B = int(sid.numel())
        self.memory_hits += B
        self.sample_hits += B
        self.bank_hits += B
        # Model is fp32 in this script; cast fp16/bf16 bank outputs back to fp32.
        if cans.dtype != torch.float32:
            cans = cans.float()
        return cans

    def _raw_transform_batch(self, sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model):
        canonical_batch = self._stack_canonical_batch(sample_ids, layer_name, canonical_inputs, flips, model)
        if canonical_batch.dim() != 4:
            raise RuntimeError(f"PlanTCRC-v2 expected 4D canonical activation batch, got {tuple(canonical_batch.shape)}")

        B, C, Hc, Wc = canonical_batch.shape
        h, w = self._target_hw(layer_name)
        stride = self._layer_stride(layer_name)

        if self._use_bilinear_subpixel():
            out = self._bilinear_window_from_canonical(canonical_batch, layer_name, crop_i, crop_j, flips)
            self.bank_bucket_batches += 1
            self.bank_bucket_groups += int(B)
            self.bank_bucket_samples += B
            self.tcrc_transforms += B
            self.plan_vectorized_batches += 1
            self.plan_vectorized_samples += B
            return out

        ci = crop_i.to(self.device, non_blocking=True).long()
        cj = crop_j.to(self.device, non_blocking=True).long()
        fl = flips.to(self.device, non_blocking=True).long()
        cj_eff = torch.where(fl == 1, torch.full_like(cj, 8) - cj, cj)
        y0 = self._vector_offsets(ci, stride).clamp(0, max(int(Hc) - int(h), 0))
        x0 = self._vector_offsets(cj_eff, stride).clamp(0, max(int(Wc) - int(w), 0))

        # Bucketed slicing avoids constructing a huge B*C*H*W gather-index tensor.
        if self.use_bucket_transform:
            max_x = max(int(Wc) - int(w), 0) + 1
            state = y0 * max_x + x0
            uniq = torch.unique(state, sorted=False)
            out = torch.empty((B, C, h, w), device=self.device, dtype=canonical_batch.dtype)
            for st in uniq:
                idx = torch.nonzero(state == st, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                first = int(idx[0].item())
                yy = int(y0[first].item())
                xx = int(x0[first].item())
                out.index_copy_(0, idx, canonical_batch.index_select(0, idx)[:, :, yy:yy + h, xx:xx + w].contiguous())
            groups = int(uniq.numel())
            self.bank_bucket_batches += 1
            self.bank_bucket_groups += groups
            self.bank_bucket_samples += B
            self.tcrc_transforms += B
            self.plan_vectorized_batches += 1
            self.plan_vectorized_samples += B
            return out

        # Optional fallback to parent vectorized gather.
        return super()._raw_transform_batch(sample_ids, layer_name, canonical_inputs, crop_i, crop_j, flips, model)

    def get_stats(self):
        stats = super().get_stats()
        stats.update({
            'plantcrc_v2_bank_hits': self.bank_hits,
            'plantcrc_v2_bank_misses': self.bank_misses,
            'plantcrc_v2_bank_allocated_gb': self.bank_allocated_bytes / 1e9,
            'plantcrc_v2_bank_bucket_batches': self.bank_bucket_batches,
            'plantcrc_v2_bank_bucket_groups': self.bank_bucket_groups,
            'plantcrc_v2_bank_bucket_samples': self.bank_bucket_samples,
            'plantcrc_v2_bank_populated': dict(self.bank_populated),
            'plantcrc_v2_bank_dtype': self.bank_dtype,
            'plantcrc_v2_bucket_transform': self.use_bucket_transform,
        })
        return stats


# =============================================================================
# HYBRID PLASTICITY + GRADIENT + DELTA-RESIDUAL FREEZER
# =============================================================================
class ReferenceModelManager:
    """CPU INT8 reference model for Egeria-style plasticity evaluation."""
    def __init__(self, device='cpu'):
        self.device = device
        self.reference_model = None

    def update(self, model):
        print("  [RefModel] Updating INT8 reference model on CPU")
        model_copy = copy.deepcopy(model).to('cpu')
        model_copy.eval()
        # Dynamic quantization is safe as a lightweight reference path.
        # Conv2d may remain fp32 in PyTorch dynamic quantization; Linear is quantized.
        self.reference_model = torch.quantization.quantize_dynamic(
            model_copy, {nn.Linear}, dtype=torch.qint8
        )

    def get_activation(self, x_cpu, layer_name):
        if self.reference_model is None:
            return None
        with torch.no_grad():
            return self.reference_model.forward_until_layer(x_cpu, normalize_freeze_name(layer_name))


class PlasticityEvaluator:
    """Egeria-style SP-loss plasticity tracker."""
    def __init__(self, window_size=10, tolerance_factor=0.20, min_tolerance=1e-4):
        self.window_size = window_size
        self.tolerance_factor = tolerance_factor
        self.min_tolerance = min_tolerance
        self.layer_names = list(FREEZE_UNITS)
        self.raw = {l: [] for l in self.layer_names}
        self.smoothed = {l: [] for l in self.layer_names}
        self.tolerance = {}

    def compute_sp_loss(self, act_train, act_ref):
        # Flatten activation maps to [batch, features].
        a = act_train.view(act_train.size(0), -1)
        b = act_ref.view(act_ref.size(0), -1)

        # Pairwise similarity matrices.
        sim_a = torch.mm(a, a.t())
        sim_b = torch.mm(b, b.t())

        # Normalized Frobenius distance.
        return (torch.norm(sim_a - sim_b, p='fro') /
                (torch.norm(sim_b, p='fro') + 1e-8)).item()

    def observe(self, layer_name, act_train, act_ref):
        layer_name = normalize_freeze_name(layer_name)
        p = self.compute_sp_loss(act_train, act_ref)
        self.raw[layer_name].append(p)

        recent = self.raw[layer_name][-self.window_size:]
        p_smooth = float(np.mean(recent))
        self.smoothed[layer_name].append(p_smooth)

        if layer_name not in self.tolerance and len(self.smoothed[layer_name]) >= 4:
            vals = self.smoothed[layer_name][:4]
            slopes = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
            self.tolerance[layer_name] = max(self.tolerance_factor * max(slopes), self.min_tolerance)

        return p, p_smooth

    def slope(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        vals = self.smoothed[layer_name][-self.window_size:]
        if len(vals) < self.window_size:
            return None
        x = np.arange(len(vals), dtype=np.float64)
        y = np.array(vals, dtype=np.float64)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(abs(m))

    def get_stats(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        return {
            'plasticity_raw': self.raw[layer_name][-1] if self.raw[layer_name] else None,
            'plasticity_smoothed': self.smoothed[layer_name][-1] if self.smoothed[layer_name] else None,
            'plasticity_slope': self.slope(layer_name),
            'plasticity_tolerance': self.tolerance.get(layer_name, None),
        }

    def reset(self):
        for l in self.layer_names:
            self.raw[l].clear()
            self.smoothed[l].clear()
        self.tolerance.clear()


class HybridLayerFreezer:
    """Hybrid freeze policy:
    - Egeria signal: representation plasticity slope.
    - ResiCache signal: normalized gradient norm.
    - Compute-benefit gate: avoid freezing layers whose cached reuse is not worth it.
    """
    def __init__(self, model, evaluator, eval_interval=50, bootstrap_epochs=5,
                 grad_window_size=20, grad_threshold=0.08,
                 hybrid_mode='weighted',
                 plasticity_weight=0.50, gradient_weight=0.30, compute_weight=0.20,
                 min_compute_benefit=0.05, cache_overhead_fraction=0.05,
                 min_freeze_layer='layer1', max_freeze_layer='layer3_3',
                 grad_track_interval=1, grad_track_scope='all'):
        self.model = model
        self.evaluator = evaluator
        self.eval_interval = eval_interval
        self.bootstrap_epochs = bootstrap_epochs
        self.grad_window_size = grad_window_size
        self.grad_threshold = grad_threshold
        self.hybrid_mode = hybrid_mode
        self.plasticity_weight = plasticity_weight
        self.gradient_weight = gradient_weight
        self.compute_weight = compute_weight
        self.min_compute_benefit = min_compute_benefit
        self.cache_overhead_fraction = cache_overhead_fraction
        self.min_freeze_layer = normalize_freeze_name(min_freeze_layer)
        self.max_freeze_layer = normalize_freeze_name(max_freeze_layer)
        self.grad_track_interval = max(1, int(grad_track_interval))
        if grad_track_scope not in ('all', 'candidate'):
            raise ValueError('grad_track_scope must be all or candidate')
        self.grad_track_scope = grad_track_scope
        self.grad_updates_performed = 0
        self.grad_updates_skipped = 0

        self.layer_names = list(FREEZE_UNITS)
        self.layer_compute_fraction = dict(BLOCKWISE_COMPUTE_FRACTION)
        self.gradient_history = {l: [] for l in self.layer_names}
        self.grad_reference = {}
        self.iteration = 0
        self.initial_lr = None
        self.last_lr = None
        self.cache_warmup = 0
        self.last_decision = {}

    def _layer_gradient_norm(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        if layer_name == 'conv1':
            params = list(self.model.conv1.parameters()) + list(self.model.bn1.parameters())
        else:
            params = list(self.model._module_for_freeze_unit(layer_name).parameters())
        total_sq = 0.0
        seen = False
        for p in params:
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += float(torch.sum(g * g).item())
                seen = True
        if not seen:
            return None
        return float(total_sq ** 0.5)

    def compute_gradient_norms(self):
        # Low-overhead option: track only the current frontmost active layer.
        # This avoids scanning every parameter tensor every batch.
        if self.grad_track_scope == 'candidate':
            candidate = self.get_frontmost_active_layer()
            if candidate is None:
                return {}
            norm = self._layer_gradient_norm(candidate)
            return {candidate: norm} if norm is not None and norm > 0 else {}

        grad_norms = {}
        for name in self.layer_names:
            norm = self._layer_gradient_norm(name)
            if norm is not None and norm > 0:
                grad_norms[name] = norm
        return grad_norms

    def update_gradient_history(self):
        # Track gradients only every N training iterations to reduce overhead.
        # Freezing decisions wait until enough sampled gradient history exists.
        if (self.iteration % self.grad_track_interval) != 0:
            self.grad_updates_skipped += 1
            return
        grad_norms = self.compute_gradient_norms()
        if not grad_norms:
            self.grad_updates_skipped += 1
            return
        self.grad_updates_performed += 1
        for layer, norm in grad_norms.items():
            self.gradient_history[layer].append(float(norm))
            if len(self.gradient_history[layer]) > self.grad_window_size:
                self.gradient_history[layer].pop(0)
            # Use early maximum as the normalization reference.
            self.grad_reference[layer] = max(self.grad_reference.get(layer, norm), norm)

    def get_gradient_tracking_stats(self):
        return {
            'grad_track_interval': self.grad_track_interval,
            'grad_track_scope': self.grad_track_scope,
            'grad_updates_performed': self.grad_updates_performed,
            'grad_updates_skipped': self.grad_updates_skipped,
            'gradient_history_lengths': {k: len(v) for k, v in self.gradient_history.items()},
        }

    def get_frontmost_active_layer(self):
        min_idx = self.layer_names.index(self.min_freeze_layer)
        max_idx = self.layer_names.index(self.max_freeze_layer)

        for i, layer in enumerate(self.layer_names):
            if i < min_idx or i > max_idx:
                continue
            if layer not in self.model.frozen_layers:
                return layer

        return None

    def get_frontmost_frozen_layer(self):
        # Deepest frozen layer. Cache reuse starts from this layer output.
        for l in reversed(self.layer_names):
            if l in self.model.frozen_layers:
                return l
        return None

    def in_cache_warmup(self):
        return self.cache_warmup > 0

    def _prefix_constraint_ok(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        min_idx = self.layer_names.index(self.min_freeze_layer)
        idx = self.layer_names.index(layer_name)

        for i in range(min_idx, idx):
            if self.layer_names[i] not in self.model.frozen_layers:
                return False

        return True

    def _min_layer_ok(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        idx = self.layer_names.index(layer_name)
        return (idx >= self.layer_names.index(self.min_freeze_layer) and
                idx <= self.layer_names.index(self.max_freeze_layer))

    def _gradient_score(self, layer_name):
        h = self.gradient_history[layer_name]
        if len(h) < max(5, self.grad_window_size // 2):
            return None
        current = float(np.mean(h[len(h)//2:]))
        ref = max(self.grad_reference.get(layer_name, max(h)), 1e-8)
        return current / ref

    def _plasticity_score(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        slope = self.evaluator.slope(layer_name)
        tol = self.evaluator.tolerance.get(layer_name, None)
        if slope is None or tol is None:
            return None
        return slope / max(tol, 1e-8)

    def _compute_benefit(self, layer_name):
        layer_name = normalize_freeze_name(layer_name)
        saved = self.layer_compute_fraction[layer_name]
        benefit = saved - self.cache_overhead_fraction
        return benefit

    def maybe_freeze(self, layer_name):
        if layer_name is not None:
            layer_name = normalize_freeze_name(layer_name)
        if layer_name is None:
            return None, False
        if layer_name in self.model.frozen_layers:
            return None, False
        if not self._prefix_constraint_ok(layer_name):
            return None, False
        if not self._min_layer_ok(layer_name):
            self.last_decision[layer_name] = {'decision': 'skip_layer_range'}
            return None, False

        g_score = self._gradient_score(layer_name)
        p_score = self._plasticity_score(layer_name)
        benefit = self._compute_benefit(layer_name)

        decision = {
            'layer': layer_name,
            'gradient_score': g_score,
            'gradient_threshold': self.grad_threshold,
            'plasticity_score': p_score,
            'compute_benefit': benefit,
            'min_compute_benefit': self.min_compute_benefit,
            'mode': self.hybrid_mode,
        }

        # In gradient_only mode, we intentionally do not require plasticity/reference history.
        # This supports reference-free ResiCache ablations.
        if g_score is None or (self.hybrid_mode != 'gradient_only' and p_score is None):
            decision['decision'] = 'wait_insufficient_history'
            self.last_decision[layer_name] = decision
            return None, False

        compute_score = self.min_compute_benefit / max(benefit, 1e-8)

        if self.hybrid_mode == 'strict':
            should = (p_score < 1.0 and
                      g_score < self.grad_threshold and
                      benefit > self.min_compute_benefit)
            hybrid_score = max(p_score, g_score / max(self.grad_threshold, 1e-8), compute_score)
        elif self.hybrid_mode == 'plasticity_only':
            should = (p_score < 1.0 and benefit > self.min_compute_benefit)
            hybrid_score = p_score
        elif self.hybrid_mode == 'gradient_only':
            should = (g_score < self.grad_threshold and benefit > self.min_compute_benefit)
            hybrid_score = g_score / max(self.grad_threshold, 1e-8)
        else:
            # Weighted confidence score. Freeze if below 1.
            hybrid_score = (
                self.plasticity_weight * p_score
                + self.gradient_weight * (g_score / max(self.grad_threshold, 1e-8))
                + self.compute_weight * compute_score
            )
            should = hybrid_score < 1.0 and benefit > self.min_compute_benefit

        decision['hybrid_score'] = float(hybrid_score)
        decision['decision'] = 'freeze' if should else 'keep_training'
        self.last_decision[layer_name] = decision

        if should:
            self.model.freeze_layer(layer_name)
            self.cache_warmup = 10
            pscore_str = 'NA' if p_score is None else f'{p_score:.3f}'
            print(f"  [HybridFreezer] FROZEN {layer_name}: "
                  f"Pscore={pscore_str}, Gscore={g_score:.3f}, "
                  f"Benefit={benefit:.3f}, Hybrid={hybrid_score:.3f}")
            return layer_name, True

        return None, False

    def force_freeze_range(self, max_layer=None, min_layer=None):
        """Force-freeze a contiguous prefix of blockwise units.

        Used for accuracy-vs-freeze-depth curves.  The optimizer does not need
        to be rebuilt: PyTorch SGD skips parameters whose grad is None after
        requires_grad=False.
        """
        lo = freeze_unit_index(min_layer or self.min_freeze_layer)
        hi = freeze_unit_index(max_layer or self.max_freeze_layer)
        if hi < lo:
            raise ValueError('force_freeze_range requires max_layer >= min_layer')
        newly = []
        for unit in self.layer_names[lo:hi + 1]:
            if unit not in self.model.frozen_layers:
                self.model.freeze_layer(unit)
                newly.append(unit)
        if newly:
            self.cache_warmup = 10
            print(f"  [Force] FROZEN blockwise prefix through {self.layer_names[hi]}: {newly}")
        return newly

    def step_end(self):
        self.iteration += 1
        if self.cache_warmup > 0:
            self.cache_warmup -= 1

    def check_lr_change(self, current_lr):
        if self.initial_lr is None:
            self.initial_lr = current_lr
            self.last_lr = current_lr
            return False
        if current_lr < self.last_lr * 0.1:
            self.last_lr = current_lr
            return True
        return False

    def reset_after_lr(self):
        self.model.unfreeze_all()
        self.evaluator.reset()
        self.gradient_history = {l: [] for l in self.layer_names}
        self.grad_reference = {}
        self.cache_warmup = 0


# =============================================================================
# DATASET
# =============================================================================
class CIFAR10WithAugKey(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset returning both augmented and canonical padded inputs.

    Returns:
        cache_key: key used by original delta cache policies
        img: augmented 32x32 normalized tensor used for actual training
        target: label
        canonical: padded 40x40 normalized tensor before crop/flip, used by TCRC
        crop_i, crop_j, flip: augmentation parameters
        index: original sample id

    This preserves compatibility with previous sample-cache experiments while
    enabling transform-consistent residual caching.
    """
    def __init__(self, root, train=True, download=False, aug_mode='random',
                 cache_key_mode='sample', fixed_aug_seed=12345):
        super().__init__(root=root, train=train, download=download, transform=None)
        self.aug_mode = aug_mode
        self.cache_key_mode = cache_key_mode
        self.fixed_aug_seed = int(fixed_aug_seed)
        self.epoch = 0
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _sample_aug_params(self, index):
        if self.aug_mode == 'none' or not self.train:
            return 0, 0, 0, 0
        if self.aug_mode == 'fixed_sample':
            rng = random.Random(self.fixed_aug_seed + int(index) * 1000003)
            i = rng.randint(0, 8)
            j = rng.randint(0, 8)
            flip = 1 if rng.random() < 0.5 else 0
        elif self.aug_mode == 'fixed_epoch':
            rng = random.Random(self.fixed_aug_seed + int(self.epoch) * 10000019 + int(index) * 1000003)
            i = rng.randint(0, 8)
            j = rng.randint(0, 8)
            flip = 1 if rng.random() < 0.5 else 0
        elif self.aug_mode == 'random':
            i = random.randint(0, 8)
            j = random.randint(0, 8)
            flip = 1 if random.random() < 0.5 else 0
        else:
            raise ValueError(f'Unknown aug_mode: {self.aug_mode}')
        aug_code = ((i * 9) + j) * 2 + flip
        return i, j, flip, aug_code

    def _make_cache_key(self, index, aug_code):
        index = int(index)
        if self.cache_key_mode == 'sample':
            return index
        if self.cache_key_mode == 'sample_aug':
            return index * 1000 + int(aug_code)
        if self.cache_key_mode == 'sample_epoch':
            return index * 1000 + (int(self.epoch) % 1000)
        raise ValueError(f'Unknown cache_key_mode: {self.cache_key_mode}')

    def _to_norm_tensor(self, img):
        img = TF.to_tensor(img)
        return TF.normalize(img, self.mean, self.std)

    def __getitem__(self, index):
        original = Image.fromarray(self.data[index])
        target = self.targets[index]
        i, j, flip, aug_code = self._sample_aug_params(index)

        # Canonical image used by TCRC: padded but not cropped and not flipped.
        # Horizontal flip is handled in the TCRC cache by using a separate flipped
        # canonical feature entry.
        canonical_img = TF.pad(original, 4, padding_mode='constant') if self.train else original

        img = original
        if self.train and self.aug_mode != 'none':
            img = TF.pad(img, 4, padding_mode='constant')
            img = TF.crop(img, i, j, 32, 32)
            if flip:
                img = TF.hflip(img)

        img = self._to_norm_tensor(img)
        canonical = self._to_norm_tensor(canonical_img)
        cache_key = self._make_cache_key(index, aug_code)
        return cache_key, img, target, canonical, int(i), int(j), int(flip), int(index)


# =============================================================================
# TRAINING
# =============================================================================
class RunTimers:
    """Lightweight cumulative timers for profiling ResiCache overhead.

    Timings are intentionally reported as wall-clock seconds. Optional CUDA
    synchronization improves attribution accuracy but adds measurement overhead.
    """
    def __init__(self, device=None, cuda_sync=False):
        self.device = device
        self.cuda_sync = cuda_sync
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def _sync(self):
        if self.cuda_sync and self.device is not None and getattr(self.device, 'type', None) == 'cuda':
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        return time.time()

    def stop(self, name, start_time):
        self._sync()
        elapsed = time.time() - start_time
        self.total[name] += elapsed
        self.count[name] += 1
        return elapsed

    def snapshot(self):
        out = {}
        for k in sorted(self.total):
            out[k] = {
                'total_sec': self.total[k],
                'count': self.count[k],
                'avg_sec': self.total[k] / max(self.count[k], 1),
            }
        return out


def train_epoch(model, trainloader, optimizer, criterion, device, cache, freezer,
                evaluator, ref_manager, epoch,
                residual_refresh_interval=50, residual_probe_prob=0.0,
                timing=None, disable_cache_reuse=False, cache_impl='delta',
                tcrc_gate_threshold=0.0, tcrc_gate_every=0, tcrc_gate_samples=16):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Inform augmentation-aware datasets which epoch is being executed.
    if hasattr(trainloader.dataset, 'set_epoch'):
        trainloader.dataset.set_epoch(epoch)

    # Keep frozen BatchNorm/blocks in eval mode.
    model.set_frozen_eval()

    for batch_idx, batch in enumerate(trainloader):
        # New TCRC dataset returns augmented input plus canonical padded input and
        # augmentation parameters.  Older three-field batches are still accepted.
        if len(batch) == 8:
            cache_keys, inputs, targets, canonical_inputs, crop_i, crop_j, flips, orig_indices = batch
            canonical_inputs = canonical_inputs.to(device, non_blocking=True)
            crop_i = crop_i.to(device)
            crop_j = crop_j.to(device)
            flips = flips.to(device)
            orig_sample_ids = orig_indices.tolist()
        else:
            cache_keys, inputs, targets = batch
            canonical_inputs = None
            crop_i = crop_j = flips = None
            orig_sample_ids = cache_keys.tolist()
        inputs, targets = inputs.to(device), targets.to(device)
        # Original delta cache uses the chosen cache key. TCRC uses the stable
        # original sample id because it derives the crop/flip-specific activation
        # from a canonical cached feature map.
        sample_ids = cache_keys.tolist()

        t0 = timing.start() if timing else None
        optimizer.zero_grad()
        if timing:
            timing.stop('optimizer_zero_grad', t0)

        active_frozen_layer = freezer.get_frontmost_frozen_layer()
        in_warmup = freezer.in_cache_warmup()
        use_cache_layer = (active_frozen_layer and normalize_freeze_name(active_frozen_layer) != FREEZE_UNITS[-1] and not disable_cache_reuse)

        force_refresh = False
        if use_cache_layer and not in_warmup:
            if residual_refresh_interval > 0 and batch_idx % residual_refresh_interval == 0:
                force_refresh = True
            elif residual_probe_prob > 0.0 and random.random() < residual_probe_prob:
                force_refresh = True

        t_forward = timing.start() if timing else None

        if cache_impl in ('tcrc', 'plantcrc', 'plantcrc_v2') and use_cache_layer and canonical_inputs is None:
            raise RuntimeError('TCRC cache_impl requires dataset batch with canonical inputs and augmentation params.')

        if cache_impl in ('tcrc', 'plantcrc', 'plantcrc_v2') and use_cache_layer and in_warmup:
            # Populate canonical cache but train using the true augmented activation.
            # This avoids using a transform approximation immediately after freeze.
            t = timing.start() if timing else None
            activation = model.forward_until_layer(inputs, active_frozen_layer)
            if timing:
                timing.stop('forward_until_frozen_layer', t)
            t = timing.start() if timing else None
            cache.put_batch(orig_sample_ids, active_frozen_layer, canonical_inputs,
                            crop_i=crop_i, crop_j=crop_j, flips=flips, model=model)
            if timing:
                timing.stop('tcrc_canonical_populate', t)
            t = timing.start() if timing else None
            outputs = model.forward_from_layer(activation, active_frozen_layer)
            if timing:
                timing.stop('forward_from_cached_layer', t)

        elif cache_impl in ('tcrc', 'plantcrc', 'plantcrc_v2') and use_cache_layer and force_refresh:
            # Probe true current activation, update residual correction, but use the
            # true activation for this batch so probing itself cannot hurt accuracy.
            t = timing.start() if timing else None
            activation = model.forward_until_layer(inputs, active_frozen_layer)
            if timing:
                timing.stop('forward_until_frozen_layer', t)
            t = timing.start() if timing else None
            cache.refresh_batch(orig_sample_ids, active_frozen_layer, canonical_inputs,
                                crop_i, crop_j, flips, model, activation)
            if timing:
                timing.stop('tcrc_probe_refresh', t)
            t = timing.start() if timing else None
            outputs = model.forward_from_layer(activation, active_frozen_layer)
            if timing:
                timing.stop('forward_from_cached_layer', t)

        elif cache_impl in ('tcrc', 'plantcrc', 'plantcrc_v2') and use_cache_layer:
            gate_use_true = False
            gate_rel_error = None
            # TTA-safe probe gate: periodically compare a small reconstructed
            # slice with true frozen-prefix activations.  If the transform error
            # is high, use true activations for the whole batch.  This trades
            # some per-epoch speed for a less distorted learning trajectory.
            if (tcrc_gate_threshold is not None and tcrc_gate_threshold > 0.0 and
                    tcrc_gate_every is not None and tcrc_gate_every > 0 and
                    batch_idx % int(tcrc_gate_every) == 0):
                probe_n = min(int(tcrc_gate_samples), inputs.size(0))
                if probe_n > 0:
                    t = timing.start() if timing else None
                    with torch.no_grad():
                        true_probe = model.forward_until_layer(inputs[:probe_n], active_frozen_layer).detach()
                    if timing:
                        timing.stop('tcrc_gate_true_probe', t)
                    t = timing.start() if timing else None
                    with torch.no_grad():
                        cached_probe = cache.get_batch(orig_sample_ids[:probe_n], active_frozen_layer,
                                                       canonical_inputs[:probe_n],
                                                       crop_i[:probe_n], crop_j[:probe_n], flips[:probe_n], model)
                    if timing:
                        timing.stop('tcrc_gate_cached_probe', t)
                    if cached_probe is not None and cached_probe.shape == true_probe.shape:
                        rel = (true_probe - cached_probe).flatten(1).norm(dim=1) / true_probe.flatten(1).norm(dim=1).clamp_min(1e-8)
                        gate_rel_error = float(rel.mean().item())
                        if gate_rel_error > float(tcrc_gate_threshold):
                            gate_use_true = True

            if gate_use_true:
                if hasattr(cache, 'tcrc_gate_fallbacks'):
                    cache.tcrc_gate_fallbacks += 1
                    cache.tcrc_gate_error_sum += float(gate_rel_error or 0.0)
                t = timing.start() if timing else None
                activation = model.forward_until_layer(inputs, active_frozen_layer)
                if timing:
                    timing.stop('forward_until_frozen_layer_gate_fallback', t)
                t = timing.start() if timing else None
                outputs = model.forward_from_layer(activation, active_frozen_layer)
                if timing:
                    timing.stop('forward_from_cached_layer', t)
            else:
                if hasattr(cache, 'tcrc_gate_accepts') and gate_rel_error is not None:
                    cache.tcrc_gate_accepts += 1
                    cache.tcrc_gate_error_sum += float(gate_rel_error)
                t = timing.start() if timing else None
                corrected_act = cache.get_batch(orig_sample_ids, active_frozen_layer, canonical_inputs,
                                                crop_i, crop_j, flips, model)
                if timing:
                    timing.stop('tcrc_get_transform_batch', t)
                if corrected_act is not None:
                    t = timing.start() if timing else None
                    outputs = model.forward_from_layer(corrected_act, active_frozen_layer)
                    if timing:
                        timing.stop('forward_from_cached_layer', t)
                else:
                    # Safety fallback if the TCRC probe disables reuse.
                    t = timing.start() if timing else None
                    outputs = model(inputs)
                    if timing:
                        timing.stop('full_forward_no_cache', t)

        elif use_cache_layer and in_warmup:
            t = timing.start() if timing else None
            activation = model.forward_until_layer(inputs, active_frozen_layer)
            if timing:
                timing.stop('forward_until_frozen_layer', t)
            t = timing.start() if timing else None
            cache.put_batch(sample_ids, active_frozen_layer, activation)
            if timing:
                timing.stop('cache_put_batch', t)
            t = timing.start() if timing else None
            outputs = model.forward_from_layer(activation, active_frozen_layer)
            if timing:
                timing.stop('forward_from_cached_layer', t)
        elif use_cache_layer and force_refresh:
            t = timing.start() if timing else None
            activation = model.forward_until_layer(inputs, active_frozen_layer)
            if timing:
                timing.stop('forward_until_frozen_layer', t)
            t = timing.start() if timing else None
            cache.refresh_batch(sample_ids, active_frozen_layer, activation)
            if timing:
                timing.stop('cache_refresh_batch', t)
            t = timing.start() if timing else None
            outputs = model.forward_from_layer(activation, active_frozen_layer)
            if timing:
                timing.stop('forward_from_cached_layer', t)
        elif use_cache_layer:
            t = timing.start() if timing else None
            corrected_act = cache.get_batch(sample_ids, active_frozen_layer)
            if timing:
                timing.stop('cache_get_batch', t)
            if corrected_act is not None:
                t = timing.start() if timing else None
                outputs = model.forward_from_layer(corrected_act, active_frozen_layer)
                if timing:
                    timing.stop('forward_from_cached_layer', t)
            else:
                t = timing.start() if timing else None
                activation = model.forward_until_layer(inputs, active_frozen_layer)
                if timing:
                    timing.stop('forward_until_frozen_layer', t)
                t = timing.start() if timing else None
                cache.put_batch(sample_ids, active_frozen_layer, activation)
                if timing:
                    timing.stop('cache_put_batch', t)
                t = timing.start() if timing else None
                outputs = model.forward_from_layer(activation, active_frozen_layer)
                if timing:
                    timing.stop('forward_from_cached_layer', t)
        else:
            t = timing.start() if timing else None
            outputs = model(inputs)
            if timing:
                timing.stop('full_forward_no_cache', t)
        if timing:
            timing.stop('forward_total', t_forward)

        t = timing.start() if timing else None
        loss = criterion(outputs, targets)
        if timing:
            timing.stop('loss_compute', t)

        t = timing.start() if timing else None
        loss.backward()
        if timing:
            timing.stop('backward', t)

        t = timing.start() if timing else None
        freezer.update_gradient_history()
        if timing:
            timing.stop('gradient_history_update', t)

        # Freeze evaluation is periodic and uses the frontmost active layer.
        # If ref_manager is None, this becomes true reference-free gradient-only ResiCache.
        candidate = freezer.get_frontmost_active_layer()
        if (epoch >= freezer.bootstrap_epochs and candidate is not None and
                freezer.iteration % freezer.eval_interval == 0):
            if freezer.hybrid_mode == 'gradient_only' and ref_manager is None:
                t = timing.start() if timing else None
                newly_frozen, clear_cache = freezer.maybe_freeze(candidate)
                if timing:
                    timing.stop('freeze_decision_no_reference', t)
                if clear_cache:
                    t = timing.start() if timing else None
                    cache.clear()
                    if timing:
                        timing.stop('cache_clear', t)
                    print(f"  [Cache] Cleared after freezing {newly_frozen}")
            elif ref_manager is not None and ref_manager.reference_model is not None:
                with torch.no_grad():
                    t = timing.start() if timing else None
                    act_train = model.forward_until_layer(inputs, candidate)
                    if timing:
                        timing.stop('plasticity_train_activation', t)
                    t = timing.start() if timing else None
                    act_ref = ref_manager.get_activation(inputs.detach().cpu(), candidate)
                    if timing:
                        timing.stop('reference_forward_activation', t)
                    if act_ref is not None:
                        t = timing.start() if timing else None
                        act_ref = act_ref.to(device)
                        evaluator.observe(candidate, act_train, act_ref)
                        if timing:
                            timing.stop('plasticity_sp_loss_and_slope', t)
                        t = timing.start() if timing else None
                        newly_frozen, clear_cache = freezer.maybe_freeze(candidate)
                        if timing:
                            timing.stop('freeze_decision_with_reference', t)
                        if clear_cache:
                            t = timing.start() if timing else None
                            cache.clear()
                            if timing:
                                timing.stop('cache_clear', t)
                            print(f"  [Cache] Cleared after freezing {newly_frozen}")

        t = timing.start() if timing else None
        optimizer.step()
        if timing:
            timing.stop('optimizer_step', t)
        freezer.step_end()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            stats = cache.get_stats()
            cand = freezer.get_frontmost_active_layer()
            pstats = evaluator.get_stats(cand) if cand is not None else {}
            print(f'    [Train] Epoch: {epoch} [{batch_idx}/{len(trainloader)}] '
                  f'Loss: {running_loss / (batch_idx + 1):.3f} | '
                  f'Acc: {100. * correct / total:.2f}% | '
                  f'Frozen: {ordered_frozen_layers(model.frozen_layers)} | '
                  f'Cache: {100.0 * stats["hit_rate"]:.1f}% | '
                  f'DeltaUpd: {stats["delta_updates"]} | '
                  f'Candidate: {cand} | P_slope: {pstats.get("plasticity_slope")}')

    return running_loss / len(trainloader), 100. * correct / total

def test(model, testloader, criterion, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'    [Test] Loss: {test_loss / len(testloader):.3f} | Acc: {acc:.2f}%')
    return test_loss / len(testloader), acc


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser(description='ResiCache Evaluation Suite - ResNet56 CIFAR10')
    parser.add_argument('--output_json', type=str, default='results/resicache_eval.jsonl')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache_dir', type=str, default='./resicache_eval_samples')
    parser.add_argument('--gpu_cache_size', type=int, default=50000)
    parser.add_argument('--target_acc', type=float, default=92.1)
    parser.add_argument('--aug_mode', '--augmentation_mode', dest='aug_mode', type=str, default='random',
                        choices=['random', 'fixed_sample', 'fixed_epoch', 'none',
                                 'deterministic_per_sample', 'deterministic_per_epoch'],
                        help='Training augmentation mode for cache correctness ablations. Aliases: deterministic_per_sample=fixed_sample, deterministic_per_epoch=fixed_epoch.')
    parser.add_argument('--cache_key_mode', type=str, default='sample',
                        choices=['sample', 'sample_aug', 'sample_epoch'],
                        help='Cache-key policy: sample-only, augmentation-aware, or epoch-aware.')
    parser.add_argument('--fixed_aug_seed', type=int, default=12345,
                        help='Seed for deterministic augmentation modes.')

    # Freeze rule.
    parser.add_argument('--hybrid_mode', type=str, default='weighted',
                        choices=['weighted', 'strict', 'plasticity_only', 'gradient_only'])
    parser.add_argument('--eval_interval', type=int, default=300)
    parser.add_argument('--bootstrap_epochs', type=int, default=30)
    parser.add_argument('--grad_window_size', type=int, default=20)
    parser.add_argument('--grad_threshold', type=float, default=0.35)
    parser.add_argument('--grad_track_interval', type=int, default=1,
                        help='Compute layer gradient norms every N training iterations. Higher values reduce overhead.')
    parser.add_argument('--grad_track_scope', type=str, default='all', choices=['all', 'candidate'],
                        help='Track gradients for all layers or only the current frontmost active candidate layer.')
    parser.add_argument('--plasticity_window_size', type=int, default=10)
    parser.add_argument('--plasticity_tolerance_factor', type=float, default=0.20)
    parser.add_argument('--plasticity_weight', type=float, default=0.50)
    parser.add_argument('--gradient_weight', type=float, default=0.30)
    parser.add_argument('--compute_weight', type=float, default=0.20)
    parser.add_argument('--min_compute_benefit', type=float, default=0.10)
    parser.add_argument('--cache_overhead_fraction', type=float, default=0.05)
    parser.add_argument('--min_freeze_layer', type=str, default='layer1',
                        choices=FREEZE_CHOICES,
                        help='First blockwise freeze unit. Coarse aliases like layer2 are accepted.')
    parser.add_argument('--max_freeze_layer', type=str, default='layer3_3',
                        choices=FREEZE_CHOICES,
                        help='Deepest blockwise freeze unit. Coarse aliases like layer2 map to their last block.')
    parser.add_argument('--ref_update_epochs', type=int, default=10)
    parser.add_argument('--disable_reference_model', action='store_true',
                        help='Disable CPU reference model and SP-loss plasticity path. Requires --hybrid_mode gradient_only.')
    parser.add_argument('--disable_lr_unfreeze', action='store_true',
                        help='Do not unfreeze layers when LR drops. Useful for speed/accuracy ablation.')
    parser.add_argument('--force_freeze_epoch', type=int, default=-1,
                        help='If >=0, force-freeze the full blockwise prefix from min_freeze_layer to max_freeze_layer at this epoch. Useful for accuracy-vs-freeze-depth sweeps.')
    parser.add_argument('--force_freeze_schedule', type=str, default='',
                        help='Optional staged freeze schedule, e.g. "45:layer1.3,85:layer2.0". Each event force-freezes the prefix from min_freeze_layer to the given layer and clears cache.')

    # Activation cache implementation.
    parser.add_argument('--cache_impl', type=str, default='delta', choices=['delta', 'tcrc', 'plantcrc', 'plantcrc_v2'],
                        help='delta = previous sample activation cache; tcrc = original TCRC; plantcrc = vectorized TCRC; plantcrc_v2 = dense-bank bucketed PlanTCRC.')
    parser.add_argument('--tcrc_crop_rounding', type=str, default='floor', choices=['floor', 'round', 'ceil'],
                        help='How to map image crop offsets to feature-space crop offsets for TCRC.')
    parser.add_argument('--tcrc_subpixel', type=str, default='none', choices=['none', 'bilinear'],
                        help='v6: use bilinear/sub-pixel crop reconstruction instead of integer feature offsets.')
    parser.add_argument('--tcrc_probe_error_threshold', type=float, default=0.0,
                        help='If >0, disable TCRC reuse when mean probe error exceeds this threshold.')
    parser.add_argument('--cache_start_epoch', type=int, default=-1,
                        help='If >=0, delay activation-cache reuse until this epoch. Freezing still happens normally.')
    parser.add_argument('--cache_start_acc', type=float, default=-1.0,
                        help='If >=0, delay activation-cache reuse until best test accuracy so far reaches this value.')
    parser.add_argument('--cache_start_warmup_batches', type=int, default=10,
                        help='Number of batches to populate cache with true activations when cache reuse first starts.')
    parser.add_argument('--tcrc_gate_threshold', type=float, default=0.0,
                        help='If >0, enable probe-error-gated fallback with this mean relative-error threshold.')
    parser.add_argument('--tcrc_gate_every', type=int, default=0,
                        help='Run the TCRC gate probe every N batches. 0 disables gate probing.')
    parser.add_argument('--tcrc_gate_samples', type=int, default=16,
                        help='Number of samples used for each gate probe.')
    parser.add_argument('--tcrc_state_delta', action='store_true',
                        help='Enable crop/flip-state-conditioned residual correction for TCRC/PlanTCRC.')
    parser.add_argument('--plantcrc_bank_dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16', 'same'],
                        help='Dense-bank storage dtype for plantcrc_v2. fp32 is safest; fp16 uses less GPU memory.')
    parser.add_argument('--plantcrc_num_samples', type=int, default=50000,
                        help='Number of training samples for dense-bank PlanTCRC-v2 indexing.')
    parser.add_argument('--plantcrc_no_bucket_transform', action='store_true',
                        help='Disable bucketed slicing in plantcrc_v2 and fall back to vectorized gather.')

    # Delta-residual cache.
    parser.add_argument('--residual_momentum', type=float, default=0.90)
    parser.add_argument('--residual_clip', type=float, default=5.0)
    parser.add_argument('--residual_refresh_interval', type=int, default=40)
    parser.add_argument('--residual_probe_prob', type=float, default=0.01)
    parser.add_argument('--no_disk_cache', action='store_true')
    parser.add_argument('--disable_cache_reuse', action='store_true',
                        help='Freeze layers but do not use activation cache. This is a plain-freezing ablation.')

    # Profiling.
    parser.add_argument('--profile_timing', action='store_true',
                        help='Record detailed timing breakdown in JSON logs.')
    parser.add_argument('--profile_cuda_sync', action='store_true',
                        help='Synchronize CUDA before/after timers. More accurate but slower.')

    args = parser.parse_args()

    # Normalize user-friendly augmentation aliases used in command files.
    if args.aug_mode == 'deterministic_per_sample':
        args.aug_mode = 'fixed_sample'
    elif args.aug_mode == 'deterministic_per_epoch':
        args.aug_mode = 'fixed_epoch'

    if args.disable_reference_model and args.hybrid_mode != 'gradient_only':
        raise ValueError('--disable_reference_model currently requires --hybrid_mode gradient_only.')
    # Normalize and validate blockwise freeze range.
    args.min_freeze_layer = normalize_freeze_name(args.min_freeze_layer)
    args.max_freeze_layer = normalize_freeze_name(args.max_freeze_layer)
    if freeze_unit_index(args.max_freeze_layer) < freeze_unit_index(args.min_freeze_layer):
        raise ValueError('--max_freeze_layer must be at or after --min_freeze_layer')

    # Parse optional staged force-freeze schedule.
    # Format: "45:layer1.3,85:layer2.0".
    force_freeze_events = []
    if args.force_freeze_schedule.strip():
        for item in args.force_freeze_schedule.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                raise ValueError(f'Bad --force_freeze_schedule item: {item}. Expected epoch:layer')
            ep_s, layer_s = item.split(':', 1)
            ep = int(ep_s)
            layer = normalize_freeze_name(layer_s.strip())
            if layer not in FREEZE_UNITS:
                raise ValueError(f'Bad schedule layer {layer_s}; normalized to {layer}, not in FREEZE_UNITS')
            if freeze_unit_index(layer) < freeze_unit_index(args.min_freeze_layer):
                raise ValueError(f'Schedule layer {layer} is before min_freeze_layer {args.min_freeze_layer}')
            force_freeze_events.append((ep, layer))
        force_freeze_events = sorted(force_freeze_events, key=lambda x: x[0])
        args.max_freeze_layer = max(
            [args.max_freeze_layer] + [layer for _, layer in force_freeze_events],
            key=freeze_unit_index
        )

    os.makedirs(os.path.dirname(args.output_json) if args.output_json else '.', exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Main] Using device: {device}")
    if device.type == 'cuda':
        print(f"[Main] GPU: {torch.cuda.get_device_name(0)}")

    print("[Main] Loading CIFAR-10 dataset from:", args.data_root)
    print(f"[Main] Augmentation ablation: aug_mode={args.aug_mode}, cache_key_mode={args.cache_key_mode}")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10WithAugKey(
        root=args.data_root, train=True, download=False,
        aug_mode=args.aug_mode, cache_key_mode=args.cache_key_mode,
        fixed_aug_seed=args.fixed_aug_seed
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g,
        pin_memory=True if device.type == 'cuda' else False
    )

    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False
    )

    print("[Main] Initializing ResNet-56 model...")
    model = ResNet56(num_classes=10).to(device)

    if args.cache_impl == 'plantcrc_v2':
        cache = BankPlanTransformConsistentResidualCache(
            gpu_cache_size=args.gpu_cache_size,
            device=device,
            residual_momentum=args.residual_momentum,
            residual_clip=None if args.residual_clip <= 0 else args.residual_clip,
            crop_rounding=args.tcrc_crop_rounding,
            probe_error_threshold=args.tcrc_probe_error_threshold,
            subpixel_mode=args.tcrc_subpixel,
            num_samples=args.plantcrc_num_samples,
            bank_dtype=args.plantcrc_bank_dtype,
            use_bucket_transform=not args.plantcrc_no_bucket_transform,
        )
    elif args.cache_impl == 'plantcrc':
        cache = PlanTransformConsistentResidualCache(
            gpu_cache_size=args.gpu_cache_size,
            device=device,
            residual_momentum=args.residual_momentum,
            residual_clip=None if args.residual_clip <= 0 else args.residual_clip,
            crop_rounding=args.tcrc_crop_rounding,
            probe_error_threshold=args.tcrc_probe_error_threshold,
            subpixel_mode=args.tcrc_subpixel,
        )
    elif args.cache_impl == 'tcrc':
        cache = TransformConsistentResidualCache(
            gpu_cache_size=args.gpu_cache_size,
            device=device,
            residual_momentum=args.residual_momentum,
            residual_clip=None if args.residual_clip <= 0 else args.residual_clip,
            crop_rounding=args.tcrc_crop_rounding,
            probe_error_threshold=args.tcrc_probe_error_threshold,
            subpixel_mode=args.tcrc_subpixel,
        )
    else:
        cache = DeltaResidualSampleCache(
            cache_dir=args.cache_dir,
            gpu_cache_size=args.gpu_cache_size,
            device=device,
            residual_momentum=args.residual_momentum,
            residual_clip=None if args.residual_clip <= 0 else args.residual_clip,
            disk_cache=not args.no_disk_cache
        )

    if hasattr(cache, 'use_state_delta'):
        cache.use_state_delta = bool(args.tcrc_state_delta)
        if cache.use_state_delta:
            print('[Main] TCRC state-conditioned residual correction ENABLED')

    evaluator = PlasticityEvaluator(
        window_size=args.plasticity_window_size,
        tolerance_factor=args.plasticity_tolerance_factor
    )
    freezer = HybridLayerFreezer(
        model=model,
        evaluator=evaluator,
        eval_interval=args.eval_interval,
        bootstrap_epochs=args.bootstrap_epochs,
        grad_window_size=args.grad_window_size,
        grad_threshold=args.grad_threshold,
        hybrid_mode=args.hybrid_mode,
        plasticity_weight=args.plasticity_weight,
        gradient_weight=args.gradient_weight,
        compute_weight=args.compute_weight,
        min_compute_benefit=args.min_compute_benefit,
        cache_overhead_fraction=args.cache_overhead_fraction,
        min_freeze_layer=args.min_freeze_layer,
        max_freeze_layer=args.max_freeze_layer,
        grad_track_interval=args.grad_track_interval,
        grad_track_scope=args.grad_track_scope,
    )

    timing = RunTimers(device=device, cuda_sync=args.profile_cuda_sync) if args.profile_timing else None

    if args.disable_reference_model:
        print('[Main] Reference model DISABLED: running true reference-free gradient-only ResiCache.')
        ref_manager = None
    else:
        ref_manager = ReferenceModelManager()
        t = timing.start() if timing else None
        ref_manager.update(model)
        if timing:
            timing.stop('reference_model_update', t)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    print("[Main] Starting ResiCache training...")
    print(f"[Main] Mode={args.hybrid_mode}, Ref={not args.disable_reference_model}, Eval={args.eval_interval}, "
          f"Boot={args.bootstrap_epochs}, GradWin={args.grad_window_size}, GradThresh={args.grad_threshold}, "
          f"PWin={args.plasticity_window_size}, PtolFactor={args.plasticity_tolerance_factor}, "
          f"Weights=({args.plasticity_weight},{args.gradient_weight},{args.compute_weight}), "
          f"MinBenefit={args.min_compute_benefit}, MinLayer={args.min_freeze_layer}, MaxLayer={args.max_freeze_layer}, "
          f"DeltaMomentum={args.residual_momentum}, Refresh={args.residual_refresh_interval}, "
          f"Probe={args.residual_probe_prob}, CacheImpl={args.cache_impl}, DiskCache={not args.no_disk_cache}, CacheReuse={not args.disable_cache_reuse}, "
          f"LRUnfreeze={not args.disable_lr_unfreeze}, Target={args.target_acc}, "
          f"AugMode={args.aug_mode}, CacheKey={args.cache_key_mode}, "
          f"GradTrackEvery={args.grad_track_interval}, GradTrackScope={args.grad_track_scope}, "
          f"ForceSchedule={force_freeze_events}, CacheStartEpoch={args.cache_start_epoch}, "
          f"CacheStartAcc={args.cache_start_acc}, GateThresh={args.tcrc_gate_threshold}, "
          f"GateEvery={args.tcrc_gate_every}, StateDelta={args.tcrc_state_delta}")

    total_training_start = time.time()
    time_to_target_sec = None
    epoch_to_target = None
    best_acc = 0.0
    cache_reuse_was_allowed = False

    for epoch in range(args.epochs):
        start_time = time.time()

        if (not args.disable_lr_unfreeze) and freezer.check_lr_change(optimizer.param_groups[0]['lr']):
            print("  [Main] LR changed; unfreezing all layers and clearing cache")
            freezer.reset_after_lr()
            t = timing.start() if timing else None
            cache.clear()
            if timing:
                timing.stop('cache_clear_lr_drop', t)
            if ref_manager is not None:
                t = timing.start() if timing else None
                ref_manager.update(model)
                if timing:
                    timing.stop('reference_model_update', t)

        if (ref_manager is not None and epoch % args.ref_update_epochs == 0 and epoch != 0):
            t = timing.start() if timing else None
            ref_manager.update(model)
            if timing:
                timing.stop('reference_model_update', t)

        for sched_epoch, sched_layer in force_freeze_events:
            if epoch == sched_epoch:
                print(f"  [Main] SCHEDULED FORCE-FREEZE prefix {args.min_freeze_layer} -> {sched_layer} at epoch {epoch}")
                freezer.force_freeze_range(max_layer=sched_layer, min_layer=args.min_freeze_layer)
                t = timing.start() if timing else None
                cache.clear()
                if timing:
                    timing.stop('cache_clear_scheduled_force_freeze', t)

        if (not force_freeze_events and args.force_freeze_epoch >= 0 and epoch == args.force_freeze_epoch):
            print(f"  [Main] FORCE-FREEZING blockwise prefix {args.min_freeze_layer} -> {args.max_freeze_layer} at epoch {epoch}")
            freezer.force_freeze_range(max_layer=args.max_freeze_layer, min_layer=args.min_freeze_layer)
            t = timing.start() if timing else None
            cache.clear()
            if timing:
                timing.stop('cache_clear_force_freeze', t)

        # TTA-safe delayed cache activation.  Freezing can happen earlier, but
        # activation-cache reuse is only allowed after the requested epoch/accuracy
        # gates have opened.
        epoch_gate_ok = (args.cache_start_epoch < 0 or epoch >= args.cache_start_epoch)
        acc_gate_ok = (args.cache_start_acc < 0 or best_acc >= args.cache_start_acc)
        cache_reuse_allowed = bool(epoch_gate_ok and acc_gate_ok)
        if cache_reuse_allowed and not cache_reuse_was_allowed:
            print(f"  [Main] Cache reuse gate opened at epoch {epoch}: best_acc={best_acc:.2f}, warmup_batches={args.cache_start_warmup_batches}")
            if args.cache_start_warmup_batches > 0:
                freezer.cache_warmup = max(freezer.cache_warmup, int(args.cache_start_warmup_batches))
            t = timing.start() if timing else None
            cache.clear()
            if timing:
                timing.stop('cache_clear_cache_start', t)
        cache_reuse_was_allowed = cache_reuse_allowed
        effective_disable_cache_reuse = bool(args.disable_cache_reuse or (not cache_reuse_allowed))

        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, criterion, device,
            cache, freezer, evaluator, ref_manager, epoch,
            residual_refresh_interval=args.residual_refresh_interval,
            residual_probe_prob=args.residual_probe_prob,
            timing=timing,
            disable_cache_reuse=effective_disable_cache_reuse,
            cache_impl=args.cache_impl,
            tcrc_gate_threshold=args.tcrc_gate_threshold,
            tcrc_gate_every=args.tcrc_gate_every,
            tcrc_gate_samples=args.tcrc_gate_samples,
        )

        test_loss, test_acc = test(model, testloader, criterion, device)
        epoch_time = time.time() - start_time
        elapsed_total = time.time() - total_training_start
        best_acc = max(best_acc, test_acc)
        if time_to_target_sec is None and test_acc >= args.target_acc:
            time_to_target_sec = elapsed_total
            epoch_to_target = epoch

        stats = cache.get_stats()
        candidate = freezer.get_frontmost_active_layer()
        pstats = evaluator.get_stats(candidate) if candidate is not None else {}

        result = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_test_acc_so_far': best_acc,
            'target_acc': args.target_acc,
            'time_to_target_sec': time_to_target_sec,
            'epoch_to_target': epoch_to_target,
            'elapsed_total_sec': elapsed_total,
            'frozen_layers': ordered_frozen_layers(model.frozen_layers),
            'frontmost_frozen_layer': freezer.get_frontmost_frozen_layer(),
            'frontmost_active_layer': candidate,
            'cache_stats': stats,
            'plasticity_stats': pstats,
            'last_freeze_decision': freezer.last_decision,
            'gradient_tracking_stats': freezer.get_gradient_tracking_stats(),
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
            'timing': timing.snapshot() if timing else {},
            'config': {
                'script': 'resicache_plantcrc_v6_subpixel_resnet56_cifar10.py',
                'hybrid_mode': args.hybrid_mode,
                'cache_impl': args.cache_impl,
                'tcrc_crop_rounding': args.tcrc_crop_rounding,
                'tcrc_subpixel': args.tcrc_subpixel,
                'tcrc_probe_error_threshold': args.tcrc_probe_error_threshold,
                'cache_start_epoch': args.cache_start_epoch,
                'cache_start_acc': args.cache_start_acc,
                'cache_start_warmup_batches': args.cache_start_warmup_batches,
                'tcrc_gate_threshold': args.tcrc_gate_threshold,
                'tcrc_gate_every': args.tcrc_gate_every,
                'tcrc_gate_samples': args.tcrc_gate_samples,
                'tcrc_state_delta': args.tcrc_state_delta,
                'cache_reuse_allowed_this_epoch': cache_reuse_allowed,
                'disable_reference_model': args.disable_reference_model,
                'disable_cache_reuse': args.disable_cache_reuse,
                'disable_lr_unfreeze': args.disable_lr_unfreeze,
                'eval_interval': args.eval_interval,
                'bootstrap_epochs': args.bootstrap_epochs,
                'grad_window_size': args.grad_window_size,
                'grad_threshold': args.grad_threshold,
                'grad_track_interval': args.grad_track_interval,
                'grad_track_scope': args.grad_track_scope,
                'plasticity_window_size': args.plasticity_window_size,
                'plasticity_tolerance_factor': args.plasticity_tolerance_factor,
                'plasticity_weight': args.plasticity_weight,
                'gradient_weight': args.gradient_weight,
                'compute_weight': args.compute_weight,
                'min_compute_benefit': args.min_compute_benefit,
                'cache_overhead_fraction': args.cache_overhead_fraction,
                'min_freeze_layer': args.min_freeze_layer,
                'max_freeze_layer': args.max_freeze_layer,
                'force_freeze_epoch': args.force_freeze_epoch,
                'force_freeze_schedule': args.force_freeze_schedule,
                'parsed_force_freeze_events': force_freeze_events,
                'num_freeze_units': len(FREEZE_UNITS),
                'residual_momentum': args.residual_momentum,
                'residual_clip': args.residual_clip,
                'residual_refresh_interval': args.residual_refresh_interval,
                'residual_probe_prob': args.residual_probe_prob,
                'disk_cache': not args.no_disk_cache,
                'target_acc': args.target_acc,
                'seed': args.seed,
                'aug_mode': args.aug_mode,
                'cache_key_mode': args.cache_key_mode,
                'fixed_aug_seed': args.fixed_aug_seed,
            }
        }

        with open(args.output_json, 'a') as f:
            f.write(json.dumps(result) + '\n')

        print(f"[Main] Epoch {epoch} | Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | "
              f"Frozen: {ordered_frozen_layers(model.frozen_layers)} | "
              f"Cache: {100.0 * stats['hit_rate']:.1f}% | "
              f"DeltaUpd: {stats['delta_updates']} | "
              f"AvgDeltaNorm: {stats['avg_delta_norm']:.4f} | "
              f"TTA@{args.target_acc}: {time_to_target_sec} | "
              f"Time: {epoch_time:.1f}s")

        scheduler.step()

    print("[Main] ResiCache training completed!")
    print(f"[Main] Results saved to {args.output_json}")


if __name__ == '__main__':
    main()
