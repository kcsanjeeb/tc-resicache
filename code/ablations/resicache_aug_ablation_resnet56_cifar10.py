#!/usr/bin/env python3
"""
ResiCache augmentation/cache ablation experiment - ResNet-56 CIFAR-10.

Key changes from previous version:
1. shuffle=True in training DataLoader (standard practice)
2. Sample-level caching instead of batch-level (enables cache hits across epochs
   even with shuffling, because each sample's activation is keyed by its stable
   dataset index rather than the batch composition)
3. Deterministic shuffling via worker_init_fn and Generator for reproducibility

The sample-level cache stores activations per individual sample using the dataset
index as the key. This makes cache lookups invariant to batch shuffling: sample 42
will always find its cached activation regardless of which batch it appears in.
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
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

DATA_ROOT = '/mnt/data2/data'

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

    def forward_until_layer(self, x, layer_name):
        out = torch.relu(self.bn1(self.conv1(x)))
        if layer_name == 'conv1': return out
        out = self.layer1(out)
        if layer_name == 'layer1': return out
        out = self.layer2(out)
        if layer_name == 'layer2': return out
        out = self.layer3_0(out)
        if layer_name == 'layer3_0': return out
        out = self.layer3_1(out)
        if layer_name == 'layer3_1': return out
        out = self.layer3_2(out)
        if layer_name == 'layer3_2': return out
        out = self.layer3_3(out)
        if layer_name == 'layer3_3': return out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def forward_from_layer(self, x, layer_name):
        out = x
        if layer_name == 'conv1':
            out = self.layer1(out);
            out = self.layer2(out);
            out = self.layer3_0(out)
            out = self.layer3_1(out);
            out = self.layer3_2(out);
            out = self.layer3_3(out)
        elif layer_name == 'layer1':
            out = self.layer2(out);
            out = self.layer3_0(out)
            out = self.layer3_1(out);
            out = self.layer3_2(out);
            out = self.layer3_3(out)
        elif layer_name == 'layer2':
            out = self.layer3_0(out);
            out = self.layer3_1(out)
            out = self.layer3_2(out);
            out = self.layer3_3(out)
        elif layer_name == 'layer3_0':
            out = self.layer3_1(out);
            out = self.layer3_2(out);
            out = self.layer3_3(out)
        elif layer_name == 'layer3_1':
            out = self.layer3_2(out);
            out = self.layer3_3(out)
        elif layer_name == 'layer3_2':
            out = self.layer3_3(out)
        elif layer_name == 'layer3_3':
            pass
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def freeze_layer(self, layer_name):
        self.frozen_layers.add(layer_name)
        if layer_name == 'conv1':
            for p in [self.conv1, self.bn1]:
                for param in p.parameters(): param.requires_grad = False
        elif layer_name == 'layer1':
            for p in self.layer1.parameters(): p.requires_grad = False
        elif layer_name == 'layer2':
            for p in self.layer2.parameters(): p.requires_grad = False
        elif layer_name == 'layer3_0':
            for p in self.layer3_0.parameters(): p.requires_grad = False
        elif layer_name == 'layer3_1':
            for p in self.layer3_1.parameters(): p.requires_grad = False
        elif layer_name == 'layer3_2':
            for p in self.layer3_2.parameters(): p.requires_grad = False
        elif layer_name == 'layer3_3':
            for p in self.layer3_3.parameters(): p.requires_grad = False

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
        m = self.reference_model
        with torch.no_grad():
            out = torch.relu(m.bn1(m.conv1(x_cpu)))
            if layer_name == 'conv1': return out
            out = m.layer1(out)
            if layer_name == 'layer1': return out
            out = m.layer2(out)
            if layer_name == 'layer2': return out
            out = m.layer3_0(out)
            if layer_name == 'layer3_0': return out
            out = m.layer3_1(out)
            if layer_name == 'layer3_1': return out
            out = m.layer3_2(out)
            if layer_name == 'layer3_2': return out
            out = m.layer3_3(out)
            if layer_name == 'layer3_3': return out
        return out


class PlasticityEvaluator:
    """Egeria-style SP-loss plasticity tracker."""
    def __init__(self, window_size=10, tolerance_factor=0.20, min_tolerance=1e-4):
        self.window_size = window_size
        self.tolerance_factor = tolerance_factor
        self.min_tolerance = min_tolerance
        self.layer_names = ['conv1', 'layer1', 'layer2', 'layer3_0', 'layer3_1', 'layer3_2', 'layer3_3']
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
        vals = self.smoothed[layer_name][-self.window_size:]
        if len(vals) < self.window_size:
            return None
        x = np.arange(len(vals), dtype=np.float64)
        y = np.array(vals, dtype=np.float64)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(abs(m))

    def get_stats(self, layer_name):
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
                 min_freeze_layer='layer1'):
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
        self.min_freeze_layer = min_freeze_layer
        self.max_freeze_layer = max_freeze_layer

        self.layer_names = ['conv1', 'layer1', 'layer2', 'layer3_0', 'layer3_1', 'layer3_2', 'layer3_3']
        self.layer_compute_fraction = {
            'conv1': 0.02,
            'layer1': 0.12,
            'layer2': 0.35,
            'layer3_0': 0.55,
            'layer3_1': 0.70,
            'layer3_2': 0.85,
            'layer3_3': 0.97,
        }
        self.gradient_history = {l: [] for l in self.layer_names}
        self.grad_reference = {}
        self.iteration = 0
        self.initial_lr = None
        self.last_lr = None
        self.cache_warmup = 0
        self.last_decision = {}

    def compute_gradient_norms(self):
        grad_norms = {}
        if self.model.conv1.weight.grad is not None:
            grad_norms['conv1'] = self.model.conv1.weight.grad.norm().item()
        for name in ['layer1', 'layer2', 'layer3_0', 'layer3_1', 'layer3_2', 'layer3_3']:
            module = getattr(self.model, name)
            norm = sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None)
            if norm > 0:
                grad_norms[name] = norm
        return grad_norms

    def update_gradient_history(self):
        grad_norms = self.compute_gradient_norms()
        for layer, norm in grad_norms.items():
            self.gradient_history[layer].append(float(norm))
            if len(self.gradient_history[layer]) > self.grad_window_size:
                self.gradient_history[layer].pop(0)
            # Use early maximum as the normalization reference.
            self.grad_reference[layer] = max(self.grad_reference.get(layer, norm), norm)

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
        min_idx = self.layer_names.index(self.min_freeze_layer)
        idx = self.layer_names.index(layer_name)

        for i in range(min_idx, idx):
            if self.layer_names[i] not in self.model.frozen_layers:
                return False

        return True

    def _min_layer_ok(self, layer_name):
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
        slope = self.evaluator.slope(layer_name)
        tol = self.evaluator.tolerance.get(layer_name, None)
        if slope is None or tol is None:
            return None
        return slope / max(tol, 1e-8)

    def _compute_benefit(self, layer_name):
        saved = self.layer_compute_fraction[layer_name]
        benefit = saved - self.cache_overhead_fraction
        return benefit

    def maybe_freeze(self, layer_name):
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
    """CIFAR-10 training dataset that returns an integer cache key.

    aug_mode:
      - random: standard RandomCrop(32, padding=4) + RandomHorizontalFlip.
      - fixed_sample: deterministic crop/flip per sample id, stable across epochs.
      - fixed_epoch: deterministic crop/flip per sample id and epoch.
      - none: no random augmentation; only ToTensor + Normalize.

    cache_key_mode:
      - sample: key = sample id only; previous ResiCache behavior.
      - sample_aug: key = sample id + crop/flip code; avoids reusing cached
        activations from a different random augmentation of the same image.
      - sample_epoch: key = sample id + epoch; diagnostic, low cross-epoch reuse.
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

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = self.targets[index]
        i, j, flip, aug_code = self._sample_aug_params(index)

        if self.train and self.aug_mode != 'none':
            img = TF.pad(img, 4, padding_mode='constant')
            img = TF.crop(img, i, j, 32, 32)
            if flip:
                img = TF.hflip(img)

        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)
        cache_key = self._make_cache_key(index, aug_code)
        return cache_key, img, target


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
                timing=None, disable_cache_reuse=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Inform augmentation-aware datasets which epoch is being executed.
    if hasattr(trainloader.dataset, 'set_epoch'):
        trainloader.dataset.set_epoch(epoch)

    # Keep frozen BatchNorm/blocks in eval mode.
    for lname in model.frozen_layers:
        if lname == 'conv1':
            model.bn1.eval()
        elif hasattr(model, lname):
            getattr(model, lname).eval()

    for batch_idx, (cache_keys, inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        sample_ids = cache_keys.tolist()

        t0 = timing.start() if timing else None
        optimizer.zero_grad()
        if timing:
            timing.stop('optimizer_zero_grad', t0)

        active_frozen_layer = freezer.get_frontmost_frozen_layer()
        in_warmup = freezer.in_cache_warmup()
        use_cache_layer = (active_frozen_layer and active_frozen_layer != 'layer3_3' and not disable_cache_reuse)

        force_refresh = False
        if use_cache_layer and not in_warmup:
            if residual_refresh_interval > 0 and batch_idx % residual_refresh_interval == 0:
                force_refresh = True
            elif residual_probe_prob > 0.0 and random.random() < residual_probe_prob:
                force_refresh = True

        t_forward = timing.start() if timing else None
        if use_cache_layer and in_warmup:
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
                  f'Frozen: {sorted(list(model.frozen_layers))} | '
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
    parser.add_argument('--plasticity_window_size', type=int, default=10)
    parser.add_argument('--plasticity_tolerance_factor', type=float, default=0.20)
    parser.add_argument('--plasticity_weight', type=float, default=0.50)
    parser.add_argument('--gradient_weight', type=float, default=0.30)
    parser.add_argument('--compute_weight', type=float, default=0.20)
    parser.add_argument('--min_compute_benefit', type=float, default=0.10)
    parser.add_argument('--cache_overhead_fraction', type=float, default=0.05)
    parser.add_argument('--min_freeze_layer', type=str, default='layer1',
                        choices=['conv1', 'layer1', 'layer2', 'layer3_0', 'layer3_1', 'layer3_2', 'layer3_3'])
    parser.add_argument('--max_freeze_layer', type=str, default='layer3_3',
                        choices=['conv1', 'layer1', 'layer2', 'layer3_0', 'layer3_1', 'layer3_2', 'layer3_3'])
    parser.add_argument('--ref_update_epochs', type=int, default=10)
    parser.add_argument('--disable_reference_model', action='store_true',
                        help='Disable CPU reference model and SP-loss plasticity path. Requires --hybrid_mode gradient_only.')
    parser.add_argument('--disable_lr_unfreeze', action='store_true',
                        help='Do not unfreeze layers when LR drops. Useful for speed/accuracy ablation.')

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
    # Validate layer range.
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3_0', 'layer3_1', 'layer3_2', 'layer3_3']
    if layer_names.index(args.max_freeze_layer) < layer_names.index(args.min_freeze_layer):
        raise ValueError('--max_freeze_layer must be at or after --min_freeze_layer')

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

    cache = DeltaResidualSampleCache(
        cache_dir=args.cache_dir,
        gpu_cache_size=args.gpu_cache_size,
        device=device,
        residual_momentum=args.residual_momentum,
        residual_clip=None if args.residual_clip <= 0 else args.residual_clip,
        disk_cache=not args.no_disk_cache
    )

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
          f"Probe={args.residual_probe_prob}, DiskCache={not args.no_disk_cache}, CacheReuse={not args.disable_cache_reuse}, "
          f"LRUnfreeze={not args.disable_lr_unfreeze}, Target={args.target_acc}, "
          f"AugMode={args.aug_mode}, CacheKey={args.cache_key_mode}")

    total_training_start = time.time()
    time_to_target_sec = None
    epoch_to_target = None
    best_acc = 0.0

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

        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, criterion, device,
            cache, freezer, evaluator, ref_manager, epoch,
            residual_refresh_interval=args.residual_refresh_interval,
            residual_probe_prob=args.residual_probe_prob,
            timing=timing,
            disable_cache_reuse=args.disable_cache_reuse,
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
            'frozen_layers': sorted(list(model.frozen_layers)),
            'frontmost_frozen_layer': freezer.get_frontmost_frozen_layer(),
            'frontmost_active_layer': candidate,
            'cache_stats': stats,
            'plasticity_stats': pstats,
            'last_freeze_decision': freezer.last_decision,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
            'timing': timing.snapshot() if timing else {},
            'config': {
                'script': 'resicache_aug_ablation_resnet56_cifar10.py',
                'hybrid_mode': args.hybrid_mode,
                'disable_reference_model': args.disable_reference_model,
                'disable_cache_reuse': args.disable_cache_reuse,
                'disable_lr_unfreeze': args.disable_lr_unfreeze,
                'eval_interval': args.eval_interval,
                'bootstrap_epochs': args.bootstrap_epochs,
                'grad_window_size': args.grad_window_size,
                'grad_threshold': args.grad_threshold,
                'plasticity_window_size': args.plasticity_window_size,
                'plasticity_tolerance_factor': args.plasticity_tolerance_factor,
                'plasticity_weight': args.plasticity_weight,
                'gradient_weight': args.gradient_weight,
                'compute_weight': args.compute_weight,
                'min_compute_benefit': args.min_compute_benefit,
                'cache_overhead_fraction': args.cache_overhead_fraction,
                'min_freeze_layer': args.min_freeze_layer,
                'max_freeze_layer': args.max_freeze_layer,
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
              f"Frozen: {sorted(list(model.frozen_layers))} | "
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
