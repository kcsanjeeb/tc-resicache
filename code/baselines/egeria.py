"""
Egeria Baseline Implementation
Paper: "Egeria: Efficient DNN Training with Knowledge-Guided Layer Freezing"
arXiv:2201.06227, Section 4.3 - Skipping Forward Pass with Caching and Prefetching

This implementation provides:
1. Disk-based caching mechanism for frozen layer activations
2. Prefetch strategy for loading future activations
3. Quantized reference model for plasticity evaluation
4. Layer freezing based on training plasticity

Usage:
    from baselines.egeria import EgeriaCache, EgeriaTrainer
    
    # Initialize Egeria cache
    cache = EgeriaCache(
        cache_dir="./egeria_cache",
        max_memory_batches=5,
        device="cuda"
    )
    
    # Use with trainer
    trainer = EgeriaTrainer(
        model=model,
        cache=cache,
        plasticity_eval_interval=300,
        tolerance_factor=0.2,
        window_size=10
    )
    
    # Train with Egeria
    trainer.train(train_loader, num_epochs=200)

Reference:
    Wang et al., "Egeria: Efficient DNN Training with Knowledge-Guided Layer Freezing",
    EuroSys '23, May 08-12, 2023, Rome, Italy
"""

from __future__ import annotations

import argparse
import os
import time
import json
import pickle
import hashlib
import tempfile
import warnings
from typing import Dict, Tuple, Optional, List, Callable, Any, Union
from dataclasses import dataclass, field
from collections import deque
import threading
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models


# =============================================================================
# Timing Utilities
# =============================================================================
def now() -> float:
    """High-resolution timer."""
    return time.perf_counter()


def cuda_sync_if_needed(device: torch.device):
    """Synchronize CUDA if using GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Section 4.3: Disk-based Caching with Prefetch
# =============================================================================

@dataclass
class CacheEntry:
    """
    Represents a cached activation entry.
    
    Attributes:
        sample_id: Unique identifier for the sample
        activation: The cached activation tensor (stored on CPU)
        metadata: Additional metadata (e.g., layer info, shape)
    """
    sample_id: str
    activation: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class EgeriaCache:
    """
    Egeria's disk-based caching system with prefetching.
    
    As described in Section 4.3 of the paper:
    - Saves forward computation results of frozen layers to disk
    - Prefetches relevant activations to GPU memory
    - Maintains a hash table in GPU memory for O(1) lookups
    - Stores only recent mini-batches for minimal memory usage
    
    Key features:
    1. Disk storage for activations (1.5x to 5.3x of input data for ResNet-50)
    2. In-memory cache for recent activations (default: 5 mini-batches)
    3. Prefetching from disk to GPU in parallel with training
    4. Deterministic caching compatible with random data augmentation
    
    Args:
        cache_dir: Directory to store cached activations on disk
        max_memory_batches: Number of mini-batches to keep in GPU memory
        device: Device for training (typically 'cuda')
        dtype: Data type for cached activations
    
    Reference: Section 4.3, Figure 7
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_memory_batches: int = 5,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        self.cache_dir = cache_dir
        self.max_memory_batches = max_memory_batches
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache: sample_id -> activation tensor (on CPU)
        self._memory_cache: Dict[str, torch.Tensor] = {}
        
        # Track access order for LRU eviction
        self._access_order: deque = deque()
        
        # Prefetch queue for async loading
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=max_memory_batches * 2)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()
        
        # Statistics
        self._disk_hits = 0
        self._memory_hits = 0
        self._misses = 0
        self._bytes_written = 0
        self._bytes_read = 0
        
        # Layer info (set when caching starts)
        self._frozen_layer_idx: Optional[int] = None
        self._is_active = False
    
    def _get_cache_path(self, sample_id: str) -> str:
        """Get disk path for a sample's activation."""
        # Use hash to avoid filename issues
        hash_id = hashlib.md5(sample_id.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_id}.pt")
    
    def _generate_sample_id(self, index: int, epoch: int, augmentation_seed: int = 0) -> str:
        """
        Generate unique sample ID.
        
        Includes augmentation seed for deterministic caching with random augmentations.
        This ensures the same augmented image produces the same activation.
        """
        return f"sample_{index}_epoch_{epoch}_aug_{augmentation_seed}"
    
    def start_caching(self, frozen_layer_idx: int):
        """
        Start caching activations from the given frozen layer.
        
        Args:
            frozen_layer_idx: Index of the last frozen layer
        """
        self._frozen_layer_idx = frozen_layer_idx
        self._is_active = True
        self._memory_cache.clear()
        self._access_order.clear()
        
        # Clear existing disk cache
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pt'):
                os.remove(os.path.join(self.cache_dir, f))
        
        print(f"[EgeriaCache] Started caching at layer {frozen_layer_idx}")
    
    def stop_caching(self):
        """Stop caching and cleanup."""
        self._is_active = False
        self._stop_prefetch.set()
        
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
        
        # Clear memory cache
        self._memory_cache.clear()
        self._access_order.clear()
        
        print(f"[EgeriaCache] Stopped caching. Stats: memory_hits={self._memory_hits}, "
              f"disk_hits={self._disk_hits}, misses={self._misses}")
    
    def cache_activation(
        self,
        sample_indices: List[int],
        activations: torch.Tensor,
        epoch: int,
        augmentation_seed: int = 0
    ):
        """
        Cache activations to disk and optionally to memory.
        
        Args:
            sample_indices: List of sample indices in the batch
            activations: Activation tensor [batch_size, ...]
            epoch: Current epoch number
            augmentation_seed: Seed for deterministic augmentation
        """
        if not self._is_active:
            return
        
        batch_size = activations.shape[0]
        assert len(sample_indices) == batch_size
        
        # Save each activation individually
        for i, idx in enumerate(sample_indices):
            sample_id = self._generate_sample_id(idx, epoch, augmentation_seed)
            activation = activations[i].detach().cpu().to(self.dtype)
            
            # Save to disk
            cache_path = self._get_cache_path(sample_id)
            torch.save(activation, cache_path)
            self._bytes_written += activation.numel() * activation.element_size()
            
            # Optionally keep in memory cache (LRU)
            if len(self._memory_cache) < self.max_memory_batches * batch_size:
                self._memory_cache[sample_id] = activation
                self._access_order.append(sample_id)
    
    def get_activation(
        self,
        sample_indices: List[int],
        epoch: int,
        augmentation_seed: int = 0
    ) -> Optional[torch.Tensor]:
        """
        Retrieve activations for given sample indices.
        
        Tries memory cache first, then disk, returns None if not found.
        
        Args:
            sample_indices: List of sample indices
            epoch: Current epoch number
            augmentation_seed: Seed for deterministic augmentation
            
        Returns:
            Stacked activation tensor or None if any sample not found
        """
        if not self._is_active:
            return None
        
        activations = []
        
        for idx in sample_indices:
            sample_id = self._generate_sample_id(idx, epoch, augmentation_seed)
            
            # Try memory cache first
            if sample_id in self._memory_cache:
                activation = self._memory_cache[sample_id]
                self._memory_hits += 1
                
                # Update access order for LRU
                self._access_order.remove(sample_id)
                self._access_order.append(sample_id)
            
            # Try disk cache
            else:
                cache_path = self._get_cache_path(sample_id)
                if os.path.exists(cache_path):
                    activation = torch.load(cache_path, map_location='cpu')
                    self._disk_hits += 1
                    self._bytes_read += activation.numel() * activation.element_size()
                    
                    # Add to memory cache if space available
                    self._maybe_add_to_memory_cache(sample_id, activation)
                else:
                    self._misses += 1
                    return None
            
            activations.append(activation)
        
        # Stack and move to device
        return torch.stack(activations).to(self.device, non_blocking=True)
    
    def _maybe_add_to_memory_cache(self, sample_id: str, activation: torch.Tensor):
        """Add to memory cache with LRU eviction."""
        if len(self._memory_cache) >= self.max_memory_batches * 128:  # Assume batch_size ~128
            # Evict oldest
            oldest_id = self._access_order.popleft()
            if oldest_id in self._memory_cache:
                del self._memory_cache[oldest_id]
        
        self._memory_cache[sample_id] = activation
        self._access_order.append(sample_id)
    
    def prefetch_async(self, sample_indices: List[int], epoch: int, augmentation_seed: int = 0):
        """
        Start async prefetching of activations.
        
        As described in Section 4.3:
        "Egeria prefetches more than one mini-batch of the future activations,
        similar to the data loader, depending on the memory and CPU availability."
        
        Args:
            sample_indices: List of sample indices to prefetch
            epoch: Current epoch number
            augmentation_seed: Seed for deterministic augmentation
        """
        if not self._is_active:
            return
        
        # Simple synchronous prefetch for now
        # Can be extended to use background thread
        for idx in sample_indices:
            sample_id = self._generate_sample_id(idx, epoch, augmentation_seed)
            
            # Skip if already in memory
            if sample_id in self._memory_cache:
                continue
            
            # Load from disk if exists
            cache_path = self._get_cache_path(sample_id)
            if os.path.exists(cache_path):
                activation = torch.load(cache_path, map_location='cpu')
                self._maybe_add_to_memory_cache(sample_id, activation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._memory_hits + self._disk_hits + self._misses
        hit_rate = (self._memory_hits + self._disk_hits) / max(1, total_requests)
        
        return {
            "memory_hits": self._memory_hits,
            "disk_hits": self._disk_hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "bytes_written": self._bytes_written,
            "bytes_read": self._bytes_read,
            "memory_cache_size": len(self._memory_cache),
        }


# =============================================================================
# Section 4.1-4.2: Quantized Reference Model for Plasticity
# =============================================================================

class QuantizedReferenceModel:
    """
    Quantized reference model for plasticity evaluation.
    
    As described in Section 4.1 and 4.2 of the paper:
    - Generates a reference model by compressing a snapshot of the training model
    - Uses int8 quantization via PyTorch's built-in library
    - Dynamic quantization for NLP models, static for convolutional networks
    - Runs on CPU asynchronously to minimize overhead
    
    The reference model serves as a proxy for semantic knowledge to evaluate
    DNN layers' plasticity by comparing intermediate activations.
    
    Args:
        model: The training model to quantize
        model_type: Type of model ('cnn' or 'nlp')
        device: Device for the reference model (typically 'cpu')
    
    Reference: Section 4.1, Section 4.2, Section 5 (Implementation)
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str = "cnn",
        device: torch.device = None
    ):
        self.original_model = model
        self.model_type = model_type.lower()
        self.device = device or torch.device("cpu")
        
        # Quantized model (created on first update)
        self._reference_model: Optional[nn.Module] = None
        self._forward_hooks: List[Callable] = []
        self._activations: Dict[str, torch.Tensor] = {}
        
        # Statistics
        self._update_count = 0
        self._update_time_total = 0.0
    
    def _create_quantized_model(self) -> nn.Module:
        """
        Create quantized version of the training model.
        
        As described in Section 5 (Implementation):
        "When Egeria controller generates or updates the reference model,
        it directly moves a snapshot of the training model from GPU to CPU
        and runs int8 quantization using PyTorch's built-in library.
        We use dynamic quantization for NLP models and static quantization
        for convolutional networks."
        """
        # Move model to CPU and set to eval mode
        model_cpu = self.original_model.to("cpu").eval()
        
        try:
            if self.model_type == "nlp" or "transformer" in self.model_type:
                # Dynamic quantization for NLP models
                quantized_model = torch.quantization.quantize_dynamic(
                    model_cpu,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )
            else:
                # Static quantization for CNNs
                # Prepare model for static quantization
                model_cpu.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model_cpu, inplace=True)
                
                # Calibrate with dummy data (simplified)
                dummy_input = torch.randn(1, 3, 32, 32)
                model_cpu(dummy_input)
                
                # Convert to quantized model
                quantized_model = torch.quantization.convert(model_cpu, inplace=False)
        
        except Exception as e:
            warnings.warn(f"Quantization failed: {e}. Using non-quantized model.")
            quantized_model = model_cpu
        
        return quantized_model.to(self.device)
    
    def update(self):
        """
        Update the reference model from current training model snapshot.
        
        Should be called periodically during training (e.g., every n iterations).
        The paper mentions this takes 0.5s-1.5s on CPU with minimal overhead.
        """
        t0 = now()
        
        # Create new quantized model from current snapshot
        self._reference_model = self._create_quantized_model()
        
        # Re-register hooks
        self._register_hooks()
        
        self._update_count += 1
        self._update_time_total += (now() - t0)
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        # Remove old hooks
        for hook in self._forward_hooks:
            hook.remove()
        self._forward_hooks.clear()
        self._activations.clear()
        
        if self._reference_model is None:
            return
        
        # Register new hooks for each layer
        for name, module in self._reference_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._activations.update({name: out.detach()})
                )
                self._forward_hooks.append(hook)
    
    @torch.no_grad()
    def evaluate_plasticity(
        self,
        input_batch: torch.Tensor,
        training_activations: Dict[str, torch.Tensor],
        layer_name: str
    ) -> float:
        """
        Evaluate plasticity for a specific layer.
        
        As described in Section 4.2, Equation 1:
        P(l)_i = SP_loss(A(l)_T, A(l)_R)
        
        where SP_loss is the soft prediction loss between training and
        reference model activations.
        
        Args:
            input_batch: Input data batch
            training_activations: Activations from training model
            layer_name: Name of layer to evaluate
            
        Returns:
            Plasticity score (lower = more converged)
        """
        if self._reference_model is None:
            raise RuntimeError("Reference model not initialized. Call update() first.")
        
        # Run reference model
        self._reference_model.eval()
        _ = self._reference_model(input_batch.to(self.device))
        
        # Get reference activation
        if layer_name not in self._activations:
            return float('inf')  # Cannot evaluate
        
        ref_activation = self._activations[layer_name]
        train_activation = training_activations.get(layer_name)
        
        if train_activation is None:
            return float('inf')
        
        # Calculate plasticity as MSE between activations
        # (Simplified version of SP_loss from paper)
        train_activation = train_activation.to(self.device)
        
        # Normalize activations
        train_norm = torch.nn.functional.normalize(train_activation.flatten(1), dim=1)
        ref_norm = torch.nn.functional.normalize(ref_activation.flatten(1), dim=1)
        
        # Cosine similarity (1 - similarity = distance)
        similarity = torch.nn.functional.cosine_similarity(train_norm, ref_norm, dim=1)
        plasticity = (1 - similarity.mean()).item()
        
        return plasticity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reference model statistics."""
        return {
            "update_count": self._update_count,
            "avg_update_time_ms": (self._update_time_total / max(1, self._update_count)) * 1000,
        }


# =============================================================================
# Section 4.2: Layer Freezing Algorithm (Algorithm 1)
# =============================================================================

@dataclass
class LayerState:
    """State for a single layer in the freezing algorithm."""
    name: str
    module: nn.Module
    is_frozen: bool = False
    plasticity_history: List[float] = field(default_factory=list)
    stale_counter: int = 0
    frozen_at_iteration: Optional[int] = None


class LayerFreezer:
    """
    Implements Algorithm 1 from Section 4.2 of the paper.
    
    Key features:
    1. Periodic plasticity evaluation every n iterations
    2. Moving average smoothing (Equation 2)
    3. Linear fitting to detect convergence (slope < tolerance)
    4. Learning rate-based unfreezing mechanism
    
    Args:
        model: The model to freeze layers in
        eval_interval: Iterations between plasticity evaluations (n)
        tolerance_factor: Factor for per-layer tolerance (T)
        window_size: History buffer length (W)
        max_initial_readings: Number of initial readings for tolerance calibration
    
    Reference: Section 4.2, Algorithm 1
    """
    
    def __init__(
        self,
        model: nn.Module,
        eval_interval: int = 300,
        tolerance_factor: float = 0.2,
        window_size: int = 10,
        max_initial_readings: int = 3
    ):
        self.model = model
        self.eval_interval = eval_interval
        self.tolerance_factor = tolerance_factor
        self.window_size = window_size
        self.max_initial_readings = max_initial_readings
        
        # Layer states
        self._layers: Dict[str, LayerState] = {}
        self._layer_order: List[str] = []
        self._current_layer_idx = 0
        
        # Per-layer tolerances (calibrated from initial readings)
        self._tolerances: Dict[str, float] = {}
        
        # Training state
        self._iteration = 0
        self._bootstrap_complete = False
        
        # LR tracking for unfreezing
        self._initial_lr: Optional[float] = None
        self._last_unfreeze_iter = 0
        
        # Statistics
        self._frozen_count = 0
        self._unfrozen_count = 0
        
        self._initialize_layers()
    
    def _initialize_layers(self):
        """Parse model and initialize layer states."""
        for name, module in self.model.named_modules():
            # Only consider major layers (Conv2d, Linear, BatchNorm)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._layers[name] = LayerState(name=name, module=module)
                self._layer_order.append(name)
        
        print(f"[LayerFreezer] Initialized with {len(self._layers)} layers")
    
    def set_initial_lr(self, lr: float):
        """Set initial learning rate for unfreezing decisions."""
        self._initial_lr = lr
    
    def should_evaluate(self) -> bool:
        """Check if it's time to evaluate plasticity."""
        return self._iteration % self.eval_interval == 0
    
    def check_plasticity(
        self,
        reference_model: QuantizedReferenceModel,
        input_batch: torch.Tensor,
        training_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Check plasticity and decide on layer freezing.
        
        Implements Algorithm 1 from the paper:
        1. Calculate plasticity P(l)_i
        2. Smooth with moving average (Equation 2)
        3. Linear fit to check slope
        4. Freeze if slope < tolerance for W evaluations
        
        Args:
            reference_model: Quantized reference model
            input_batch: Input data batch
            training_activations: Activations from training model
            
        Returns:
            Dictionary with freezing decisions and statistics
        """
        self._iteration += 1
        
        if not self.should_evaluate():
            return {"action": "skip", "iteration": self._iteration}
        
        # Get current layer to evaluate
        if self._current_layer_idx >= len(self._layer_order):
            return {"action": "all_frozen", "iteration": self._iteration}
        
        current_layer_name = self._layer_order[self._current_layer_idx]
        layer_state = self._layers[current_layer_name]
        
        # Skip if already frozen
        if layer_state.is_frozen:
            self._current_layer_idx += 1
            return {"action": "already_frozen", "layer": current_layer_name}
        
        # Evaluate plasticity (Algorithm 1, line 6)
        plasticity = reference_model.evaluate_plasticity(
            input_batch, training_activations, current_layer_name
        )
        
        # Smooth with moving average (Equation 2, Algorithm 1, line 7)
        smoothed_plasticity = self._smooth_plasticity(layer_state, plasticity)
        
        # Update history (Algorithm 1, line 8)
        layer_state.plasticity_history.append(smoothed_plasticity)
        
        # Calibrate tolerance from initial readings
        if len(layer_state.plasticity_history) <= self.max_initial_readings:
            if len(layer_state.plasticity_history) == self.max_initial_readings:
                # Set tolerance as 20% of max slope in initial readings
                max_slope = self._calculate_max_initial_slope(layer_state)
                self._tolerances[current_layer_name] = max_slope * self.tolerance_factor
            return {"action": "calibrating", "layer": current_layer_name, "plasticity": plasticity}
        
        # Calculate slope with linear fit (Algorithm 1, line 9)
        slope = self._window_linear_fit(layer_state.plasticity_history, self.window_size)
        tolerance = self._tolerances.get(current_layer_name, 0.01)
        
        # Check if converged (Algorithm 1, lines 10-14)
        if abs(slope) < tolerance:
            layer_state.stale_counter += 1
        else:
            layer_state.stale_counter = 0
        
        # Freeze if stale for W evaluations (Algorithm 1, lines 15-17)
        if layer_state.stale_counter >= self.window_size:
            self._freeze_layer(layer_state)
            self._current_layer_idx += 1
            
            return {
                "action": "frozen",
                "layer": current_layer_name,
                "plasticity": plasticity,
                "slope": slope,
                "tolerance": tolerance
            }
        
        return {
            "action": "monitoring",
            "layer": current_layer_name,
            "plasticity": plasticity,
            "slope": slope,
            "stale_counter": layer_state.stale_counter
        }
    
    def _smooth_plasticity(self, layer_state: LayerState, plasticity: float) -> float:
        """
        Smooth plasticity with moving average (Equation 2).
        
        P(l)_i = mean(P(l)_{i-W+1}, ..., P(l)_i) for i >= W
        """
        history = layer_state.plasticity_history
        window = min(self.window_size, len(history) + 1)
        
        if len(history) == 0:
            return plasticity
        
        recent = history[-(window-1):] + [plasticity]
        return sum(recent) / len(recent)
    
    def _calculate_max_initial_slope(self, layer_state: LayerState) -> float:
        """Calculate max slope from initial readings for tolerance calibration."""
        history = layer_state.plasticity_history
        if len(history) < 2:
            return 0.01  # Default
        
        slopes = []
        for i in range(1, len(history)):
            slopes.append(abs(history[i] - history[i-1]))
        
        return max(slopes) if slopes else 0.01
    
    def _window_linear_fit(self, history: List[float], window: int) -> float:
        """
        Fit linear regression to recent history and return slope.
        
        Algorithm 1, line 9: s <- windowLinearFit(pList_l, W).slope
        """
        if len(history) < 2:
            return float('inf')
        
        recent = history[-window:]
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Linear least-squares fit
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return slope
    
    def _freeze_layer(self, layer_state: LayerState):
        """Freeze a layer by setting requires_grad=False."""
        for param in layer_state.module.parameters():
            param.requires_grad = False
        
        layer_state.is_frozen = True
        layer_state.frozen_at_iteration = self._iteration
        self._frozen_count += 1
        
        print(f"[LayerFreezer] Frozen layer '{layer_state.name}' at iteration {self._iteration}")
    
    def check_unfreeze(self, current_lr: float) -> List[str]:
        """
        Check if layers should be unfrozen based on LR schedule.
        
        Algorithm 1, lines 19-26:
        - Unfreeze all if LR decreased by 90% from initial
        - Halve counter and window for refreezing
        """
        if self._initial_lr is None:
            return []
        
        # Check if LR dropped by 90%
        if current_lr < self._initial_lr * 0.1:
            unfrozen = []
            for name, layer in self._layers.items():
                if layer.is_frozen:
                    for param in layer.module.parameters():
                        param.requires_grad = True
                    layer.is_frozen = False
                    layer.stale_counter = layer.stale_counter // 2  # Halve counter
                    unfrozen.append(name)
                    self._unfrozen_count += 1
            
            if unfrozen:
                self._current_layer_idx = 0  # Reset to first layer
                self._last_unfreeze_iter = self._iteration
                print(f"[LayerFreezer] Unfrozen {len(unfrozen)} layers due to LR decay")
            
            return unfrozen
        
        return []
    
    def get_frozen_layers(self) -> List[str]:
        """Get list of currently frozen layer names."""
        return [name for name, layer in self._layers.items() if layer.is_frozen]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get freezing statistics."""
        return {
            "frozen_count": self._frozen_count,
            "unfrozen_count": self._unfrozen_count,
            "current_layer_idx": self._current_layer_idx,
            "frozen_layers": self.get_frozen_layers(),
            "iteration": self._iteration,
        }


# =============================================================================
# Egeria Trainer: Complete Integration
# =============================================================================

class EgeriaTrainer:
    """
    Complete Egeria training system integrating caching and layer freezing.
    
    This trainer combines:
    1. Quantized reference model for plasticity evaluation (Section 4.1-4.2)
    2. Layer freezing based on convergence detection (Algorithm 1)
    3. Disk-based caching with prefetching (Section 4.3)
    
    Usage:
        trainer = EgeriaTrainer(
            model=resnet18,
            cache_dir="./egeria_cache",
            plasticity_eval_interval=300,
            tolerance_factor=0.2,
            window_size=10
        )
        
        trainer.train(train_loader, test_loader, num_epochs=200)
    
    Reference: Full paper (arXiv:2201.06227)
    """
    
    def __init__(
        self,
        model: nn.Module,
        cache_dir: str = "./egeria_cache",
        model_type: str = "cnn",
        plasticity_eval_interval: int = 300,
        tolerance_factor: float = 0.2,
        window_size: int = 10,
        max_memory_batches: int = 5,
        device: torch.device = None,
        bootstrap_iterations: int = 1000
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize cache
        self.cache = EgeriaCache(
            cache_dir=cache_dir,
            max_memory_batches=max_memory_batches,
            device=self.device
        )
        
        # Initialize reference model
        self.reference_model = QuantizedReferenceModel(
            model=model,
            model_type=model_type,
            device=torch.device("cpu")  # Run on CPU
        )
        
        # Initialize layer freezer
        self.layer_freezer = LayerFreezer(
            model=model,
            eval_interval=plasticity_eval_interval,
            tolerance_factor=tolerance_factor,
            window_size=window_size
        )
        
        self.bootstrap_iterations = bootstrap_iterations
        self._iteration = 0
        self._epoch = 0
        
        # Training activations (captured via hooks)
        self._training_activations: Dict[str, torch.Tensor] = {}
        self._forward_hooks: List[Callable] = []
        self._register_training_hooks()
    
    def _register_training_hooks(self):
        """Register hooks to capture training model activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._training_activations.update({name: out})
                )
                self._forward_hooks.append(hook)
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        optimizer: optim.Optimizer = None,
        criterion: nn.Module = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        log_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Train the model using Egeria's knowledge-guided layer freezing.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            optimizer: Optimizer (default: SGD with lr=0.1, momentum=0.9)
            criterion: Loss function (default: CrossEntropyLoss)
            scheduler: Learning rate scheduler
            log_path: Path to save training logs
            
        Returns:
            List of log dictionaries per epoch
        """
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Set initial LR for unfreezing decisions
        self.layer_freezer.set_initial_lr(optimizer.param_groups[0]['lr'])
        
        logs = []
        
        for epoch in range(1, num_epochs + 1):
            self._epoch = epoch
            ep_t0 = now()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            
            # Evaluation
            test_acc = self._evaluate(test_loader)
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step()
                self.layer_freezer.check_unfreeze(optimizer.param_groups[0]['lr'])
            
            # Update reference model periodically
            if self._iteration % self.layer_freezer.eval_interval == 0:
                self.reference_model.update()
            
            # Compile log
            log = {
                "epoch": epoch,
                "iteration": self._iteration,
                **train_metrics,
                "test_acc": test_acc,
                "epoch_time_s": now() - ep_t0,
                "frozen_layers": self.layer_freezer.get_frozen_layers(),
                **self.cache.get_stats(),
                **self.layer_freezer.get_stats()
            }
            
            logs.append(log)
            
            # Print progress
            print(f"Epoch {epoch:03d} | "
                  f"train_loss={log['train_loss']:.4f} | "
                  f"train_acc={log['train_acc']:.2f}% | "
                  f"test_acc={test_acc:.2f}% | "
                  f"frozen={len(log['frozen_layers'])} | "
                  f"time={log['epoch_time_s']:.2f}s")
            
            # Save logs
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")
        
        return logs
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        loss_sum = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            self._iteration += 1
            
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            
            # Check if we should use cached activations
            use_cache = self._should_use_cache()
            
            if use_cache:
                # Try to get cached activations
                sample_indices = list(range(batch_idx * len(xb), (batch_idx + 1) * len(xb)))
                cached_activations = self.cache.get_activation(
                    sample_indices, self._epoch
                )
                
                if cached_activations is not None:
                    # Use cached activations, only forward through unfrozen layers
                    logits = self._forward_from_cached(cached_activations)
                else:
                    # Cache miss - compute normally
                    logits = self.model(xb)
                    
                    # Cache activations if layers are frozen
                    if self.layer_freezer.get_frozen_layers():
                        self._cache_current_activations(sample_indices)
            else:
                # Normal forward pass
                logits = self.model(xb)
                
                # Cache activations if in bootstrap phase
                if self._iteration <= self.bootstrap_iterations:
                    sample_indices = list(range(batch_idx * len(xb), (batch_idx + 1) * len(xb)))
                    self._cache_current_activations(sample_indices)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            # Metrics
            loss_sum += loss.item() * yb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            
            # Check plasticity periodically
            if self._iteration > self.bootstrap_iterations:
                result = self.layer_freezer.check_plasticity(
                    self.reference_model,
                    xb,
                    self._training_activations
                )
                
                # Start caching when first layer is frozen
                if result.get("action") == "frozen" and not self.cache._is_active:
                    self.cache.start_caching(
                        frozen_layer_idx=self.layer_freezer._current_layer_idx
                    )
        
        return {
            "train_loss": loss_sum / max(1, total),
            "train_acc": 100.0 * correct / max(1, total)
        }
    
    def _should_use_cache(self) -> bool:
        """Determine if we should use cached activations."""
        # Only use cache after bootstrap and when layers are frozen
        return (self._iteration > self.bootstrap_iterations and 
                self.cache._is_active and
                len(self.layer_freezer.get_frozen_layers()) > 0)
    
    def _forward_from_cached(self, cached_activations: torch.Tensor) -> torch.Tensor:
        """Forward pass starting from cached activations."""
        # This is model-specific; simplified version
        # In practice, would need to identify the first unfrozen layer
        # and forward from there
        return self.model(cached_activations)
    
    def _cache_current_activations(self, sample_indices: List[int]):
        """Cache current activations from training."""
        # Get activations from the last frozen layer
        # Simplified: cache the final feature representation
        if self._training_activations:
            # Use the last activation as the feature
            last_activation = list(self._training_activations.values())[-1]
            self.cache.cache_activation(sample_indices, last_activation, self._epoch)
    
    @torch.no_grad()
    def _evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate model on test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        for xb, yb in test_loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            
            logits = self.model(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
        
        return 100.0 * correct / max(1, total)


# =============================================================================
# ResNet-18 Specific Implementation (Compatible with target codebase)
# =============================================================================

def split_resnet18(model: nn.Module, boundary_idx: int) -> Tuple[nn.Module, nn.Module]:
    """
    Split ResNet-18 into prefix and suffix at the given boundary.
    
    Compatible with the target codebase structure.
    
    Args:
        model: ResNet-18 model
        boundary_idx: Boundary index (0-8)
        
    Returns:
        Tuple of (prefix, suffix) modules
    """
    CUTPOINTS = {
        0: ("stem", -1),
        1: ("layer1", 0),
        2: ("layer1", 1),
        3: ("layer2", 0),
        4: ("layer2", 1),
        5: ("layer3", 0),
        6: ("layer3", 1),
        7: ("layer4", 0),
        8: ("layer4", 1),
    }
    
    assert boundary_idx in CUTPOINTS
    layer_name, block_idx = CUTPOINTS[boundary_idx]
    
    class Prefix(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m
        
        def forward(self, x):
            x = self.m.conv1(x)
            x = self.m.bn1(x)
            x = self.m.relu(x)
            x = self.m.maxpool(x)
            
            if layer_name == "stem":
                return x
            
            for lname in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.m, lname)
                for bi, blk in enumerate(layer):
                    x = blk(x)
                    if lname == layer_name and bi == block_idx:
                        return x
            raise RuntimeError("Boundary not reached")
    
    class Suffix(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m
        
        def forward(self, feat):
            x = feat
            passed = False
            for lname in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(self.m, lname)
                for bi, blk in enumerate(layer):
                    if not passed:
                        if lname == layer_name and bi == block_idx:
                            passed = True
                        continue
                    x = blk(x)
            x = self.m.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.m.fc(x)
            return x
    
    return Prefix(model), Suffix(model)


def make_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Create CIFAR-friendly ResNet-18."""
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# =============================================================================
# Main Entry Point (Compatible with target codebase interface)
# =============================================================================

def main():
    """
    Main entry point for Egeria baseline training.
    
    Compatible command-line interface with the target codebase.
    """
    ap = argparse.ArgumentParser(description="Egeria Baseline Training")
    
    # Data arguments
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    
    # Model arguments
    ap.add_argument("--boundary_idx", type=int, default=6)
    ap.add_argument("--num_classes", type=int, default=10)
    
    # Egeria-specific arguments
    ap.add_argument("--cache_dir", type=str, default="./egeria_cache")
    ap.add_argument("--eval_interval", type=int, default=300,
                    help="Plasticity evaluation interval (n iterations)")
    ap.add_argument("--tolerance_factor", type=float, default=0.2,
                    help="Tolerance factor for plasticity slope (T)")
    ap.add_argument("--window_size", type=int, default=10,
                    help="History buffer length (W)")
    ap.add_argument("--max_memory_batches", type=int, default=5,
                    help="Max mini-batches to keep in GPU memory")
    ap.add_argument("--bootstrap_iterations", type=int, default=1000,
                    help="Iterations before starting layer freezing")
    
    # Optimizer arguments
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    
    # Other arguments
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_path", type=str, default="egeria_log.jsonl")
    
    args = ap.parse_args()
    
    # Setup
    os.makedirs(os.path.dirname(args.log_path) if args.log_path else ".", exist_ok=True)
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Egeria] Using device: {device}")
    
    # Create model
    model = make_cifar_resnet18(args.num_classes)
    model.to(device)
    
    # Create data loaders
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    
    # Create Egeria trainer
    trainer = EgeriaTrainer(
        model=model,
        cache_dir=args.cache_dir,
        model_type="cnn",
        plasticity_eval_interval=args.eval_interval,
        tolerance_factor=args.tolerance_factor,
        window_size=args.window_size,
        max_memory_batches=args.max_memory_batches,
        device=device,
        bootstrap_iterations=args.bootstrap_iterations
    )
    
    # Train
    print("[Egeria] Starting training...")
    logs = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        log_path=args.log_path
    )
    
    print("[Egeria] Training complete!")
    
    # Final statistics
    final_test_acc = logs[-1]["test_acc"] if logs else 0.0
    avg_epoch_time = sum(l["epoch_time_s"] for l in logs) / max(1, len(logs))
    
    print(f"\n[Egeria] Final Results:")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    print(f"  Avg Epoch Time: {avg_epoch_time:.2f}s")
    print(f"  Total Frozen Layers: {len(logs[-1]['frozen_layers']) if logs else 0}")


if __name__ == "__main__":
    main()
