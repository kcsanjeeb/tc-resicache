"""
AutoFreeze Baseline Implementation
Paper: "AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning"
Authors: Yuhan Liu, Saurabh Agarwal, Shivaram Venkataraman
Conference: ICLR 2021 (arXiv:2102.01386)

This implementation reproduces the gradient-norm based freezing mechanism
described in the AutoFreeze paper.

Key Concepts:
1. Gradient Norm Test: Measures the rate of change of gradient norms for each layer
   to determine which layers have converged and can be frozen.
   
   Formula (Equation 1 in paper):
   η_l = (||Δ_{T-1_l}|| - ||Δ_{T_l}||) / ||Δ_{T-1_l}||
   
   Where:
   - Δ_{T_l} is the accumulated gradient for layer l at interval T
   - η_l is the gradient norm change rate for layer l

2. Layer Freezing Strategy: Layers are ranked by η_l and the slowest changing
   layers (bottom Nth percentile) are frozen. Layers must be frozen in order
   from bottom to top (earlier layers first).

3. Caching Mechanism: When layers are frozen, their intermediate outputs can
   be cached to avoid recomputing forward passes in subsequent epochs.

Usage:
    from baselines.autofreeze import AutoFreeze
    
    # Initialize AutoFreeze
    af = AutoFreeze(
        model=model,
        optimizer=optimizer,
        eval_interval=5,      # Evaluate every 5 iterations
        percentile=50,        # Use 50th percentile (median) for freezing
        enable_caching=True   # Enable intermediate activation caching
    )
    
    # During training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # ... forward pass ...
            loss.backward()
            
            # AutoFreeze decides which layers to freeze based on gradients
            af.step()  # Must be called after loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from collections import defaultdict
import time


class AutoFreeze:
    """
    AutoFreeze: Automatically freezing model blocks to accelerate fine-tuning.
    
    This class implements the gradient-norm based freezing mechanism from the
    AutoFreeze paper (ICLR 2021). It monitors gradient changes across layers
    and automatically freezes layers that have converged.
    
    Reference:
        Liu, Y., Agarwal, S., & Venkataraman, S. (2021).
        AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning.
        ICLR 2021. arXiv:2102.01386
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        eval_interval: int = 5,
        percentile: float = 50.0,
        enable_caching: bool = True,
        cache_device: str = "cpu",
        freeze_embedding: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize AutoFreeze.
        
        Args:
            model: The neural network model to be fine-tuned
            optimizer: The optimizer used for training
            eval_interval: Number of iterations between gradient norm evaluations
                          (default: 5, as used in paper)
            percentile: Percentile threshold for freezing (default: 50.0 for median)
                       Paper shows that 50th percentile closely tracks ideal SVCCA
            enable_caching: Whether to enable caching of intermediate activations
            cache_device: Device to store cache ("cpu" or "cuda")
            freeze_embedding: Whether to freeze embedding layer when encoder is frozen
            verbose: Whether to print freezing decisions
        """
        self.model = model
        self.optimizer = optimizer
        self.eval_interval = eval_interval
        self.percentile = percentile
        self.enable_caching = enable_caching
        self.cache_device = torch.device(cache_device)
        self.freeze_embedding = freeze_embedding
        self.verbose = verbose
        
        # State tracking
        self.iteration = 0
        self.epoch = 0
        self.frozen_layers: set = set()
        self.active_layers: List[str] = []
        
        # Gradient accumulation for gradient norm test
        # Stores accumulated gradients for current interval
        self.current_gradients: Dict[str, torch.Tensor] = {}
        # Stores accumulated gradients for previous interval
        self.previous_gradients: Dict[str, torch.Tensor] = {}
        
        # Cache for intermediate activations
        self.activation_cache: Dict[int, torch.Tensor] = {}
        self.label_cache: Optional[torch.Tensor] = None
        self.cache_built = False
        
        # Metrics tracking
        self.metrics = {
            'frozen_layers_history': [],
            'gradient_norms_history': [],
            'freeze_decisions': [],
        }
        
        # Initialize layer tracking
        self._initialize_layer_tracking()
        
        if self.verbose:
            print(f"[AutoFreeze] Initialized with eval_interval={eval_interval}, "
                  f"percentile={percentile}, caching={enable_caching}")
    
    def _initialize_layer_tracking(self):
        """
        Initialize tracking for all trainable layers in the model.
        
        Identifies all layers with parameters that require gradients.
        """
        self.layer_names = []
        for name, module in self.model.named_modules():
            # Skip the root module and modules without parameters
            if name == '' or len(list(module.parameters())) == 0:
                continue
            # Check if any parameter requires grad
            has_trainable = any(p.requires_grad for p in module.parameters())
            if has_trainable:
                self.layer_names.append(name)
        
        self.active_layers = self.layer_names.copy()
        
        if self.verbose:
            print(f"[AutoFreeze] Tracking {len(self.layer_names)} layers")
    
    def _get_layer_gradient_norm(self, layer_name: str) -> float:
        """
        Compute the L2 norm of accumulated gradients for a layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            L2 norm of accumulated gradients
        """
        total_norm = 0.0
        found = False
        
        for name, param in self.model.named_parameters():
            if name.startswith(layer_name) and param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                found = True
        
        if not found:
            return 0.0
        
        return total_norm ** 0.5
    
    def _accumulate_gradients(self):
        """
        Accumulate gradient norms for all active layers.
        
        This accumulates gradients over the evaluation interval.
        """
        for layer_name in self.active_layers:
            grad_norm = self._get_layer_gradient_norm(layer_name)
            
            if layer_name not in self.current_gradients:
                self.current_gradients[layer_name] = 0.0
            
            self.current_gradients[layer_name] += grad_norm
    
    def _compute_gradient_norm_change(self, layer_name: str) -> float:
        """
        Compute the gradient norm change rate (η_l) for a layer.
        
        Implements Equation 1 from the paper:
        η_l = (||Δ_{T-1_l}|| - ||Δ_{T_l}||) / ||Δ_{T-1_l}||
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Gradient norm change rate η_l
        """
        if layer_name not in self.previous_gradients:
            return float('inf')  # No previous data, don't freeze
        
        prev_norm = self.previous_gradients[layer_name]
        curr_norm = self.current_gradients.get(layer_name, 0.0)
        
        if prev_norm == 0:
            return float('inf')  # Avoid division by zero
        
        # η_l = (||Δ_{T-1_l}|| - ||Δ_{T_l}||) / ||Δ_{T-1_l}||
        eta = (prev_norm - curr_norm) / prev_norm
        return eta
    
    def _gradient_norm_test(self) -> List[str]:
        """
        Perform the gradient norm test to determine which layers to freeze.
        
        Implements Algorithm 1 from the paper:
        1. Compute η_l for each active layer
        2. Rank layers by η_l
        3. Freeze layers where η_l is below the Nth percentile
        4. Only freeze layers in order (bottom-up)
        
        Returns:
            List of layer names to freeze
        """
        if len(self.active_layers) == 0:
            return []
        
        # Compute η_l for each active layer
        eta_values = {}
        for layer_name in self.active_layers:
            eta_values[layer_name] = self._compute_gradient_norm_change(layer_name)
        
        # Store for metrics
        self.metrics['gradient_norms_history'].append({
            'iteration': self.iteration,
            'eta_values': eta_values.copy()
        })
        
        # Compute percentile threshold
        eta_list = list(eta_values.values())
        if len(eta_list) == 0:
            return []
        
        threshold = np.percentile(eta_list, self.percentile)
        
        # Determine which layers to freeze (η_l < threshold)
        # Must freeze in order (bottom-up)
        layers_to_freeze = []
        
        # Sort layers by their position in the model (bottom to top)
        sorted_layers = sorted(
            self.active_layers,
            key=lambda x: self.layer_names.index(x) if x in self.layer_names else float('inf')
        )
        
        for layer_name in sorted_layers:
            if eta_values[layer_name] < threshold:
                layers_to_freeze.append(layer_name)
            else:
                # Stop at first layer that doesn't meet criterion
                # (must freeze in order)
                break
        
        return layers_to_freeze
    
    def _freeze_layers(self, layer_names: List[str]):
        """
        Freeze the specified layers.
        
        Args:
            layer_names: List of layer names to freeze
        """
        for layer_name in layer_names:
            if layer_name in self.frozen_layers:
                continue
            
            # Find the module
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name:
                    module = mod
                    break
            
            if module is not None:
                # Freeze all parameters in this module
                for param in module.parameters():
                    param.requires_grad = False
                
                self.frozen_layers.add(layer_name)
                if layer_name in self.active_layers:
                    self.active_layers.remove(layer_name)
                
                if self.verbose:
                    print(f"[AutoFreeze] Frozen layer: {layer_name}")
        
        # Update optimizer to only track unfrozen parameters
        self._update_optimizer()
    
    def _update_optimizer(self):
        """
        Update optimizer to only track unfrozen parameters.
        
        This is necessary because PyTorch optimizers need to know
        which parameters to update.
        """
        # Get all trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Update optimizer parameter groups
        self.optimizer.param_groups.clear()
        self.optimizer.state.clear()
        self.optimizer.add_param_group({'params': trainable_params})
    
    def _build_cache(self, dataloader: torch.utils.data.DataLoader, 
                     device: torch.device):
        """
        Build cache of intermediate activations for frozen layers.
        
        This implements the caching mechanism described in Section 3.2
        of the paper. When layers are frozen, their outputs can be
        cached to avoid recomputing forward passes.
        
        Args:
            dataloader: DataLoader for the dataset
            device: Device to use for computation
        """
        if not self.enable_caching or len(self.frozen_layers) == 0:
            return
        
        self.model.eval()
        
        # Find the boundary layer (last frozen layer)
        frozen_layer_names = sorted(
            self.frozen_layers,
            key=lambda x: self.layer_names.index(x) if x in self.layer_names else float('inf')
        )
        
        if len(frozen_layer_names) == 0:
            return
        
        # Build cache
        all_feats = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                
                # Forward pass up to frozen boundary
                feats = self._forward_to_boundary(inputs, frozen_layer_names[-1])
                
                all_feats.append(feats.cpu())
                all_labels.append(labels)
        
        # Store cache
        self.activation_cache = {
            i: feat for i, feat in enumerate(torch.cat(all_feats, dim=0))
        }
        self.label_cache = torch.cat(all_labels, dim=0)
        self.cache_built = True
        
        if self.verbose:
            print(f"[AutoFreeze] Cache built with {len(self.activation_cache)} entries")
    
    def _forward_to_boundary(self, inputs: torch.Tensor, 
                             boundary_layer: str) -> torch.Tensor:
        """
        Forward pass up to the specified boundary layer.
        
        Args:
            inputs: Input tensor
            boundary_layer: Name of the boundary layer
            
        Returns:
            Output tensor at boundary layer
        """
        x = inputs
        
        for name, module in self.model.named_modules():
            if name == '':
                continue
            
            # Check if this module is before or at boundary
            if name in self.layer_names:
                layer_idx = self.layer_names.index(name)
                boundary_idx = self.layer_names.index(boundary_layer)
                
                if layer_idx <= boundary_idx:
                    x = module(x)
        
        return x
    
    def step(self, dataloader: Optional[torch.utils.data.DataLoader] = None,
             device: Optional[torch.device] = None):
        """
        Perform one step of AutoFreeze.
        
        This should be called after loss.backward() in the training loop.
        It accumulates gradients and periodically evaluates whether to freeze layers.
        
        Args:
            dataloader: DataLoader for building cache (required if caching enabled)
            device: Device for cache building (required if caching enabled)
        """
        self.iteration += 1
        
        # Accumulate gradients
        self._accumulate_gradients()
        
        # Check if it's time to evaluate
        if self.iteration % self.eval_interval == 0:
            # Perform gradient norm test
            layers_to_freeze = self._gradient_norm_test()
            
            if len(layers_to_freeze) > 0:
                # Freeze identified layers
                self._freeze_layers(layers_to_freeze)
                
                # Record decision
                self.metrics['freeze_decisions'].append({
                    'iteration': self.iteration,
                    'frozen_layers': layers_to_freeze.copy(),
                    'num_frozen': len(self.frozen_layers)
                })
                
                # Rebuild cache if caching is enabled
                if self.enable_caching and dataloader is not None and device is not None:
                    self._build_cache(dataloader, device)
            
            # Shift gradients: current -> previous, reset current
            self.previous_gradients = self.current_gradients.copy()
            self.current_gradients = {}
    
    def get_cached_features(self, indices: List[int]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached features for given indices.
        
        Args:
            indices: List of data indices
            
        Returns:
            Tuple of (features, labels) or None if cache not available
        """
        if not self.cache_built or len(self.activation_cache) == 0:
            return None
        
        feats = []
        labels = []
        
        for idx in indices:
            if idx in self.activation_cache:
                feats.append(self.activation_cache[idx])
                labels.append(self.label_cache[idx])
        
        if len(feats) == 0:
            return None
        
        return torch.stack(feats), torch.stack(labels)
    
    def get_frozen_layers(self) -> List[str]:
        """
        Get list of currently frozen layers.
        
        Returns:
            List of frozen layer names
        """
        return list(self.frozen_layers)
    
    def get_active_layers(self) -> List[str]:
        """
        Get list of currently active (unfrozen) layers.
        
        Returns:
            List of active layer names
        """
        return self.active_layers.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics collected during training.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'total_iterations': self.iteration,
            'num_frozen_layers': len(self.frozen_layers),
            'frozen_layers': list(self.frozen_layers),
            'frozen_layers_history': self.metrics['frozen_layers_history'],
            'freeze_decisions': self.metrics['freeze_decisions'],
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            State dictionary
        """
        return {
            'iteration': self.iteration,
            'epoch': self.epoch,
            'frozen_layers': list(self.frozen_layers),
            'active_layers': self.active_layers,
            'previous_gradients': self.previous_gradients,
            'metrics': self.metrics,
            'cache_built': self.cache_built,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from checkpoint.
        
        Args:
            state_dict: State dictionary
        """
        self.iteration = state_dict['iteration']
        self.epoch = state_dict['epoch']
        self.frozen_layers = set(state_dict['frozen_layers'])
        self.active_layers = state_dict['active_layers']
        self.previous_gradients = state_dict['previous_gradients']
        self.metrics = state_dict['metrics']
        self.cache_built = state_dict.get('cache_built', False)


class AutoFreezeTrainer:
    """
    Complete training wrapper that integrates AutoFreeze into the training loop.
    
    This provides a higher-level interface for using AutoFreeze with standard
    PyTorch training loops.
    
    Example:
        trainer = AutoFreezeTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            eval_interval=5,
            percentile=50
        )
        
        for epoch in range(num_epochs):
            train_loss, train_acc = trainer.train_epoch(epoch)
            test_acc = trainer.evaluate()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_epochs: int = 200,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        eval_interval: int = 5,
        percentile: float = 50.0,
        enable_caching: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize AutoFreezeTrainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device for training
            num_epochs: Number of training epochs
            lr: Learning rate
            momentum: SGD momentum
            weight_decay: Weight decay
            eval_interval: Gradient norm evaluation interval
            percentile: Percentile threshold for freezing
            enable_caching: Whether to enable caching
            verbose: Whether to print progress
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.verbose = verbose
        
        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Initialize AutoFreeze
        self.autofreeze = AutoFreeze(
            model=model,
            optimizer=self.optimizer,
            eval_interval=eval_interval,
            percentile=percentile,
            enable_caching=enable_caching,
            verbose=verbose
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.history = []
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch with AutoFreeze.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        self.model.train()
        self.autofreeze.epoch = epoch
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # AutoFreeze step (gradient norm test and freezing)
            self.autofreeze.step(
                dataloader=self.train_loader,
                device=self.device
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        
        return train_loss, train_acc
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on test set.
        
        Returns:
            Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return 100.0 * correct / total
    
    def train(self) -> List[Dict[str, Any]]:
        """
        Complete training loop.
        
        Returns:
            Training history
        """
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            test_acc = self.evaluate()
            
            epoch_time = time.time() - start_time
            
            # Record history
            record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'epoch_time': epoch_time,
                'num_frozen_layers': len(self.autofreeze.get_frozen_layers()),
            }
            self.history.append(record)
            
            if self.verbose:
                frozen_info = f"frozen={record['num_frozen_layers']}"
                print(f"Epoch {epoch+1}/{self.num_epochs} | "
                      f"train_loss={train_loss:.4f} | "
                      f"train_acc={train_acc:.2f}% | "
                      f"test_acc={test_acc:.2f}% | "
                      f"{frozen_info} | "
                      f"time={epoch_time:.2f}s")
        
        return self.history
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all training metrics.
        
        Returns:
            Dictionary containing training history and AutoFreeze metrics
        """
        return {
            'history': self.history,
            'autofreeze_metrics': self.autofreeze.get_metrics(),
        }


def create_autofreeze_model(
    base_model: nn.Module,
    eval_interval: int = 5,
    percentile: float = 50.0,
    enable_caching: bool = True,
    verbose: bool = False,
) -> Tuple[nn.Module, AutoFreeze, torch.optim.Optimizer]:
    """
    Convenience function to create a model with AutoFreeze integration.
    
    Args:
        base_model: Base neural network model
        eval_interval: Gradient norm evaluation interval
        percentile: Percentile threshold for freezing
        enable_caching: Whether to enable caching
        verbose: Whether to print progress
        
    Returns:
        Tuple of (model, autofreeze, optimizer)
    """
    # Create optimizer
    optimizer = torch.optim.SGD(
        base_model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Create AutoFreeze
    autofreeze = AutoFreeze(
        model=base_model,
        optimizer=optimizer,
        eval_interval=eval_interval,
        percentile=percentile,
        enable_caching=enable_caching,
        verbose=verbose
    )
    
    return base_model, autofreeze, optimizer


# -----------------------------------------------------------------------------
# Compatibility layer for integration with existing codebase
# -----------------------------------------------------------------------------

class AutoFreezeAdapter:
    """
    Adapter class to integrate AutoFreeze with the existing RedSync codebase.
    
    This provides a compatible interface that matches the expected patterns
    in the main training script.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        eval_interval: int = 5,
        percentile: float = 50.0,
        enable_caching: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize AutoFreeze adapter.
        
        Args:
            model: Neural network model
            optimizer: PyTorch optimizer
            eval_interval: Gradient norm evaluation interval
            percentile: Percentile threshold for freezing
            enable_caching: Whether to enable caching
            verbose: Whether to print progress
        """
        self.autofreeze = AutoFreeze(
            model=model,
            optimizer=optimizer,
            eval_interval=eval_interval,
            percentile=percentile,
            enable_caching=enable_caching,
            verbose=verbose
        )
        self.model = model
        self.optimizer = optimizer
    
    def after_backward(self, epoch: int = 0):
        """
        Call this after loss.backward() in the training loop.
        
        Args:
            epoch: Current epoch number
        """
        self.autofreeze.epoch = epoch
        self.autofreeze.step()
    
    def get_frozen_count(self) -> int:
        """
        Get number of frozen layers.
        
        Returns:
            Number of frozen layers
        """
        return len(self.autofreeze.get_frozen_layers())
    
    def get_frozen_layers(self) -> List[str]:
        """
        Get list of frozen layer names.
        
        Returns:
            List of frozen layer names
        """
        return self.autofreeze.get_frozen_layers()


# Export main classes
__all__ = [
    'AutoFreeze',
    'AutoFreezeTrainer', 
    'AutoFreezeAdapter',
    'create_autofreeze_model',
]
