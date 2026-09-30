# AutoFreeze Baseline Implementation

## Paper Reference
**AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning**  
Authors: Yuhan Liu, Saurabh Agarwal, Shivaram Venkataraman  
Conference: ICLR 2021  
arXiv: [2102.01386](https://arxiv.org/abs/2102.01386)

## Implementation Overview

This implementation reproduces the gradient-norm based freezing mechanism from the AutoFreeze paper.

### Key Components

1. **Gradient Norm Test (Equation 1)**
   ```
   η_l = (||Δ_{T-1_l}|| - ||Δ_{T_l}||) / ||Δ_{T-1_l}||
   ```
   - Measures rate of change of gradient norms for each layer
   - Determines which layers have converged and can be frozen

2. **Layer Freezing Strategy (Algorithm 1)**
   - Ranks layers by η_l (gradient norm change rate)
   - Freezes layers in bottom-up order (earlier layers first)
   - Uses Nth percentile threshold (default: 50th percentile/median)

3. **Caching Mechanism**
   - Caches intermediate activations of frozen layers
   - Avoids recomputing forward passes in subsequent epochs
   - Significant speedup when combined with freezing

## Usage

### Basic Usage

```python
from baselines.autofreeze import AutoFreeze

# Initialize
af = AutoFreeze(
    model=model,
    optimizer=optimizer,
    eval_interval=5,      # Evaluate every 5 iterations
    percentile=50.0,      # Use 50th percentile (median)
    enable_caching=True,  # Enable caching
    verbose=True
)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # AutoFreeze step (after backward, before optimizer step)
        af.step()
        
        optimizer.step()
```

### Using AutoFreezeTrainer (Complete Training Wrapper)

```python
from baselines.autofreeze import AutoFreezeTrainer

trainer = AutoFreezeTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    num_epochs=200,
    lr=0.1,
    eval_interval=5,
    percentile=50.0,
    enable_caching=True,
    verbose=True
)

# Train
history = trainer.train()
```

### Integration with Existing Codebase

```python
from baselines.autofreeze import AutoFreezeAdapter

# Create adapter
adapter = AutoFreezeAdapter(
    model=model,
    optimizer=optimizer,
    eval_interval=5,
    percentile=50.0,
    verbose=True
)

# In training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # ... forward and backward ...
        loss.backward()
        
        # Call after backward
        adapter.after_backward(epoch=epoch)
        
        optimizer.step()
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | - | Neural network model |
| `optimizer` | Optimizer | - | PyTorch optimizer |
| `eval_interval` | int | 5 | Iterations between gradient norm evaluations |
| `percentile` | float | 50.0 | Percentile threshold for freezing (0-100) |
| `enable_caching` | bool | True | Enable intermediate activation caching |
| `cache_device` | str | "cpu" | Device for cache storage |
| `verbose` | bool | False | Print freezing decisions |

## Paper Results

From the AutoFreeze paper:
- **Single GPU**: Up to 2.55x speedup with caching enabled
- **64 GPU cluster**: Up to 4.38x speedup (performance mode)
- **Cost reduction**: Up to 5.03x (efficiency mode)
- **Accuracy**: Maintains accuracy within 0.5% of full fine-tuning

## Key Findings from Paper

1. **Bottom-up convergence**: Earlier layers converge faster than later layers
2. **Median percentile**: 50th percentile closely tracks ideal SVCCA-based freezing
3. **Gradient norm test**: Effective proxy for convergence without needing final weights
4. **Caching benefit**: Significant additional speedup when combined with freezing

## Implementation Details

### Gradient Norm Computation
- Accumulates L2 norms of gradients over evaluation interval
- Computes change rate between consecutive intervals
- Handles edge cases (division by zero, missing gradients)

### Freezing Strategy
- Only freezes layers in bottom-up order (enforced by algorithm)
- Updates optimizer to track only unfrozen parameters
- Maintains set of frozen and active layers

### Caching
- Builds cache when layers are frozen
- Stores intermediate activations on specified device
- Provides interface for retrieving cached features

## Files

- `/mnt/okcomputer/output/baselines/autofreeze.py` - Main implementation (860 lines)
- `/mnt/okcomputer/output/baselines/autofreeze_README.md` - This documentation

## Testing

```python
# Run basic tests
python -c "
import sys
sys.path.insert(0, '/mnt/okcomputer/output')
from baselines.autofreeze import AutoFreeze, AutoFreezeTrainer, AutoFreezeAdapter
print('All imports successful!')
"
```

## References

```bibtex
@article{liu2021autofreeze,
  title={AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning},
  author={Liu, Yuhan and Agarwal, Saurabh and Venkataraman, Shivaram},
  journal={arXiv preprint arXiv:2102.01386},
  year={2021}
}
```
