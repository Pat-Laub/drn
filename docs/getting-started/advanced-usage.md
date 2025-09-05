# Advanced Usage Guide

This guide covers advanced DRN usage for users who want direct control over PyTorch tensors and the training process. Use this when you need custom training loops or integration with existing PyTorch codebases.

## When to Use Advanced Mode

Use advanced mode when you need:

- **Custom training loops** with specific optimization strategies
- **Manual tensor management** for performance optimization
- **Integration with existing PyTorch codebases** 
- **Fine-grained control** over model training
- **Custom loss functions** beyond the provided ones

## Core Concepts

### Tensor-First Workflow

```python
import torch
from drn import GLM, DRN, train
import numpy as np

# Manual tensor conversion (based on test patterns)
def generate_tensor_data(n=1000, seed=1):
    """Generate tensor data similar to test_fit_models_synthetic.py"""
    rng = np.random.default_rng(seed)
    x_all = rng.random(size=(n, 4))
    epsilon = rng.normal(0, 0.2, n)

    means = np.exp(
        0
        - 0.5 * x_all[:, 0]
        + 0.5 * x_all[:, 1]
        + np.sin(np.pi * x_all[:, 0])
        - np.sin(np.pi * np.log(x_all[:, 2] + 1))
        + np.cos(x_all[:, 1] * x_all[:, 2])
    ) + np.cos(x_all[:, 1])
    
    y_all = means + epsilon**2

    # Convert to tensors
    X_tensor = torch.tensor(x_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)
    
    return X_tensor, y_tensor

# Generate tensor data
X_train, y_train = generate_tensor_data(800, seed=1)
X_val, y_val = generate_tensor_data(200, seed=2)

print(f"X_train shape: {X_train.shape}")
print(f"y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
```

### Using the `train` Function Directly

```python
# Create PyTorch datasets (as in tests)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

# Train GLM using the train function (from test_fit_models_synthetic.py)
torch.manual_seed(1)
glm = GLM("gamma")
train(glm, train_dataset, val_dataset, epochs=5)

print("✓ GLM training completed using train() function")
```

### Manual Dispersion Updates

```python
# Update dispersion after training (from tests)
glm.update_dispersion(X_train, y_train)
print(f"Updated dispersion: {glm.dispersion.item():.4f}")
```

## DRN Training with Custom Parameters

Based on test patterns, here's how to train DRN with manual control:

```python
from drn.models import drn_cutpoints

# Create cutpoints (pattern from tests)
cutpoints = drn_cutpoints(
    c_0=y_train.min().item() * 0.9,
    c_K=y_train.max().item() * 1.1,
    p=0.1,
    y=y_train.numpy(),
    min_obs=10
)

print(f"Generated {len(cutpoints)} cutpoints")

# Create DRN with baseline
torch.manual_seed(2)  # For reproducibility (as in tests)
drn = DRN(glm, cutpoints, num_hidden_layers=2, hidden_size=100)

# Train using the train function
train(drn, train_dataset, val_dataset, epochs=5, lr=0.001)

print("✓ DRN training completed")
```

## Model Evaluation with CRPS

From the test suite, here's how to properly evaluate models:

```python
from drn.metrics import crps

def evaluate_model_crps(model, X_test, y_test, grid_size=1000):
    """Evaluate model using CRPS (from test patterns)."""
    
    # Generate grid for CRPS calculation
    grid = torch.linspace(0, y_test.max().item() * 1.1, grid_size).unsqueeze(-1)
    
    # Get model predictions
    dists = model.predict(X_test)
    cdfs = dists.cdf(grid)
    
    # Calculate CRPS
    grid = grid.squeeze()
    crps_scores = crps(y_test, grid, cdfs)
    
    return crps_scores.mean()

# Evaluate both models
X_test, y_test = generate_tensor_data(200, seed=3)

glm_crps = evaluate_model_crps(glm, X_test, y_test)
drn_crps = evaluate_model_crps(drn, X_test, y_test)

print(f"GLM CRPS: {glm_crps:.4f}")
print(f"DRN CRPS: {drn_crps:.4f}")
print(f"CRPS improvement: {((glm_crps - drn_crps) / glm_crps * 100):.1f}%")
```

## CANN Model Training

Based on the test patterns for CANN:

```python
from drn.models import CANN

# Train CANN (from test_fit_models_synthetic.py pattern)
torch.manual_seed(2)
baseline_for_cann = GLM("gamma")
train(baseline_for_cann, train_dataset, val_dataset, epochs=2)

cann = CANN(baseline_for_cann, num_hidden_layers=2, hidden_size=100)
train(cann, train_dataset, val_dataset, epochs=2)

print("✓ CANN training completed")

# Evaluate CANN
cann_crps = evaluate_model_crps(cann, X_test, y_test)
print(f"CANN CRPS: {cann_crps:.4f}")
```

## Device Management

For GPU usage (based on test patterns):

```python
# Check device availability (from test patterns)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU: {device}")

# Move data to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

# Move model to device
glm = glm.to(device)

print(f"✓ Data and model moved to {device}")
```

## Working with Different Model Types

### GLM with Different Distributions

```python
# Test different GLM distributions (pattern from test_glm_distributions.py)
distributions = ['gaussian', 'gamma']

for dist_name in distributions:
    print(f"\nTraining GLM with {dist_name} distribution:")
    
    # Create and train model
    glm_dist = GLM(dist_name)
    train(glm_dist, train_dataset, val_dataset, epochs=3)
    
    # Evaluate
    crps_score = evaluate_model_crps(glm_dist, X_test, y_test)
    print(f"{dist_name} GLM CRPS: {crps_score:.4f}")
```

### Quantile Evaluation

Testing quantile functionality (from test patterns):

```python
from drn.utils import binary_search_icdf

# Test quantile calculation
test_percentiles = [10, 50, 90]
quantiles = glm.quantiles(X_test[:5], test_percentiles)

print(f"Quantiles shape: {quantiles.shape}")
print(f"Test percentiles: {test_percentiles}")
print(f"Quantile values for first sample: {quantiles[0]}")
```

## Model Checkpointing

Basic model saving/loading (minimal example):

```python
# Save model state
torch.save(glm.state_dict(), 'glm_model.pth')
torch.save(drn.state_dict(), 'drn_model.pth')

# Load model state
glm_loaded = GLM("gamma")
glm_loaded.load_state_dict(torch.load('glm_model.pth'))
glm_loaded.eval()

print("✓ Model checkpointing completed")
```

## Advanced Training Configuration

Using PyTorch Lightning trainer options (from test patterns):

```python
# Advanced training with specific parameters
train(
    drn,
    train_dataset,
    val_dataset,
    epochs=10,
    batch_size=64,
    lr=0.001,
    patience=5,
    # Additional trainer arguments
    accelerator='cpu',  # or 'gpu' if available
    devices=1,
    enable_progress_bar=True
)
```

## Performance Optimization

### Batch Size Optimization

```python
# Test different batch sizes for performance
batch_sizes = [32, 64, 128]

for batch_size in batch_sizes:
    print(f"Testing batch size: {batch_size}")
    
    # Create data loaders with different batch sizes
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Time training step
    import time
    model = GLM("gamma")
    start_time = time.time()
    train(model, train_dataset, val_dataset, epochs=1)
    training_time = time.time() - start_time
    
    print(f"Training time with batch_size {batch_size}: {training_time:.2f}s")
```

## Integration with Existing PyTorch Code

If you have existing PyTorch training loops:

```python
import torch.nn as nn
import torch.optim as optim

# Manual training loop (advanced users)
def custom_training_loop(model, train_dataset, epochs=5):
    """Custom training loop for advanced users."""
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.loss(data, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(train_loader):.4f}')
    
    model.eval()

# Use custom training loop
custom_model = GLM("gamma")
custom_training_loop(custom_model, train_dataset, epochs=3)
```

## Key Differences from Simple Usage

**Data Handling**

- Manual tensor conversion and device management
- Explicit DataLoader creation
- Direct control over batching and shuffling

**Training Control**

- Use `train()` function instead of `.fit()`
- Manual epoch and learning rate management
- Custom training loops possible

**Evaluation**

- Manual CRPS calculation with grid generation
- Direct access to distribution objects
- Custom metric implementation

## When to Use Each Approach

**Use Simple Usage (pandas/numpy) when:**

- Prototyping and experimenting
- Standard workflows are sufficient
- Working with mixed data types
- Want scikit-learn-like interface

**Use Advanced Usage (tensors) when:**

- Need custom training loops
- Integrating with existing PyTorch code
- Performance optimization required
- Custom loss functions needed

## Next Steps

- **[API Reference](../api/index.md)** - Complete technical documentation
- **[Training](../api/training.md)** - Training function details
- **[Quick Start](quickstart.md)** - Compare with pandas approach