# BaseModel

The abstract foundation for all distributional regression models in DRN. Provides unified interface, PyTorch Lightning integration, and common functionality.

---

## Class Definition

::: drn.models.base.BaseModel
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      show_bases: true
      show_inheritance_diagram: true

---

## Overview

`BaseModel` is the abstract base class that all DRN models inherit from. It provides:

- **Unified Interface** - Consistent `.fit()`, `.predict()`, and `.quantiles()` methods
- **PyTorch Lightning Integration** - Built-in training loops, early stopping, checkpointing
- **Automatic Data Handling** - Conversion between pandas/numpy and PyTorch tensors
- **GPU Support** - Automatic device detection and tensor placement
- **Distribution Interface** - Common methods for working with distributional predictions

## Key Methods

### Training Methods

#### `fit(X_train, y_train, X_val=None, y_val=None, **kwargs)`
Main training method that handles the complete training workflow:

- **Data preprocessing** and tensor conversion
- **DataLoader creation** with batching and shuffling  
- **Validation setup** with early stopping (if validation data provided)
- **Model checkpointing** and best weight restoration
- **GPU acceleration** when available

**Parameters:**
- `X_train, y_train` - Training features and targets (pandas/numpy/tensor)
- `X_val, y_val` - Optional validation data for early stopping
- `batch_size` - Training batch size (default: 128)
- `epochs` - Maximum training epochs (default: 10)
- `patience` - Early stopping patience (default: 5)
- `**trainer_kwargs` - Additional PyTorch Lightning trainer arguments

### Prediction Methods

#### `predict(x_raw)`
Creates distributional predictions from input features:

- **Automatic preprocessing** - Handles pandas/numpy to tensor conversion
- **Returns distribution objects** - With `.mean`, `.quantiles()`, `.cdf()`, etc.
- **Consistent interface** - Same API across all model types

#### `quantiles(x, percentiles, l=None, u=None, **kwargs)`
Unified quantile calculation supporting all distribution types:

- **Multiple percentiles** - Calculate several quantiles at once
- **Bounds specification** - Optional lower/upper bounds for search
- **Automatic fallback** - Uses binary search when analytic quantiles unavailable

### Abstract Methods

Subclasses must implement:

#### `loss(x, y) -> torch.Tensor`
Define the loss function for model training.

#### `_predict(x) -> Distribution`  
Core prediction logic returning PyTorch distribution objects.

---

## Usage Examples

### Basic Model Implementation

```python
from drn.models.base import BaseModel
import torch
import torch.nn as nn

class CustomModel(BaseModel):
    def __init__(self, input_dim, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Define architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # mean and log_std
        )
    
    def loss(self, x, y):
        """Implement loss function."""
        pred = self.network(x)
        mean, log_std = pred[:, 0], pred[:, 1]
        std = torch.exp(log_std)
        
        # Gaussian negative log-likelihood
        nll = 0.5 * torch.log(2 * torch.pi * std**2) + \
              0.5 * ((y - mean) / std)**2
        return nll.mean()
    
    def _predict(self, x):
        """Implement prediction logic."""
        pred = self.network(x)
        mean, log_std = pred[:, 0], pred[:, 1]
        std = torch.exp(log_std)
        
        return torch.distributions.Normal(mean, std)

# Usage
model = CustomModel(input_dim=5)
model.fit(X_train, y_train, X_val, y_val, epochs=100)

predictions = model.predict(X_test)
mean_pred = predictions.mean
quantiles = model.quantiles(X_test, [10, 50, 90])
```

### Training Configuration

```python
# Basic training
model.fit(X_train, y_train)

# With validation and early stopping
model.fit(X_train, y_train, X_val, y_val, patience=20)

# Custom training parameters
model.fit(
    X_train, y_train,
    batch_size=256,
    epochs=200,
    patience=30,
    
    # PyTorch Lightning trainer options
    accelerator='gpu',
    devices=1,
    max_epochs=200,
    enable_progress_bar=True
)
```

### Working with Predictions

```python
# Get distribution object
dist = model.predict(X_test)

# Point estimates  
mean_pred = dist.mean
mode_pred = dist.mode  # if available

# Distributional properties
pdf_vals = dist.density(y_grid)       # PDF evaluation
cdf_vals = dist.cdf(y_grid)          # CDF evaluation  
log_prob = dist.log_prob(y_true)     # Log-likelihood
samples = dist.sample((1000,))       # Generate samples

# Quantiles using model method
quantiles = model.quantiles(X_test, [5, 25, 50, 75, 95])

# Or using distribution method (if available)
quantiles = dist.quantiles([5, 25, 50, 75, 95], l=-10, u=100)
```

---

## Implementation Details

### Data Preprocessing

The `preprocess()` method handles:
- **Tensor conversion** - pandas/numpy → PyTorch tensors
- **Device placement** - Automatic GPU/CPU handling
- **Column transformation** - Applied when model has `.ct` attribute
- **Type consistency** - Ensures float32 tensors

### Training Infrastructure

Built on PyTorch Lightning for robust training:
- **Automatic optimization** - Adam optimizer with configurable learning rate
- **Logging integration** - Training/validation loss tracking
- **Checkpointing** - Best model weight preservation
- **Early stopping** - Validation-based with configurable patience
- **Progress tracking** - Built-in progress bars

### Distribution Interface

All models return PyTorch distribution objects supporting:
- `.mean` - Expected value
- `.mode` - Most likely value (when defined)
- `.variance` - Variance (when defined)
- `.cdf(value)` - Cumulative probability
- `.log_prob(value)` - Log-likelihood
- `.sample(shape)` - Random sampling
- `.quantiles(percentiles)` - Quantile calculation (custom implementation)

---

## Advanced Features

### Custom Preprocessing

```python
class PreprocessedModel(BaseModel):
    def __init__(self, preprocessor):
        super().__init__()
        self.ct = preprocessor  # Will be applied in preprocess()
    
    def preprocess(self, x, targets=False):
        """Custom preprocessing logic."""
        if not targets and hasattr(self, 'ct'):
            x = self.ct.transform(x)
        return super().preprocess(x, targets)
```

### GPU Optimization

```python
# Automatic GPU detection
if torch.cuda.is_available():
    model = model.cuda()

# GPU memory optimization
model.fit(
    X_train, y_train,
    batch_size=64,  # Reduce if memory issues
    accelerator='gpu',
    precision=16    # Mixed precision training
)
```

### Custom Training Loops

```python
# Access Lightning trainer directly
trainer_kwargs = {
    'max_epochs': 100,
    'accelerator': 'gpu',
    'devices': 2,  # Multi-GPU
    'precision': 16,
    'gradient_clip_val': 1.0
}

model.fit(X_train, y_train, **trainer_kwargs)
```

---

## See Also

- **[GLM](glm.md)** - Generalized Linear Models implementation
- **[DRN](drn.md)** - Distributional Refinement Network
- **[Training](../training.md)** - Training functions and utilities
- **[Quick Start](../../getting-started/quickstart.md)** - Practical examples