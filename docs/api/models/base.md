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
