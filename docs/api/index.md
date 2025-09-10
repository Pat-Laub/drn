# API Reference

This section provides comprehensive documentation for all DRN classes, functions, and modules. The API documentation is automatically generated from the source code docstrings, similar to R package documentation.

## Overview

The DRN package is organized into several key modules:

- **[Models](models/index.md)** - Core distributional regression models (GLM, DRN, CANN, MDN, DDR)
- **[Distributions](distributions.md)** - Custom distribution implementations and utilities  
- **[Training](training.md)** - Training functions, loss functions, and optimization utilities
- **[Metrics](metrics.md)** - Evaluation metrics for distributional forecasting
- **[Interpretability](interpretability.md)** - Model explanation and visualization tools
- **[Utilities](utils.md)** - Data preprocessing, splitting, and helper functions

## Quick Reference

### Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| [`BaseModel`](models/base.md#drn.models.base.BaseModel) | Abstract base for all models | `.fit()`, `.predict()`, `.quantiles()` |
| [`GLM`](models/glm.md#drn.models.glm.GLM) | Generalized Linear Models | `.fit()`, `.predict()`, `.clone()` |
| [`DRN`](models/drn.md#drn.models.drn.DRN) | Distributional Refinement Network | `.fit()`, `.predict()`, `.log_adjustments()` |

### Key Functions

| Function | Purpose | Module |
|----------|---------|--------|
| [`train()`](training.md#drn.train.train) | Train models with PyTorch | `drn.train` |
| [`drn_loss()`](training.md#drn.models.drn_loss) | DRN loss function | `drn.models` |
| [`crps()`](metrics.md#drn.metrics.crps) | Continuous Ranked Probability Score | `drn.metrics` |
| [`split_and_preprocess()`](utils.md#drn.utils.split_and_preprocess) | Data preprocessing | `drn.utils` |

<!-- ### Import Patterns

```python
# Core functionality
from drn import GLM, DRN, train

# Specific modules
from drn.models import drn_loss, drn_cutpoints
from drn.metrics import crps, rmse, quantile_losses  
from drn.utils import split_and_preprocess
from drn.interpretability import DRNExplainer

# Advanced usage
import drn.models
import drn.distributions
``` -->

<!-- 
## Module Structure

```
drn/
├── models/                 # Core regression models
│   ├── base.py            # BaseModel abstract class
│   ├── glm.py             # Generalized Linear Models
│   ├── drn.py             # Distributional Refinement Network
│   ├── cann.py            # Combined Actuarial Neural Network
│   ├── mdn.py             # Mixture Density Network
│   ├── ddr.py             # Deep Distribution Regression
│   └── constant.py        # Constant prediction model
├── distributions/          # Distribution implementations
│   ├── histogram.py       # Histogram distribution
│   ├── extended_histogram.py  # Extended histogram
│   ├── inverse_gaussian.py   # Inverse Gaussian
│   └── estimation.py      # Parameter estimation utilities
├── train.py               # Training framework
├── metrics.py             # Evaluation metrics
├── interpretability.py    # Model interpretation
├── kernel_shap_explainer.py  # SHAP integration
└── utils.py               # Utility functions
```

## Getting Help

If you need additional help beyond this API documentation:

1. **Check examples** in each function's documentation
2. **Review tutorials** for step-by-step guidance
3. **Examine source code** on [GitHub](https://github.com/EricTianDong/drn)
4. **Open an issue** for bug reports or feature requests
5. **Contact maintainers** at [tiandong1999@gmail.com](mailto:tiandong1999@gmail.com)

## Contributing to Documentation

API documentation is generated from docstrings in the source code. To improve documentation:

1. **Follow NumPy docstring format** for consistency
2. **Include type hints** in function signatures  
3. **Add examples** showing typical usage
4. **Document edge cases** and error conditions
5. **Update docstrings** when changing functionality

Example of well-documented function:
```python
def example_function(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Brief description of what the function does.
    
    Longer description providing more context about the function's purpose,
    mathematical background, or implementation details.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (n_samples, n_features).
        Should contain numerical values only.
    threshold : float, default=0.5
        Threshold value for classification, must be between 0 and 1.
        
    Returns
    -------
    torch.Tensor
        Binary predictions of shape (n_samples,) with values in {0, 1}.
        
    Examples
    --------
    >>> import torch
    >>> x = torch.randn(100, 5)
    >>> result = example_function(x, threshold=0.7)
    >>> result.shape
    torch.Size([100])
    
    Notes
    -----
    This function applies threshold-based classification using the sigmoid
    function internally.
    
    References
    ----------
    .. [1] Smith, J. (2020). "Example Methods in Machine Learning"
    """
    # Implementation here
    pass
``` -->