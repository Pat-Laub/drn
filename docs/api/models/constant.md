# Constant - Constant Prediction Model

Simple baseline model that predicts constant distributions for benchmarking and ablation studies.

---

## Class Definition

::: drn.models.constant.Constant
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      show_bases: true

## Overview

The Constant model provides:
- **Simple baselines** - Predict same distribution for all inputs
- **Benchmarking** - Performance comparison baseline
- **Ablation studies** - Control model for feature importance
- **Sanity checks** - Verify that other models add value

## Key Features
- **Distribution fitting** - Fits single distribution to all training data
- **Fast inference** - No feature processing required
- **Multiple distributions** - Supports Gaussian, Gamma, etc.
- **Debugging tool** - Helps identify modeling issues

## Quick Example

```python
from drn.models import Constant

# Fit constant Gamma distribution
constant_model = Constant(distribution='gamma')
constant_model.fit(X_train, y_train)

# All predictions are identical
pred1 = constant_model.predict(X_test[:1])
pred2 = constant_model.predict(X_test[100:101])

# Same distribution parameters
assert torch.allclose(pred1.mean, pred2.mean)
assert torch.allclose(pred1.variance, pred2.variance)
```

## Use Cases

### Benchmarking
```python
from drn.metrics import rmse, crps

# Compare against constant baseline
constant_baseline = Constant('gamma')
constant_baseline.fit(X_train, y_train)

your_model = SomeModel()
your_model.fit(X_train, y_train)

# Evaluate both
constant_pred = constant_baseline.predict(X_test)
your_pred = your_model.predict(X_test)

constant_rmse = rmse(y_test, constant_pred.mean)
your_rmse = rmse(y_test, your_pred.mean)

print(f"Constant baseline RMSE: {constant_rmse:.4f}")
print(f"Your model RMSE: {your_rmse:.4f}")
print(f"Improvement: {((constant_rmse - your_rmse) / constant_rmse * 100):.1f}%")
```

## Performance Characteristics

The Constant model excels in simplicity and speed:

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Training Speed** | ⭐⭐⭐⭐⭐ | Instant - just fits distribution to data |
| **Memory Usage** | ⭐⭐⭐⭐⭐ | Minimal - stores only distribution parameters |
| **Flexibility** | ⭐ | None - same prediction for all inputs |
| **Interpretability** | ⭐⭐⭐⭐⭐ | Perfect - single fitted distribution |
| **Stability** | ⭐⭐⭐⭐⭐ | Maximum - no complexity to cause instability |

## Supported Distributions

The Constant model supports various distributions, but can be easily extended:

### Continuous Distributions
```python
# Gaussian for unbounded data
constant_gaussian = Constant('gaussian')

# Gamma for positive data  
constant_gamma = Constant('gamma')

# Log-normal for multiplicative processes
constant_lognormal = Constant('lognormal')

# Inverse Gaussian for right-skewed positive data
constant_ig = Constant('inversegaussian')
```
