# DeepGLM - Deep Generalized Linear Model

A neural network approach to distributional regression that combines the interpretable structure of GLMs with the flexibility of deep learning. DeepGLM learns nonlinear feature representations while maintaining distributional outputs.

---

## Class Definition

::: drn.models.deepglm.DeepGLM
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      show_bases: true

---

## Overview

The `DeepGLM` class extends traditional GLMs by learning nonlinear feature transformations through a feed-forward neural network before applying the GLM head. Key features:

- **Nonlinear Feature Learning** - Multi-layer neural network for complex patterns
- **Distributional Outputs** - Full probability distributions, not just point predictions
- **Multiple Distribution Families** - Gamma, Gaussian, and Inverse Gaussian support
- **End-to-End Training** - Unified loss function combining representation learning and GLM fitting
- **Flexible Architecture** - Configurable network depth and width
- **Post-hoc Dispersion Estimation** - Classical statistical estimation after neural training

## Supported Distributions

### Gamma Distribution
```python
deepglm = DeepGLM('gamma', num_hidden_layers=2, hidden_size=128)
```
- **Use Case**: Positive continuous data with right skew
- **Link Function**: Log (`log(μ) = η`)
- **Best For**: Insurance claims, sales amounts, service times
- **Output**: Gamma distribution with learned mean and estimated dispersion

### Gaussian Distribution  
```python
deepglm = DeepGLM('gaussian', num_hidden_layers=2, hidden_size=128)
```
- **Use Case**: Continuous data with complex nonlinear patterns
- **Link Function**: Identity (`μ = η`)
- **Best For**: Complex regression tasks with symmetric errors
- **Output**: Normal distribution with learned mean and estimated variance

### Inverse Gaussian Distribution
```python
deepglm = DeepGLM('inversegaussian', num_hidden_layers=2, hidden_size=128)
```
- **Use Case**: Positive data with extreme right tail and nonlinear relationships
- **Link Function**: Log (`log(μ) = η`)
- **Best For**: First passage times, duration modeling with complex covariates
- **Output**: Inverse Gaussian distribution with learned parameters

---

## Quick Start

### Basic Usage
```python
from drn import DeepGLM
import pandas as pd
import numpy as np

# Load data with complex nonlinear relationships
X = pd.DataFrame({
    'age': np.random.uniform(20, 80, 1000),
    'income': np.random.uniform(20000, 100000, 1000),
    'risk_score': np.random.uniform(0, 1, 1000)
})

# Create complex nonlinear target
y = np.exp(
    0.1 * X['age'] + 
    0.5 * np.log(X['income']) + 
    2.0 * X['risk_score']**2 +
    np.random.gamma(2, 0.5, 1000)
)

# Train DeepGLM
deepglm = DeepGLM(
    distribution='gamma',
    num_hidden_layers=3,
    hidden_size=64,
    dropout_rate=0.1,
    learning_rate=1e-3
)

deepglm.fit(X, y, epochs=100, batch_size=128)

# Make distributional predictions
predictions = deepglm.predict(X_test)
mean_pred = predictions.mean
percentiles = deepglm.quantiles(X_test, [10, 50, 90])
```

### Comparison with Traditional GLM

```python
from drn import GLM, DeepGLM
from drn.metrics import rmse, crps

# Train traditional GLM
traditional_glm = GLM('gamma')
traditional_glm.fit(X_train, y_train)

# Train DeepGLM
deep_glm = DeepGLM('gamma', num_hidden_layers=2, hidden_size=64)
deep_glm.fit(X_train, y_train, X_test, y_test, epochs=100)

# Compare predictions
glm_pred = traditional_glm.predict(X_test)
deep_pred = deep_glm.predict(X_test)

# Evaluation metrics
metrics = {}
for name, pred in [('GLM', glm_pred), ('DeepGLM', deep_pred)]:
    metrics[name] = {
        'rmse': rmse(y_test, pred.mean).item(),
        'nll': -pred.log_prob(torch.tensor(y_test.values)).mean().item()
    }

print("Model Comparison:")
print(f"{'Model':<10} {'RMSE':<10} {'NLL':<10}")
print("-" * 35)
for model, vals in metrics.items():
    print(f"{model:<10} {vals['rmse']:<10.3f} {vals['nll']:<10.3f}")
```
