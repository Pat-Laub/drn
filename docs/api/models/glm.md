# GLM - Generalized Linear Models

Interpretable baseline models for distributional regression. Supports multiple distribution families with both statistical and neural network training approaches.

---

## Class Definition

::: drn.models.glm.GLM
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      show_bases: true

---

## Overview

The `GLM` class provides PyTorch implementations of Generalized Linear Models with distributional outputs. Key features:

- **Multiple Distribution Families** - Gaussian, Gamma, Inverse Gaussian, Log-Normal
- **Dual Training Modes** - Statistical (statsmodels) or neural (PyTorch gradient descent)
- **Interpretable Parameters** - Access to coefficients, intercepts, and dispersion
- **Seamless Integration** - Perfect baseline for DRN refinement
- **Automatic Parameter Estimation** - Maximum likelihood via statsmodels

## Supported Distributions

### Gaussian Distribution
```python
glm = GLM('gaussian')
```
- **Use Case**: Continuous data with constant variance
- **Link Function**: Identity (`μ = Xβ`)
- **Parameters**: Mean (μ), standard deviation (σ)
- **Best For**: Symmetric, unbounded data (temperatures, stock returns)

### Gamma Distribution  
```python
glm = GLM('gamma')
```
- **Use Case**: Positive continuous data with right skew
- **Link Function**: Log (`log(μ) = Xβ`)
- **Parameters**: Shape (α), scale (β)
- **Best For**: Insurance claims, waiting times, sales amounts

### Inverse Gaussian Distribution
```python
glm = GLM('inversegaussian')
```
- **Use Case**: Positive data with extreme right tail
- **Link Function**: Log (`log(μ) = Xβ`)
- **Parameters**: Mean (μ), dispersion (λ)
- **Best For**: First passage times, service durations

### Log-Normal Distribution
```python
glm = GLM('lognormal')
```
- **Use Case**: Positive data from multiplicative processes
- **Link Function**: Identity on log-scale
- **Parameters**: Log-mean (μ), log-std (σ)  
- **Best For**: Income distributions, growth rates, file sizes

---

## Quick Start

### Basic Usage
```python
from drn import GLM
import pandas as pd

# Load data
X = pd.DataFrame({'age': [25, 35, 45], 'income': [30000, 50000, 70000]})
y = pd.Series([1200, 1800, 2400])

# Fit GLM using statsmodels approach
glm = GLM('gamma')
glm.fit(X, y)

# Make predictions
predictions = glm.predict(X_test)
mean_pred = predictions.mean
quantiles = predictions.quantiles([10, 50, 90])
```

### Alternative: Gradient Descent Training
```python
# Neural network style training
glm = GLM('gaussian', learning_rate=0.001)
glm.fit(
    X_train, y_train,
    X_val, y_val,
    grad_descent=True,
    epochs=1000,
    batch_size=128,
    patience=50
)
```

---

## Detailed Examples

### Insurance Claims Modeling

```python
import pandas as pd
import numpy as np
from drn import GLM
from drn.metrics import rmse, crps

# Load insurance data
claims_data = pd.read_csv('insurance_claims.csv')
X = claims_data[['age', 'vehicle_age', 'region', 'policy_type']]
y = claims_data['claim_amount']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Gamma GLM (ideal for insurance claims)
glm_gamma = GLM('gamma')
glm_gamma.fit(X_train, y_train)

# Evaluate predictions
pred_dist = glm_gamma.predict(X_test)

# Point prediction accuracy
rmse_score = rmse(y_test, pred_dist.mean)
print(f"RMSE: ${rmse_score:.2f}")

# Distributional accuracy
grid = np.linspace(0, y_test.max() * 1.2, 1000)
crps_score = crps(y_test, grid, pred_dist.cdf(grid)).mean()
print(f"CRPS: ${crps_score:.2f}")

# Risk measures
percentiles = [50, 75, 90, 95, 99]
risk_measures = glm_gamma.quantiles(X_test, percentiles)

print("Risk Measures:")
for i, p in enumerate(percentiles):
    avg_risk = risk_measures[:, i].mean()
    print(f"{p}th percentile: ${avg_risk:.2f}")
```

---

## Parameter Access and Interpretation

### Extracting Fitted Parameters

```python
# After fitting
glm = GLM('gamma')
glm.fit(X_train, y_train)

# Access parameters
print("Model Parameters:")
print(f"Coefficients: {glm.linear.weight.data}")
print(f"Intercept: {glm.linear.bias.data}")
print(f"Dispersion: {glm.dispersion.data}")

# Interpret coefficients (for Gamma GLM with log link)
feature_names = X_train.columns
for i, name in enumerate(feature_names):
    coef = glm.linear.weight[0, i].item()
    effect = np.exp(coef)  # Multiplicative effect
    print(f"{name}: {coef:.4f} (multiplicative effect: {effect:.4f})")
```

### Statsmodels equivalence

Under the hood, `GLM` uses `statsmodels` for statistical fitting.
You can make the equivalent GLM directly to access detailed statistical outputs.

```python
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
import numpy as np

# For detailed statistical analysis, access underlying statsmodels results
X_np = X_train.values
y_np = y_train.values

# Fit using statsmodels directly for statistical inference
X_sm = sm.add_constant(X_np)
sm_model = sm.GLM(y_np, X_sm, family=Gamma(link=sm.families.links.Log()))
sm_results = sm_model.fit()

print(sm_results.summary())
print(f"AIC: {sm_results.aic:.2f}")
print(f"BIC: {sm_results.bic:.2f}")
```

---

## Advanced Features

### Integration with DRN

```python
from drn import GLM, DRN

# Train baseline GLM
baseline = GLM('gamma')
baseline.fit(X_train, y_train)

# Initialize DRN with GLM baseline
drn_model = DRN(baseline=baseline).fit(X_train, y_train)
```
