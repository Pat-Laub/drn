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

### 🎯 Gaussian Distribution
```python
glm = GLM('gaussian')
```
- **Use Case**: Continuous data with constant variance
- **Link Function**: Identity (`μ = Xβ`)
- **Parameters**: Mean (μ), standard deviation (σ)
- **Best For**: Symmetric, unbounded data (temperatures, stock returns)

### 🎯 Gamma Distribution  
```python
glm = GLM('gamma')
```
- **Use Case**: Positive continuous data with right skew
- **Link Function**: Log (`log(μ) = Xβ`)
- **Parameters**: Shape (α), scale (β)
- **Best For**: Insurance claims, waiting times, sales amounts

### 🎯 Inverse Gaussian Distribution
```python
glm = GLM('inversegaussian')
```
- **Use Case**: Positive data with extreme right tail
- **Link Function**: Log (`log(μ) = Xβ`)
- **Parameters**: Mean (μ), dispersion (λ)
- **Best For**: First passage times, service durations

### 🎯 Log-Normal Distribution
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

# Fit GLM using maximum likelihood (recommended)
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

### Model Comparison Across Distributions

```python
from drn.metrics import rmse, crps

# Compare different distribution families
distributions = ['gaussian', 'gamma', 'lognormal']
results = {}

for dist in distributions:
    # Train model
    glm = GLM(dist)
    glm.fit(X_train, y_train)
    
    # Evaluate
    pred = glm.predict(X_test)
    rmse_val = rmse(y_test, pred.mean)
    
    # CRPS evaluation
    grid = np.linspace(y_test.min() - y_test.std(), 
                      y_test.max() + y_test.std(), 1000)
    crps_val = crps(y_test, grid, pred.cdf(grid)).mean()
    
    results[dist] = {
        'rmse': rmse_val.item(),
        'crps': crps_val.item(),
        'nll': -pred.log_prob(y_test).mean().item()
    }

# Print comparison
print("Distribution Comparison:")
print(f"{'Distribution':<15} {'RMSE':<10} {'CRPS':<10} {'NLL':<10}")
print("-" * 50)
for dist, metrics in results.items():
    print(f"{dist:<15} {metrics['rmse']:<10.3f} {metrics['crps']:<10.3f} {metrics['nll']:<10.3f}")
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

### Statistical Significance (Statsmodels Integration)

```python
import statsmodels.api as sm
import numpy as np

# For detailed statistical analysis, access underlying statsmodels results
X_np = X_train.values
y_np = y_train.values

# Fit using statsmodels directly for statistical inference
from statsmodels.genmod.families import Gamma
X_sm = sm.add_constant(X_np)
sm_model = sm.GLM(y_np, X_sm, family=Gamma(link=sm.families.links.Log()))
sm_results = sm_model.fit()

print(sm_results.summary())
print(f"AIC: {sm_results.aic:.2f}")
print(f"BIC: {sm_results.bic:.2f}")
```

---

## Advanced Features

### Custom Parameter Initialization

```python
# Manual parameter setting
glm = GLM('gamma')

# Set specific coefficients
glm.linear = torch.nn.Linear(X_train.shape[1], 1)
glm.linear.weight.data = torch.tensor([[0.5, -0.3, 0.8]])
glm.linear.bias.data = torch.tensor([2.1])
glm.dispersion.data = torch.tensor([1.5])

# Fine-tune with gradient descent
glm.fit(X_train, y_train, grad_descent=True, epochs=100)
```

### Model Cloning and Ensembles

```python
# Clone for ensemble methods
base_glm = GLM('gamma')
base_glm.fit(X_train, y_train)

# Create ensemble of slightly different models
ensemble = []
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    cloned_glm = base_glm.clone()
    
    # Add slight randomization
    noise = torch.randn_like(cloned_glm.linear.weight) * 0.01
    cloned_glm.linear.weight.data += noise
    
    ensemble.append(cloned_glm)

# Ensemble predictions
ensemble_preds = []
for model in ensemble:
    pred = model.predict(X_test)
    ensemble_preds.append(pred.mean)

# Average predictions
avg_pred = torch.stack(ensemble_preds).mean(dim=0)
```

### Integration with DRN

```python
from drn import DRN
from drn.models import drn_cutpoints, drn_loss
from drn import train

# Train baseline GLM
baseline = GLM('gamma')
baseline.fit(X_train, y_train)

# Create cutpoints for DRN refinement
cutpoints = drn_cutpoints(
    c_0=y_train.min() * 0.9,
    c_K=y_train.max() * 1.1, 
    p=0.1,
    y=y_train,
    min_obs=20
)

# Initialize DRN with GLM baseline
drn_model = DRN(
    baseline=baseline,
    cutpoints=cutpoints,
    hidden_size=128,
    num_hidden_layers=2
)

# Train DRN refinement
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train.values, dtype=torch.float32),
    torch.tensor(y_train.values, dtype=torch.float32)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_val.values, dtype=torch.float32),
    torch.tensor(y_val.values, dtype=torch.float32)
)

train(
    drn_model,
    lambda pred, y: drn_loss(pred, y, kl_alpha=1e-4, dv_alpha=1e-3),
    train_dataset,
    val_dataset,
    epochs=100
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Singular Matrix" Error
```
LinAlgError: singular matrix
```
**Cause**: Perfect multicollinearity in features  
**Solution**: Remove redundant features or add regularization
```python
# Check for multicollinearity
correlation_matrix = X_train.corr()
print("High correlations (> 0.9):")
high_corr = correlation_matrix[abs(correlation_matrix) > 0.9]
print(high_corr)

# Remove highly correlated features
X_train_reduced = X_train.drop(['redundant_column'], axis=1)
```

#### Issue: Poor Convergence with Gradient Descent
```
RuntimeError: loss not decreasing
```
**Solutions**:
```python
# Lower learning rate
glm = GLM('gamma', learning_rate=1e-4)

# Use statsmodels instead (more robust)
glm.fit(X_train, y_train, grad_descent=False)

# Add regularization
glm.fit(X_train, y_train, grad_descent=True, 
        weight_decay=1e-5)  # L2 regularization
```

#### Issue: Negative Predictions for Positive Distributions
**Cause**: Link function should ensure positivity  
**Check**: Verify log-link is working correctly
```python
# For gamma/inverse Gaussian, predictions should be positive
pred = glm.predict(X_test)
assert torch.all(pred.mean > 0), "Negative predictions detected!"

# Check link function implementation
print(f"Using log link: {glm.distribution in ['gamma', 'inversegaussian']}")
```

---

## Performance Optimization

### Large Dataset Handling

```python
# Use gradient descent for very large datasets
glm = GLM('gamma')
glm.fit(
    X_large, y_large,
    grad_descent=True,
    batch_size=512,    # Larger batches
    epochs=50,         # Fewer epochs
    patience=10
)
```

### Memory Optimization

```python
# Process data in chunks for memory efficiency
chunk_size = 10000
predictions = []

for i in range(0, len(X_test), chunk_size):
    chunk_X = X_test.iloc[i:i+chunk_size]
    chunk_pred = glm.predict(chunk_X)
    predictions.append(chunk_pred.mean)

# Combine results
all_predictions = torch.cat(predictions)
```

---

## See Also

- **[BaseModel](base.md)** - Common model interface
- **[DRN](drn.md)** - Using GLM as baseline for refinement
- **[Training](../training.md)** - Advanced training techniques
- **[Quick Start](../../getting-started/quickstart.md)** - Practical examples