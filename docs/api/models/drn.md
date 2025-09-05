# DRN - Distributional Refinement Network

The main DRN model that combines interpretable baselines with flexible neural network refinements for advanced distributional forecasting.

---

## Class Definition

::: drn.models.drn.DRN
    options:
      show_root_heading: false
      show_source: false  
      heading_level: 3
      show_bases: true

---

## Overview

The **Distributional Refinement Network (DRN)** is the flagship model of this package. It addresses the fundamental challenge of building models that are both interpretable and flexible by:

1. **Starting with an interpretable baseline** (typically a GLM)
2. **Adding neural refinements** through a deep network
3. **Operating on discretized regions** defined by cutpoints  
4. **Balancing flexibility with regularization** through multiple penalty terms

The result is a model that maintains the interpretability of the baseline while achieving superior distributional forecasting performance.

## Architecture Overview

```mermaid
flowchart TB
    A[Input Features X] --> B[Baseline Model]
    A --> C[Neural Network]
    
    B --> D[Baseline Distribution P₀(y|x)]
    C --> E[Log Adjustments δ(x)]
    
    D --> F[Discretization via Cutpoints]
    E --> F
    
    F --> G[Refined Distribution P(y|x)]
    
    G --> H[Final Predictions]
    
    subgraph "Regularization"
        I[KL Divergence: KL(P₀||P)]
        J[Roughness: Smoothness penalty] 
        K[Mean: Deviation from baseline mean]
    end
    
    F --> I
    F --> J
    F --> K
```

---

## Key Parameters

### 🏗️ Model Architecture
- **`baseline`** - The baseline model (typically GLM) providing interpretable foundation
- **`cutpoints`** - Discretization points defining refinement regions  
- **`hidden_size`** - Neurons per hidden layer (default: 75)
- **`num_hidden_layers`** - Number of hidden layers (default: 2)
- **`dropout_rate`** - Dropout probability for regularization (default: 0.2)

### ⚙️ Training Control
- **`baseline_start`** - Initialize neural weights to zero (start from baseline)
- **`learning_rate`** - Adam optimizer learning rate (default: 1e-3)
- **`loss_metric`** - Loss function type ("jbce" or "nll")

### 🎛️ Regularization Parameters
- **`kl_alpha`** - KL divergence penalty weight (default: 0.0)
- **`mean_alpha`** - Mean deviation penalty weight (default: 0.0)
- **`dv_alpha`** - Density variation (roughness) penalty weight (default: 0.0)
- **`kl_direction`** - KL divergence direction ("forwards" or "backwards")

---

## Quick Start

### Basic DRN Training

```python
from drn import GLM, DRN, train
from drn.models import drn_cutpoints, drn_loss
import torch

# 1. Train interpretable baseline
baseline = GLM('gamma')
baseline.fit(X_train, y_train)

# 2. Define refinement region  
cutpoints = drn_cutpoints(
    c_0=y_train.min() * 1.1,
    c_K=y_train.max() * 1.1,
    p=0.1,        # 10% cutpoints-to-observation ratio
    y=y_train,
    min_obs=10    # Minimum observations per interval
)

# 3. Initialize DRN
drn_model = DRN(
    baseline=baseline,
    cutpoints=cutpoints,
    hidden_size=128,
    num_hidden_layers=2,
    dropout_rate=0.1
)

# 4. Train with custom loss
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
    lambda pred, y: drn_loss(
        pred, y,
        kl_alpha=1e-4,
        dv_alpha=1e-3,
        mean_alpha=1e-5
    ),
    train_dataset,
    val_dataset,
    epochs=100
)
```

---

## Cutpoints System

The heart of DRN's flexibility lies in its cutpoints system, which discretizes the response space into refinement regions.

### 🎯 Cutpoint Generation

::: drn.models.drn_cutpoints
    options:
      show_root_heading: false
      show_source: false
      heading_level: 4

### Cutpoint Strategies

```python
from drn.models import drn_cutpoints
import numpy as np

# Strategy 1: Quantile-based (recommended)
cutpoints = drn_cutpoints(
    c_0=y_train.quantile(0.01),  # 1st percentile lower bound
    c_K=y_train.quantile(0.99),  # 99th percentile upper bound  
    p=0.08,                      # 8% cutpoints-to-observation ratio
    y=y_train,
    min_obs=15
)

# Strategy 2: Fixed bounds
cutpoints = drn_cutpoints(
    c_0=0,              # Fixed lower bound
    c_K=1000,           # Fixed upper bound
    p=0.1,
    y=y_train,
    min_obs=20
)

# Strategy 3: Data-driven bounds
margin = (y_train.max() - y_train.min()) * 0.1
cutpoints = drn_cutpoints(
    c_0=y_train.min() - margin,
    c_K=y_train.max() + margin,
    p=0.05,
    y=y_train,
    min_obs=25
)

print(f"Generated {len(cutpoints)} cutpoints")
print(f"Refinement range: [{cutpoints[0]:.2f}, {cutpoints[-1]:.2f}]")
```

### Cutpoint Guidelines

| Dataset Size | Recommended `p` | Recommended `min_obs` | Rationale |
|-------------|-----------------|----------------------|-----------|
| < 1,000 | 0.15-0.20 | 0-5 | More flexibility, fewer constraints |
| 1,000-10,000 | 0.08-0.12 | 5-15 | Balanced approach |
| 10,000-100,000 | 0.03-0.08 | 15-50 | More stability, avoid overfitting |
| > 100,000 | 0.01-0.03 | 50-100 | Conservative, focus on main effects |

---

## Regularization Deep Dive

DRN uses sophisticated regularization to balance baseline adherence with neural flexibility.

### 🎯 KL Divergence Control (`kl_alpha`)

Controls deviation from baseline distribution:

```python
# Conservative refinement (stay close to baseline)
kl_alpha = 1e-5
# Effect: Small deviations, preserves baseline structure

# Moderate refinement  
kl_alpha = 1e-4
# Effect: Balanced refinement, good starting point

# Aggressive refinement
kl_alpha = 1e-3
# Effect: Large deviations allowed, more neural influence
```

**Direction Control:**
- `kl_direction='forwards'`: `KL(P_baseline || P_drn)` - Penalize when DRN differs from baseline
- `kl_direction='backwards'`: `KL(P_drn || P_baseline)` - Penalize when baseline differs from DRN

### 🎯 Roughness Penalty (`dv_alpha`)

Ensures smooth density functions:

```python
# High smoothness (conservative shapes)
dv_alpha = 1e-2
# Effect: Very smooth densities, fewer local maxima

# Moderate smoothness
dv_alpha = 1e-3  
# Effect: Balanced flexibility and smoothness

# High flexibility (complex shapes allowed)
dv_alpha = 1e-4
# Effect: Complex density shapes possible
```

### 🎯 Mean Penalty (`mean_alpha`)

Controls predicted mean deviation:

```python
# Force mean close to baseline
mean_alpha = 1e-3
# Effect: DRN mean stays near baseline mean

# Moderate mean constraint
mean_alpha = 1e-5
# Effect: Some mean deviation allowed

# Free mean adjustment
mean_alpha = 0.0
# Effect: Mean can deviate freely from baseline
```

---

## Complete Training Example

### Advanced Insurance Claims Model

```python
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from drn import GLM, DRN, train
from drn.models import drn_cutpoints, drn_loss
from drn.metrics import crps, rmse, quantile_losses
from drn.utils import split_and_preprocess

# Load and preprocess data
claims_data = pd.read_csv('insurance_claims.csv')
X = claims_data[['age', 'income', 'vehicle_age', 'region', 'policy_type']]
y = claims_data['claim_amount']

# Sophisticated data preprocessing
x_train, x_val, x_test, y_train, y_val, y_test, \
x_train_raw, x_val_raw, x_test_raw, \
num_features, cat_features, all_categories, ct = split_and_preprocess(
    X, y,
    numerical_features=['age', 'income', 'vehicle_age'],
    categorical_features=['region', 'policy_type'],
    test_size=0.2,
    val_size=0.1,
    num_standard=True,
    seed=42
)

# Convert to tensors
X_train = torch.tensor(x_train.values, dtype=torch.float32)
Y_train = torch.tensor(y_train.values, dtype=torch.float32) 
X_val = torch.tensor(x_val.values, dtype=torch.float32)
Y_val = torch.tensor(y_val.values, dtype=torch.float32)
X_test = torch.tensor(x_test.values, dtype=torch.float32)
Y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Step 1: Train baseline GLM
print("Training baseline GLM...")
baseline = GLM('gamma')  # Gamma distribution for insurance claims
baseline.fit(X_train, Y_train)

baseline_pred = baseline.predict(X_test)
baseline_rmse = rmse(Y_test, baseline_pred.mean)
print(f"Baseline RMSE: ${baseline_rmse:.2f}")

# Step 2: Design sophisticated cutpoints
print("Designing cutpoint strategy...")
cutpoints = drn_cutpoints(
    c_0=Y_train.quantile(0.005).item(),  # 0.5th percentile (handle outliers)
    c_K=Y_train.quantile(0.995).item(),  # 99.5th percentile  
    p=0.06,                              # 6% ratio for large dataset
    y=y_train,
    min_obs=30                           # Ensure statistical significance
)

print(f"Cutpoints: {len(cutpoints)} intervals")
print(f"Range: ${cutpoints[0]:.0f} - ${cutpoints[-1]:.0f}")

# Step 3: Configure DRN architecture
print("Configuring DRN architecture...")
torch.manual_seed(42)  # Reproducibility

drn_model = DRN(
    baseline=baseline,
    cutpoints=cutpoints,
    
    # Architecture decisions
    hidden_size=256,           # Larger network for complex data
    num_hidden_layers=3,       # Deeper for more capacity
    dropout_rate=0.15,         # Moderate regularization
    
    # Training configuration  
    baseline_start=True,       # Initialize from baseline (recommended)
    learning_rate=0.0005       # Conservative learning rate
)

# Step 4: Define sophisticated loss function
def sophisticated_drn_loss(pred_dist, y_true):
    """Custom loss with adaptive regularization."""
    
    # Base loss with carefully tuned regularization
    base_loss = drn_loss(
        pred_dist, y_true,
        kl_alpha=2e-4,          # Moderate KL penalty
        dv_alpha=8e-4,          # Smoothness penalty
        mean_alpha=1e-5,        # Light mean constraint
        kl_direction='forwards'
    )
    
    # Optional: Add custom terms
    # Penalize extreme predictions
    mean_pred = pred_dist.mean
    extreme_penalty = torch.mean(torch.relu(mean_pred - Y_train.max() * 2))
    
    return base_loss + 1e-3 * extreme_penalty

# Step 5: Advanced training with monitoring  
print("Training DRN with advanced configuration...")

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

# Train with careful monitoring
training_history = train(
    drn_model,
    sophisticated_drn_loss,
    train_dataset,
    val_dataset,
    
    # Training parameters
    lr=0.0005,
    batch_size=128,           # Balanced batch size
    epochs=150,               # Sufficient training
    patience=25,              # Patient early stopping
    
    # Monitoring
    log_interval=5,
    verbose=True
)

print("DRN training completed!")

# Step 6: Comprehensive evaluation
print("\n" + "="*50)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*50)

# Generate predictions
drn_pred = drn_model.predict(X_test)

# Point prediction metrics
drn_rmse = rmse(Y_test, drn_pred.mean)
improvement = (baseline_rmse - drn_rmse) / baseline_rmse * 100

print(f"\n📊 Point Prediction Performance:")
print(f"Baseline RMSE:     ${baseline_rmse:.2f}")
print(f"DRN RMSE:         ${drn_rmse:.2f}")  
print(f"Improvement:       {improvement:.1f}%")

# Distributional metrics
grid = torch.linspace(Y_test.min() - 100, Y_test.max() + 100, 2000)

baseline_cdf = baseline_pred.cdf(grid.unsqueeze(-1))
drn_cdf = drn_pred.cdf(grid.unsqueeze(-1))

baseline_crps = crps(Y_test, grid, baseline_cdf).mean()
drn_crps = crps(Y_test, grid, drn_cdf).mean()
crps_improvement = (baseline_crps - drn_crps) / baseline_crps * 100

print(f"\n📈 Distributional Performance:")
print(f"Baseline CRPS:     ${baseline_crps:.2f}")
print(f"DRN CRPS:         ${drn_crps:.2f}")
print(f"CRPS Improvement:  {crps_improvement:.1f}%")

# Risk measure evaluation  
print(f"\n🎯 Risk Measure Performance:")
risk_percentiles = [90, 95, 99]

for percentile in risk_percentiles:
    baseline_ql = quantile_losses(
        percentile/100, baseline, "Baseline", X_test, Y_test,
        l=Y_test.min()-100, u=Y_test.max()+100
    )
    drn_ql = quantile_losses(
        percentile/100, drn_model, "DRN", X_test, Y_test,
        l=Y_test.min()-100, u=Y_test.max()+100
    )
    ql_improvement = (baseline_ql - drn_ql) / baseline_ql * 100
    
    print(f"{percentile}th percentile loss: "
          f"Baseline={baseline_ql:.4f}, DRN={drn_ql:.4f} "
          f"({ql_improvement:+.1f}%)")

print("\n✅ DRN training and evaluation complete!")
```

---

## Advanced Features

### 🔧 Custom Loss Functions

```python
def custom_drn_loss(pred_dist, y_true):
    """Custom loss with domain-specific penalties."""
    
    # Base DRN loss
    base_loss = drn_loss(pred_dist, y_true, 
                        kl_alpha=1e-4, dv_alpha=1e-3, mean_alpha=1e-5)
    
    # Custom penalty: discourage predictions below zero
    mean_pred = pred_dist.mean  
    negative_penalty = torch.mean(torch.relu(-mean_pred)) * 1e-2
    
    # Custom penalty: encourage reasonable variance
    if hasattr(pred_dist, 'variance'):
        var_penalty = torch.mean(torch.relu(pred_dist.variance - y_true.var() * 5))
        return base_loss + negative_penalty + var_penalty * 1e-4
    
    return base_loss + negative_penalty
```

### 🔧 Lazy Initialization

```python
# DRN can automatically determine cutpoints during training
drn_model = DRN(
    baseline=baseline,
    cutpoints=None,         # Will be determined automatically
    proportion=0.08,        # Cutpoints-to-observation ratio
    min_obs=15,            # Minimum observations per interval
    hidden_size=128
)

# Cutpoints are created during .fit() call
drn_model.fit(X_train, y_train)
print(f"Auto-generated {len(drn_model.cutpoints)} cutpoints")
```

### 🔧 Multi-Stage Training

```python
# Stage 1: High regularization for stable initialization
stage1_loss = lambda pred, y: drn_loss(pred, y, 
                                      kl_alpha=1e-3, dv_alpha=1e-2, mean_alpha=1e-4)
train(drn_model, stage1_loss, train_dataset, val_dataset, epochs=30)

# Stage 2: Reduce regularization for flexibility  
stage2_loss = lambda pred, y: drn_loss(pred, y,
                                      kl_alpha=1e-4, dv_alpha=1e-3, mean_alpha=1e-5)
train(drn_model, stage2_loss, train_dataset, val_dataset, epochs=50)

# Stage 3: Fine-tuning with minimal regularization
stage3_loss = lambda pred, y: drn_loss(pred, y,
                                      kl_alpha=1e-5, dv_alpha=1e-4, mean_alpha=0)
train(drn_model, stage3_loss, train_dataset, val_dataset, epochs=20)
```

---

## Hyperparameter Tuning Guide

### 🎯 Quick Tuning Strategy

1. **Start Conservative**: `kl_alpha=1e-4, dv_alpha=1e-3, mean_alpha=1e-5`
2. **Check Baseline Quality**: If baseline is poor, decrease `kl_alpha`
3. **Monitor Smoothness**: If densities are jagged, increase `dv_alpha`
4. **Adjust Complexity**: More cutpoints = more flexibility but harder training

### 🎯 Systematic Grid Search

```python
def tune_drn_hyperparameters(baseline, X_train, Y_train, X_val, Y_val):
    """Simple hyperparameter tuning for DRN."""
    
    best_score = float('inf')
    best_params = None
    
    # Define search grid
    kl_alphas = [1e-5, 1e-4, 1e-3]
    dv_alphas = [1e-4, 1e-3, 1e-2]  
    hidden_sizes = [64, 128, 256]
    
    for kl_alpha in kl_alphas:
        for dv_alpha in dv_alphas:
            for hidden_size in hidden_sizes:
                
                # Create model
                drn = DRN(baseline, cutpoints, hidden_size=hidden_size)
                
                # Define loss
                loss_fn = lambda pred, y: drn_loss(pred, y, 
                                                  kl_alpha=kl_alpha, 
                                                  dv_alpha=dv_alpha)
                
                # Train
                train(drn, loss_fn, train_dataset, val_dataset, epochs=30)
                
                # Evaluate  
                pred = drn.predict(X_val)
                score = rmse(Y_val, pred.mean)
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'kl_alpha': kl_alpha,
                        'dv_alpha': dv_alpha, 
                        'hidden_size': hidden_size
                    }
    
    return best_params, best_score
```

---

## Troubleshooting

### ⚠️ Common Issues

#### Training Loss Not Decreasing
```python
# Solutions:
# 1. Lower learning rate
drn_model = DRN(baseline, cutpoints, learning_rate=1e-4)

# 2. Reduce regularization  
loss_fn = lambda pred, y: drn_loss(pred, y, kl_alpha=1e-5, dv_alpha=1e-4)

# 3. Check baseline quality
baseline_pred = baseline.predict(X_val)
baseline_score = rmse(Y_val, baseline_pred.mean)
print(f"Baseline validation RMSE: {baseline_score}")
```

#### Memory Issues
```python
# Reduce batch size
train(drn_model, loss_fn, train_dataset, val_dataset, batch_size=64)

# Reduce model size
drn_model = DRN(baseline, cutpoints, hidden_size=64, num_hidden_layers=1)

# Use gradient accumulation
# Effective batch size = batch_size * accumulation_steps
```

#### Overfitting
```python
# Increase regularization
loss_fn = lambda pred, y: drn_loss(pred, y, kl_alpha=1e-3, dv_alpha=1e-2)

# Increase dropout
drn_model = DRN(baseline, cutpoints, dropout_rate=0.3)

# Reduce model capacity
drn_model = DRN(baseline, cutpoints, hidden_size=32, num_hidden_layers=1)
```

---

## See Also

- **[BaseModel](base.md)** - Common model interface
- **[GLM](glm.md)** - Baseline model implementation
- **[Training](../training.md)** - Advanced training strategies
- **[Quick Start](../../getting-started/quickstart.md)** - Practical examples
- **[Advanced Usage](../../getting-started/advanced-usage.md)** - Custom training loops