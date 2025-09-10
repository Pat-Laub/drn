# Models Overview

Core distributional regression models in the DRN package. All models inherit from the common `BaseModel` interface and provide distributional predictions.

## Model Hierarchy

```mermaid
classDiagram
    BaseModel <|-- GLM
    BaseModel <|-- DeepGLM
    BaseModel <|-- DRN
    BaseModel <|-- CANN
    BaseModel <|-- MDN
    BaseModel <|-- DDR
    BaseModel <|-- Constant
    
    class BaseModel {
        +fit(X, y)
        +predict(X)
        +quantiles(X, percentiles)
        +loss(x, y)
    }
    
    class GLM {
        +distribution: str
        +clone()
    }
    
    class DRN {
        +baseline: BaseModel
        +cutpoints: list
        +log_adjustments(x)
    }
```

## Quick Reference

| Model | Purpose | Key Features | Best For |
|-------|---------|-------------|----------|
| **[BaseModel](base.md)** | Abstract base class | Common interface, PyTorch Lightning | All models inherit from this |
| **[GLM](glm.md)** | Generalized Linear Models | Interpretable, statistical foundation | Baseline models, simple relationships |
| **[DeepGLM](deepglm.md)** | Deep Generalized Linear Model | Neural feature learning + GLM head | Nonlinear relationships, distributional outputs |
| **[DRN](drn.md)** | Distributional Refinement Network | Neural + interpretable baseline | Complex distributions with interpretability |
| **[CANN](cann.md)** | Combined Actuarial Neural Network | Actuarial focus, separate parameter networks | Insurance and actuarial applications |
| **[MDN](mdn.md)** | Mixture Density Network | Multi-modal distributions | Complex, multi-peaked data |
| **[DDR](ddr.md)** | Deep Distribution Regression | Pure neural approach | Maximum flexibility, no baseline constraint |
| **[Constant](constant.md)** | Constant prediction | Simple baseline | Benchmarking, ablation studies |

## Common Usage Patterns

### Basic Model Training
```python
from drn import GLM, DRN

# Train baseline
baseline = GLM('gamma')
baseline.fit(X_train, y_train)

# Train refined model
drn_model = DRN(baseline)
drn_model.fit(X_train, y_train)
```

### Distribution Families

#### GLM Distributions
- **`gaussian`** - Normal distribution for unbounded continuous data
- **`gamma`** - Gamma distribution for positive continuous data
- **`inversegaussian`** - Inverse Gaussian for positive data with right skew
- **`lognormal`** - Log-normal for multiplicative processes

#### Advanced Distributions
- **Histogram** - Flexible discrete distributions
- **Extended Histogram** - Continuous extensions of histograms
- **Mixture Models** - Multi-component distributions

### Model Selection Guide

#### Start with GLM when:
- You need interpretability
- Data follows standard distributions
- Baseline performance is adequate
- Statistical inference is required

#### Use DRN when:
- GLM baseline is reasonable but not sufficient
- You need both flexibility and interpretability
- Complex distributional shapes are present
- Regularization control is important

#### Consider Advanced Models when:
- **CANN**: Actuarial applications with domain knowledge
- **MDN**: Multi-modal or mixture distributions expected
- **DDR**: Maximum flexibility, no interpretability needed
- **Constant**: Simple benchmarking baseline

## Model Comparison

### Performance Characteristics

| Model | Training Speed | Inference Speed | Memory Usage | Interpretability | Flexibility |
|-------|---------------|----------------|--------------|------------------|-------------|
| GLM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| DeepGLM | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| DRN | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| CANN | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| MDN | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| DDR | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
