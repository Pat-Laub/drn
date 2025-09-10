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
