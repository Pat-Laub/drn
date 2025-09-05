# Interpretability

Model interpretation and explanation tools for understanding DRN predictions and distributional properties.

## DRN Explainer

Main class for interpreting DRN models using SHAP and custom visualization methods.

::: drn.interpretability.DRNExplainer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Kernel SHAP Integration

Specialized SHAP explainer for distributional properties of DRN models.

::: drn.kernel_shap_explainer.KernelSHAP_DRN
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Key Methods

### Plot Adjustment Factors

::: drn.interpretability.DRNExplainer.plot_adjustment_factors
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Plot Distributional Property Adjustment with SHAP

::: drn.interpretability.DRNExplainer.plot_dp_adjustment_shap
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### CDF Plot

::: drn.interpretability.DRNExplainer.cdf_plot
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4