# CANN - Combined Actuarial Neural Network

Advanced neural network architecture designed specifically for actuarial and insurance applications.

---

## Class Definition

::: drn.models.cann.CANN
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      show_bases: true

<!-- 
## Actuarial Applications

### Insurance Claim Modeling
```python
# Model insurance claims with gamma distribution
claim_model = CANN(
    input_dim=10,
    distribution='gamma',
    hidden_sizes=[128, 64, 32],
    use_embedding=True,
    categorical_features=[0, 3, 7]  # Policy type, region, vehicle class
)

# Incorporate actuarial constraints
claim_model.add_monotonicity_constraint('age', direction='increasing')
claim_model.add_positivity_constraint(['income', 'vehicle_value'])

# Train with actuarial loss
claim_model.fit(
    X_train, 
    y_train, 
    validation_data=(X_val, y_val),
    epochs=200,
    early_stopping=True
)

# Generate regulatory capital calculations
predictions = claim_model.predict(X_test)
var_95 = claim_model.quantiles(X_test, [95])[0]
expected_shortfall = claim_model.expected_shortfall(X_test, 95)
``` -->
