## Possible error scenarios

### Performance-Based Errors

1. Overfitting: Model memorizes training data, fails to generalize.
2. Underfitting: Model too simplistic, misses key patterns.
3. High Variance: Predictions fluctuate greatly with small data changes.
4. High Bias: Model systematically underperforms due to overly restrictive assumptions.

### Metric & Measurement Issues

1. Low Accuracy / High Error Rate: Simple broad underperformance.
2. Low Precision / Recall / F1: Often signals imbalance or threshold problems.
3. Misaligned or Inappropriate Metric: Chosen metric doesn’t reflect business needs.
4. Incorrect Metric Implementation: Bugs or mistakes in how metrics are calculated.

### Distribution / Drift Problems

1. Covariate Shift: Feature distribution changes while the relationship to the label remains.
1. Prior/Label Shift: The label distribution changes over time.
2. Concept Drift: The underlying relationship between features and label changes.
3. Temporal Mismatch / Evolving Data: Time-based shifts that invalidate training assumptions.

### Threshold / Calibration Mistakes

1. Overconfidence: Model probability estimates are too high.
2. Underconfidence: Model underestimates its confidence.
3. Poor Calibration / Wrong Threshold: Incorrect cutoff for class assignment or probability interpretation.

### Cohort / Localized Failures

1. Underperformance in Specific Segments: Certain feature subsets degrade model predictions.
2. Class Imbalance: Model struggles with minority classes.
3. Rare Classes / Outliers: Extreme cases not well captured by the model.

### Explainability / Attribution Gaps

1. Spurious Feature Correlations: Model relies on “junk” or accidental correlations.
2. Over-Reliance on Certain Features: Model fixates on a small set of features.
3. Inconsistent Model Explanations: Explainable AI methods reveal contradictory or nonsensical rationale.

### Fairness & Bias Issues

1. Protected Group Discrimination: Systematic error against certain groups.
2. Unintentional Protected Attribute Correlation: Model inadvertently encodes sensitive attributes.
3. Unfair Outcomes / Allocations: Resulting decisions harm or exclude certain demographics.

### Labeling / Annotation Errors

1. Noise in Ground Truth: Labels contain random or systematic errors.
2. Mislabeling / Ambiguous Labels: Training data lacks clarity, leading to model confusion.
3. Incomplete / Partial Label Coverage: Some samples or classes are unlabeled or underrepresented.

### Data Leakage or Training-Serving Mismatch

1. Feature Leakage: Future or external information inadvertently included in training.
2. Mismatch Between Train & Production Features: Data transformations differ between training and inference.
3. Improper Splitting Leading to Leakage: Overlapping data in train/test sets inflates performance estimates.

### Adversarial Vulnerabilities

1. Model Susceptibility to Adversarial Examples: Slightly perturbed inputs fool the model.
2. Malicious Data Injection: Attackers insert corrupt samples to degrade performance.
3. Attacks on Model Interpretability: Methods that obscure or confuse explainability tools.