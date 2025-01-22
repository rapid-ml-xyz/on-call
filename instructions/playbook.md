## Detailed RCA Playbook

Below is a **step-by-step** explanation, referencing each node in the diagram. The steps are written to help even a non-data-scientist follow a structured process for uncovering *model* (not engineering) issues and producing actionable recommendations.

---

### **A. Start: Performance Drop Detected**

1. **What You Do**  
   - Gather context: When did the performance drop start? How is it measured? Which business units or stakeholders reported it?
   - Confirm that it is indeed a **model** performance issue (not just a monitoring glitch, data pipeline issue, or label mismatch).

2. **Why It Matters**  
   - You must ensure the problem is about *model outputs* rather than environment or pipeline errors.  
   - If you suspect data pipeline or ETL bugs, route to data-engineering teams.  

3. **Inputs**  
   - (T) Performance Metrics from your real-time or batch monitoring system (e.g., accuracy, F1, or other relevant KPI).

4. **Output**  
   - Clear definition: “Yes, the model’s KPI (accuracy, AUC, or business-defined metric) dropped from X% to Y%.”

---

### **B. Define Impact Window**

1. **What You Do**  
   - Identify the **date range** or time window where the issue is happening.  
   - Determine how that compares to your baseline or reference period (often a training period or stable production period).

2. **Why It Matters**  
   - Pinpointing the time window helps you isolate *potential changes* in data distribution, user behavior, or any newly introduced features.

3. **Inputs**  
   - (T) Performance Metrics aggregated over time  
   - (V) Model Predictions or logs over different time intervals

4. **Output**  
   - A specific “impact window,” e.g., “The drop started on 2025-01-10.”

---

### **C. Compare with Baseline**

1. **What You Do**  
   - Compare the *current impacted performance metrics* against a known stable period (the baseline).  
   - This helps quantify **how big** the drop is and in which exact metrics it’s most pronounced (e.g., drop in recall vs. drop in precision).

2. **Why It Matters**  
   - Sometimes, *only certain metrics degrade* (like recall), while others remain stable. This suggests different root causes.

3. **Inputs**  
   - (T) Performance Metrics from both current (impacted) period and baseline  
   - (W) Ground Truth Labels (if available for the new window)

4. **Output**  
   - A table or chart highlighting metric shifts (e.g., “Recall down by 15%, precision stable” or “RMSE up by 10% in a regression context”).

---

### **D. Identify Pattern**

This is a **decision node** that helps us classify whether the performance degradation is *sudden* or *gradual*.

1. **Sudden Drop**  
   - Suggests a potential major data shift, new product launch, feature pipeline glitch, or *model versioning* issue (e.g., a recent deployment).  
   - Leads to (E) Time-based Analysis first (to see *exactly* which date or event triggered the sudden drop).

2. **Gradual Decline**  
   - Suggests ongoing data drift, concept drift, or evolving user behavior.  
   - Leads to (F) Drift Analysis.

---

### **E. Time-Based Analysis (for Sudden Drop)**

1. **What You Do**  
   - Slice metrics by smaller time intervals (daily, hourly) to pinpoint exactly when the performance changed.  
   - Check any external events: new feature releases, marketing campaigns, or domain shifts that coincide with the timeline.

2. **Why It Matters**  
   - A sudden performance cliff often correlates with a discrete change (new data feed, code deployment).  

3. **Outcome**  
   - Pinpoint the day/time the drop occurred.  
   - This step feeds into **(G) Cohort Analysis** because we suspect a particular group or time-based subset might be impacted.

---

### **F. Drift Analysis (for Gradual Decline)**

1. **What You Do**  
   - Compare historical feature distributions with current ones:  
     - **Statistical Tests** (KS test, PSI, KL divergence) to measure drift.  
     - Look for distribution changes in key features.  
   - Evaluate if there is **concept drift** (the relationship between features and target changes).

2. **Why It Matters**  
   - A gradual decline often implies real-world data or user behavior is drifting away from what the model was trained on.

3. **Tools**  
   - (AA) Statistical Tests (PSI, KL Divergence, etc.)  
   - (Z) Feature Importance (to see if a once-important feature is no longer stable)

4. **Outcome**  
   - If drift is detected, proceed with **(G) Cohort Analysis** to see exactly which subset is drifting the most.

---

### **G. Cohort Analysis**

1. **What You Do**  
   - Segment data by relevant dimensions: region, product line, user type, time buckets, etc.  
   - Compare performance metrics *per cohort* (e.g., accuracy in each region or recall for each product category).

2. **Why It Matters**  
   - Pinpoint if the performance drop is localized. If one or two cohorts are heavily impacted, you can focus your deeper analysis there.

3. **Inputs**  
   - (U) Raw Feature Values  
   - (W) Ground Truth Labels, if available for each segment

4. **Decision**  
   - **High Impact Segments** → (H) Feature Distribution Analysis  
   - **No Clear Segments** → (I) Global Performance Analysis

---

### **H. Feature Distribution Analysis (High Impact Segments)**

1. **What You Do**  
   - For each *high impact cohort*, analyze how the feature distributions differ from the training distribution.  
   - Plot histograms, box plots, or compute summary statistics for the relevant feature columns.

2. **Why It Matters**  
   - A distribution shift in a key feature might break the model’s assumptions or cause it to over-/under-predict in that cohort.

3. **Outcome**  
   - Determine if a **Distribution Shift** has occurred.

4. **Decision**  
   - If **Yes** → (K) Feature Importance Analysis  
   - If **No** → (L) Error Analysis  

---

### **I. Global Performance Analysis (No Clear Segments)**

1. **What You Do**  
   - If you cannot isolate any segment with drastically worse performance, you do a *global-level deep dive*.  
   - Possibly compare global distribution of features from training vs. current data.

2. **Why It Matters**  
   - The problem might be affecting *all cohorts equally* (e.g., an overall drift, or a systemic issue like an over-regularized model).

3. **Outcome**  
   - Summarize the global-level changes in performance or distribution.

4. **Next Step**  
   - Go to (M) Global Metrics Review to see *which* metrics are failing or to (O) Model Behavior Analysis if all metrics are equally degraded.

---

### **J. Distribution Shift?** (within High Impact Segments)

- This node is explicitly checking if a given segment’s feature distribution is significantly different from the training data or normal production distribution.

---

### **K. Feature Importance Analysis (If Distribution Shift is Confirmed)**

1. **What You Do**  
   - Evaluate how important the *shifted* features are in the model. If the model heavily relies on those features, the shift can be catastrophic.  
   - Check your existing model’s feature importances (e.g., from a random forest or gradient boosting model).  
   - Optionally, generate new importance metrics via (X) SHAP or (Z) Permutation Importance.

2. **Why It Matters**  
   - If a heavily used feature is drastically shifting, the model’s learned decision boundaries might be invalid for new data.

3. **Output**  
   - A ranked list of features driving the model’s predictions in the impacted segment.

4. **Next Step**  
   - (P) Local Explanation to get instance-level insight, or move directly into root cause synthesis.

---

### **L. Error Analysis (No Distribution Shift in High Impact Segments)**

1. **What You Do**  
   - Check **confusion matrix** (classification) or residuals (regression) within the high-impact segment.  
   - Identify predominant error modes: false positives vs. false negatives, large residuals, or underestimation vs. overestimation.

2. **Why It Matters**  
   - The model might have a certain structural limitation or a thresholding problem.  
   - Example: The model always misclassifies a specific sub-class, not necessarily because the feature distribution changed, but because the learned boundary is poor.

3. **Tools**  
   - (AA) Statistical tests can help verify if error rates differ significantly across sub-classes.  
   - (X) SHAP or (Z) Feature Importance to see if the model is ignoring certain features.

4. **Outcome**  
   - Enhanced understanding of *how* the model is failing in the segment.

5. **Next Step**  
   - (P) Local Explanation for specific instances or go to root cause.

---

### **M. Global Metrics Review (If No High Impact Segments)**

1. **What You Do**  
   - Break down *all* metrics (accuracy, precision, recall, F1, AUC for classification; RMSE, MAE, R^2 for regression, etc.).  
   - Compare the magnitude of changes: is the largest drop in recall? Is it in F1?

2. **Why It Matters**  
   - Distinguishes if the performance degrade is mainly about missing minority classes, or an overall shift in predictions.

3. **Decision**  
   - If a **Specific Metric** (e.g., recall) is notably worse → (N) Metric-Focused Analysis.  
   - If **All Metrics** degrade equally → (O) Model Behavior Analysis.

---

### **N. Metric-Focused Analysis**

1. **What You Do**  
   - Zero in on what influences that particular metric (e.g., recall might be impacted by threshold settings or missing certain classes).  
   - Possibly tune or simulate different thresholds to see if you can recapture lost performance.

2. **Why It Matters**  
   - In classification, a drop in recall can indicate the model is too conservative, or class imbalance changed.  
   - In regression, an increase in MSE might be due to heavier-tailed errors in new data.

3. **Next Step**  
   - (P) Local Explanation or go to root cause if you suspect thresholding or data imbalance is the culprit.

---

### **O. Model Behavior Analysis (Overall Degradation)**

1. **What You Do**  
   - Conduct a global “sanity check” on the model’s predictions across the board.  
   - Possibly run a fresh holdout or offline evaluation to confirm if the model has truly lost generalization.  
   - Evaluate if model hyperparameters or architecture are still appropriate.

2. **Why It Matters**  
   - Systemic degradation might mean your model is out of date or there’s a fundamental mismatch with new patterns in data.

3. **Next Step**  
   - (P) Local Explanation or finalize root cause.

---

### **P. Local Explanation**

1. **What You Do**  
   - Use (X) SHAP or LIME to interpret predictions of specific *misclassified* or *high-error* samples.  
   - Look for anomalies: do certain features unexpectedly dominate the explanation for incorrectly predicted instances?

2. **Why It Matters**  
   - Sometimes local examples reveal hidden biases or spurious correlations that are not visible at the global level.

3. **Outcome**  
   - Detailed instance-level clues about why the model is making certain errors.

---

### **Q. Root Cause Identified?**

This is a critical decision node:

- **If Yes** → (R) Recommendations  
- **If No** → (S) Iterate with Different Cohort

Sometimes, after analyzing one segment or set of features, you still haven’t pinned down the exact cause. *Iterate* with different slicing, different features, or deeper domain knowledge until you find a plausible explanation.

---

### **R. Recommendations**

1. **What You Do**  
   - Propose corrective actions based on the discovered root cause. Examples:  
     - **Retrain** with newly available data that reflects current distribution  
     - **Adjust thresholds** for classification metrics  
     - **Feature engineering** to handle new data patterns  
     - **Data augmentation** if certain classes/segments are underrepresented  
     - **Regular re-check or monitoring** for drift  
     - In the case of concept drift, implement a frequent retraining schedule or an online learning approach.

2. **Why It Matters**  
   - The ultimate goal is to fix the bug and restore model performance.  
   - Your recommendations must be *actionable*, tested in a staging environment, and ideally measured with an A/B test or holdout set to confirm improvement.

3. **Outcome**  
   - A plan or list of tasks for the data science and engineering teams to execute.

---

### **S. Iterate with Different Cohort**

1. **What You Do**  
   - If the current analysis did not yield a conclusive root cause, return to step (G) *Cohort Analysis* but try different segmentations.  
   - Possibly incorporate domain knowledge from SMEs to identify *non-obvious* cohorts (like seasonal changes, marketing segments, device types, etc.).

2. **Why It Matters**  
   - Real-world data is complex. Sometimes the standard segmentation (region, time) doesn’t isolate the effect; you might need advanced segmentations or domain-specific slices.

3. **Outcome**  
   - Repeats the loop until you can isolate or confirm a root cause.

---

## Expanded Notes & Best Practices

1. **Separating Data Quality vs. Model Bugs**  
   - Your process excludes data pipeline or ETL errors. However, if at any point you strongly suspect *data corruption* or missing data, triage that to the data-engineering domain (but keep track to confirm it’s not a fundamental *modeling* or *drift* issue).

2. **Extensibility to Regression & Recommendation**  
   - Replace classification metrics (accuracy, precision, recall, etc.) with regression metrics (MAE, RMSE, R^2) or recommendation metrics (precision@k, recall@k, NDCG).  
   - The steps remain largely the same—cohort analysis, distribution checks, local vs. global explanations, and so on.

3. **Tools & Libraries**  
   - (X) SHAP or LIME for local instance explanation.  
   - (Y) PDP/ICE to visualize partial dependence on key features.  
   - (Z) Feature importance from random forest, gradient boosting, or permutation importance methods.  
   - (AA) Statistical tests (e.g., KS Test, PSI, chi-square) to detect distribution or label shifts.

4. **Monitoring & Alerting**  
   - To prevent large performance drops going unnoticed, set up robust monitoring and drift-detection pipelines.  
   - This helps you catch issues *before* they become severe.

5. **Root Cause Synthesis**  
   - Often, the final solution is multi-faceted: data drift *plus* threshold misalignment, or partial label shift *plus* new user segments. Summarize all contributing factors and weigh them in your final recommendation plan.

-----

## Conclusion

By following this **robust RCA playbook**—guided by your Mermaid diagram and expanded with deeper analyses—you can isolate *why* a model is failing in production (e.g., distribution drift, threshold issues, incomplete training data, or model hyperparameter misalignment). Each decision node systematically narrows down potential culprits. Once you’ve identified the root cause(s), you can confidently propose targeted fixes—from retraining to new feature engineering—ensuring the model regains its intended performance.