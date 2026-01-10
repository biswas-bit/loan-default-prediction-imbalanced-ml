# Loan Default Risk Detection – Modeling & Evaluation

## 1. Model Overview

We trained a **XGBoost classifier** to predict loan default risk.  
Key considerations:

- Target variable is highly imbalanced (~92% non-default, 8% default)
- Model outputs probabilities; threshold is tunable
- Evaluation focused on **minority class recall, PR-AUC, and F1-score** rather than accuracy

---

## 2. Handling Imbalanced Data

- Class distribution:
- 0 91.93%
- 1 8.07%
Name: TARGET, dtype: float64
---
- Approach:
- Used `scale_pos_weight` in XGBoost
- Avoided SMOTE due to categorical-heavy features
- Optimized decision threshold to increase recall for defaulters

---

## 3. Model Training

**Final XGBoost Parameters**:

| Parameter | Value |
|-----------|-------|
| n_estimators | 400 |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | Calculated from class imbalance |
| objective | binary:logistic |
| eval_metric | auc |
| random_state | 42 |

**Training Notes**:

- Tree-based models (XGB) handle skewed features and unscaled data well
- Categorical features encoded minimally to preserve information
- Threshold tuning performed to optimize recall for minority class

---

## 4. Threshold Optimization

- Default threshold (0.5) is **too high** for minority class
- Tested thresholds from 0.05 to 0.5
- Selected threshold ~0.25–0.30 for **balanced recall and precision**
- Precision-recall trade-off is acceptable for business risk screening

---

## 5. Evaluation Metrics (Test Set)

**Classification Report at Threshold 0.25–0.30:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Non-default) | 0.98 | 0.39 | 0.56 | 56538 |
| 1 (Default)     | 0.11 | 0.90 | 0.20 | 4965 |

**Other Metrics:**
- Accuracy: 0.43 (not meaningful due to imbalance)
- Macro F1-score: 0.38
- Weighted F1-score: 0.53
- PR-AUC: 0.24 (baseline for random classifier = 0.08)

---

## 6. Trade-offs and Interpretation

| Objective | Approach | Outcome / Trade-off | Comment |
|-----------|---------|-------------------|---------|
| Capture as many defaulters as possible | Lowered decision threshold | Recall (class 1) = 0.90 | High recall ensures minimal missed defaulters |
| Minimize false positives | Acceptable loss in precision | Precision (class 1) = 0.11 | False positives can be manually reviewed in practice |
| Imbalanced data evaluation | PR-AUC and F1 for minority class | PR-AUC = 0.24 | Much better than random (0.08) |
| Accuracy | Ignored | Accuracy = 0.43 | Misleading due to imbalance; not reported as primary metric |

**Key insight:**  
> The model prioritizes **catching defaulters (high recall)** at the cost of **low precision**, which is acceptable for risk screening. Accuracy is not used as a performance metric because of the high class imbalance.

---

## 7. Business Relevance

- **False negatives** (missed defaulters) are very costly → minimized
- **False positives** (flagged safe loans) are manageable → can be reviewed manually
- **Threshold tuning** aligns model performance with business objectives
- Tree-based models allow robust handling of skewed numeric features and mixed data types

---

## 8. Next Steps (Optional Enhancements)

1. **SHAP Analysis:** Explain feature contributions for flagged defaulters  
2. **Cost-sensitive Analysis:** Model expected financial loss/gain per threshold  
3. **Probability Calibration:** Improve probability estimates using Platt or isotonic scaling  
4. **Model Comparison Table:** Compare XGB vs Random Forest vs Logistic Regression using PR-AUC and recall

---

## 9. Interview-Ready Explanation

> “We trained a tree-based classifier for loan default prediction. Due to extreme class imbalance (~92:8), we optimized for **recall on the minority class**. Threshold tuning increased recall for defaulters to 90%, accepting lower precision. PR-AUC is 0.24, which is ~3× better than random. Accuracy is not meaningful here. This approach aligns with business goals: minimize missed defaulters while allowing manual review for false positives.”

---

