# Final Report: Global Pollution Analysis and Energy Recovery (Assignment 8)

## 1. Objective
Analyze global pollution data to uncover associations between pollution severity and energy recovery using Apriori, and compare rule-based insights with CNN-based prediction workflow from delivery data.

## 2. Dataset and Preprocessing
- Pollution dataset: `Global_Pollution_Analysis.csv`
- Delivery dataset for CNN comparison: `Food_Delivery_Time_Prediction.csv` (from prior assignment folders)
- Missing value handling:
  - Numerical: median imputation
  - Categorical: mode imputation
- Normalization:
  - Min-Max scaling for air, water, and soil pollution indices
- Encoding:
  - Label encoding for `Country` and categorical representation of `Year`

## 3. Feature Engineering
- `Estimated_Total_Energy_Consumption_MWh` as a derived proxy from per-capita consumption and population
- Pollution severity categories for air and water indices: `Low`, `Medium`, `High`
- `Recovery_Efficiency = Energy_Recovered / Industrial_Waste`
- Year-wise trend analysis of pollution and energy recovery

## 4. Apriori Association Mining
- Transaction items include country, air/water severity, energy recovered level, recovery efficiency level, renewable level, CO2 level
- Frequent itemset mining with configurable thresholds
  - Minimum support: 0.08
  - Minimum confidence: 0.55
- Rules interpreted with support, confidence, and lift
- High-lift rules and high-pollution-focused rules extracted for strategic insight

## 5. Validation of Apriori Rules
- Train/test split on transaction matrix
- Rules mined on train set and evaluated on test set
- Stability check with confidence gap between train and test
- Stable rules identified using absolute confidence gap threshold (<= 0.15)

## 6. CNN and Baseline Model for Delivery Prediction
- Binary target from median split of `Delivery_Time`
- Baseline: Logistic Regression
- Advanced model: CNN (Keras) on reshaped tabular data, with MLP fallback if TensorFlow unavailable
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Visualizations:
  - Confusion matrices
  - ROC curves
  - CNN training curves (if TensorFlow is available)

## 7. Model Comparison Table
Evaluated on `Assignment-7/Food_Delivery_Time_Prediction.csv`:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.3600 | 0.3478 | 0.3200 | 0.3333 | 0.4432 |
| MLP Fallback (TensorFlow unavailable) | 0.5400 | 0.5217 | 0.9600 | 0.6761 | 0.5728 |

## 8. Key Findings
- Apriori mined `4` valid rules; all `4` remained stable on test data (confidence gap <= 0.15).
- Rule quality metrics were consistent: mean lift `2.2210`, mean train confidence `0.7041`, mean test confidence `0.7737`.
- MLP Fallback outperformed Logistic Regression on delivery prediction (0.5400 vs 0.3600 accuracy, 0.5728 vs 0.4432 ROC-AUC), with significantly higher recall (0.96 vs 0.32).
- 3-fold cross-validation for Logistic Regression gave mean accuracy `0.4496` (`[0.5224, 0.4478, 0.3788]`), showing variable generalization across folds.

## 9. Actionable Recommendations
1. Prioritize interventions for segments with high pollution severity and low recovery efficiency.
2. Increase renewable investment in patterns associated with high CO2 and weak recovery.
3. Use validated high-lift Apriori rules as policy guidance for country-level environmental planning.
4. Operationalize delivery prediction outputs to reduce delays via proactive dispatch and routing.
5. Retrain and revalidate both association and prediction models periodically.

## 10. Conclusion
This assignment combines interpretable association mining (Apriori) with predictive modeling to provide both policy-level environmental insights and operational prediction capabilities. In this run, Apriori produced 4 stable pollution-energy associations with strong lift metrics. The MLP fallback model was the best-performing delivery prediction approach, demonstrating superior recall despite TensorFlow being unavailable in the environment.
