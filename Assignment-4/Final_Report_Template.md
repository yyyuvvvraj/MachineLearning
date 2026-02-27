# Final Report: Global Pollution Analysis and Energy Recovery

## 1. Objective
Classify countries into pollution severity categories: **Low, Medium, High**.

## 2. Dataset and Preprocessing
- Dataset: `Global_Pollution_Analysis.csv`
- Missing data handling:
  - Numerical features: median imputation
  - Categorical features: most-frequent imputation
- Outlier handling: IQR-based capping (winsorization)
- Feature scaling: MinMaxScaler (supports MultinomialNB non-negative requirement)
- Categorical encoding:
  - `Country` encoded with LabelEncoder
  - `Year` encoded using LabelEncoder (as requested)

## 3. Feature Engineering
- `Pollution_Score` = average of air, water, and soil pollution indices
- `Pollution_Severity` target from `Pollution_Score` tertiles:
  - Low, Medium, High
- `Energy_Recovered_Per_Capita_Proxy` = Energy_Recovered / Population
- `Pollution_Trend_Yearly` = year-over-year change in pollution score by country

## 4. Models and Evaluation
Models trained:
- Multinomial Naive Bayes
- K-Nearest Neighbors (CV tuning for K)
- Decision Tree (CV tuning for max_depth, min_samples_split, min_samples_leaf)

Evaluation metrics:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion Matrix

## 5. Results Table
Fill from notebook outputs:

| Model | Accuracy | Precision (W) | Recall (W) | F1-score (W) |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes |  |  |  |  |
| KNN (Best) |  |  |  |  |
| Decision Tree (Best) |  |  |  |  |

## 6. Visual Findings
- Confusion matrices for all three classifiers
- Classification reports
- Decision Tree feature importance chart (optional)

## 7. Key Insights
- Best overall classifier: `...`
- Countries/classes with highest misclassification: `...`
- Most important predictive features: `...`
- Relationship observed between pollution severity and energy recovery: `...`

## 8. Actionable Recommendations
1. Prioritize policy interventions for countries consistently classified as **High** severity.
2. Improve energy recovery infrastructure in high-pollution, low-recovery regions.
3. Use yearly trend monitoring for early warning and targeted regulation.
4. Retrain model periodically with latest environmental data.

## 9. Conclusion
Summarize the best model and policy implications for pollution reduction and energy recovery.
