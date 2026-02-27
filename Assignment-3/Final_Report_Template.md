# Final Report: Food Delivery Status Classification

## 1. Objective
Predict delivery status as **Fast (0)** or **Delayed (1)** using order, route, and contextual features.

## 2. Dataset and Preprocessing
- Dataset: `Food_Delivery_Time_Prediction.csv`
- Missing value handling:
  - Numerical: median imputation
  - Categorical: most-frequent imputation
- Categorical encoding: LabelEncoder
- Feature normalization: StandardScaler on continuous features
- Feature engineering:
  - Parsed `Customer_Location` and `Restaurant_Location` into latitude/longitude
  - Computed `Haversine_Distance_km`
  - Created binary target `Delivery_Status` from median `Delivery_Time`

## 3. Models Trained
- Gaussian Naive Bayes
- K-Nearest Neighbors (with CV tuning for `n_neighbors`)
- Decision Tree (with CV tuning for `max_depth`, `min_samples_split`, `min_samples_leaf`)

## 4. Evaluation Metrics
Use values from the notebook:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gaussian Naive Bayes |  |  |  |  |  |
| KNN (Best) |  |  |  |  |  |
| Decision Tree (Best) |  |  |  |  |  |

## 5. Visual Analysis
- Confusion matrices of all three models
- ROC curves with AUC comparison
- (Optional) Decision Tree feature importance bar chart

## 6. Key Findings
- Best performing model (by F1-score): `...`
- Best model for minimizing missed delays (high Recall for class 1): `...`
- Most interpretable model: `...`
- Main limitations observed: `...`

## 7. Actionable Recommendations
1. Deploy `...` model for production use (justify with metric values).
2. Monitor delay-class Recall and Precision monthly.
3. Improve model with additional features (e.g., weekday/weekend, peak-hour indicator, rider workload).
4. Re-tune hyperparameters after collecting new data.

## 8. Conclusion
Summarize which classifier is most suitable for this business scenario and why.
