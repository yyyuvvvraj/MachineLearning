# Final Report: Food Delivery Time Prediction (CNN + Validation)

## 1. Objective
Predict food delivery status as **Fast** or **Delayed** and evaluate CNN performance against traditional baseline models.

## 2. Dataset and Preprocessing
- Dataset: `Food_Delivery_Time_Prediction.csv`
- Missing data handling:
  - Numerical features: median imputation
  - Categorical features: most-frequent imputation
- Encoding: One-Hot Encoding for categorical variables
- Normalization: StandardScaler for numerical features
- Feature engineering:
  - `Haversine_Distance_km` from customer/restaurant coordinates
  - `Rush_Hour` indicator from order time
  - Weather impact summary on delay rates
  - Binary target `Delivery_Status` from median `Delivery_Time`

## 3. CNN Methodology
- Tabular features transformed into image-like matrices for CNN input
- CNN architecture:
  - Convolution + pooling blocks
  - Dense layers with dropout
  - Sigmoid output for binary classification
- Hyperparameter tuning:
  - Filters, kernel size, and learning rate tested via manual search

## 4. Model Evaluation and Validation
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - ROC-AUC
- Validation:
  - 5-fold CV for Logistic Regression
  - Reduced-fold CNN validation (if TensorFlow available)

## 5. Model Comparison
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |  |
| CNN / MLP Fallback |  |  |  |  |  |

## 6. Visualizations Included
- Confusion matrices for both models
- ROC curve comparison
- CNN training curves (loss/accuracy, if TensorFlow used)

## 7. Key Findings
- Best-performing model: `...`
- CNN vs Logistic Regression performance gap: `...`
- Key delay-driving conditions (traffic/weather/rush-hour): `...`

## 8. Actionable Recommendations
1. Trigger proactive dispatch when high-risk delay conditions are predicted.
2. Increase rider allocation during rush-hour and adverse weather windows.
3. Use model outputs to optimize routing and reduce delayed deliveries.
4. Retrain the model periodically with recent data for stable performance.

## 9. Conclusion
Summarize final model selection and operational impact for reducing delivery delays.
