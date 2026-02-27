# Final Report: Food Delivery Time Prediction (Clustering + Neural Networks)

## 1. Objective
Predict delivery status (**Fast** vs **Delayed**) and analyze delivery patterns using clustering techniques.

## 2. Dataset and Preprocessing
- Dataset: `Food_Delivery_Time_Prediction.csv`
- Missing value handling:
  - Numerical: median imputation
  - Categorical: most-frequent imputation
- Encoding:
  - One-Hot Encoding for categorical variables
- Scaling:
  - StandardScaler on numerical features
- Feature engineering:
  - `Haversine_Distance_km` from latitude/longitude
  - `Rush_Hour` binary feature from order time
  - Binary target `Delivery_Status` from median delivery time

## 3. Clustering Analysis
### 3.1 K-Means
- Elbow Method used to select optimal K
- Chosen K: `...`
- Key cluster characteristics:
  - Cluster with highest average delivery time: `...`
  - Cluster with highest delayed-rate: `...`

### 3.2 Hierarchical Clustering
- Dendrogram used to inspect hierarchy and cluster structure
- Agglomerative cluster count selected: `...`
- Comparison with K-Means findings: `...`

## 4. Prediction Models
### 4.1 Baseline
- Logistic Regression

### 4.2 Neural Network
- Model: Feedforward Neural Network (Keras/TensorFlow) or MLP fallback
- Tuned settings:
  - Hidden layers: `...`
  - Activation: `...`
  - Learning rate: `...`
  - Epochs / batch size: `...`

## 5. Evaluation Metrics
| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |
| Neural Network |  |  |  |  |

## 6. Visualizations Included
- Elbow curve (K-Means)
- PCA scatter plot for clusters
- Dendrogram (hierarchical clustering)
- Confusion matrices
- Neural network training curves (loss/accuracy)

## 7. Key Findings
- Best model for delayed prediction: `...`
- Most influential delivery patterns discovered from clustering: `...`
- Agreement/disagreement between clustering and supervised predictions: `...`

## 8. Actionable Recommendations
1. Allocate delivery resources proactively for high-delay cluster patterns.
2. Use rush-hour-aware and traffic-aware route optimization.
3. Prioritize interventions for weather/traffic combinations that drive delayed deliveries.
4. Retrain model periodically with new order and traffic data.

## 9. Conclusion
Summarize the most effective method and practical strategies to reduce delays.
