import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Global Pollution Analysis and Energy Recovery

## Objective
The goal is to analyze global pollution data and develop strategies for pollution reduction and converting pollutants into energy.

## Phase 1: Data Collection and Exploratory Data Analysis (EDA)
### Step 1 - Data Import and Preprocessing
"""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")"""

code_load = """# Load the dataset
df = pd.read_csv('Global_Pollution_Analysis.csv')
print(df.head())
print(df.info())"""

code_missing = """# Handle Missing Values
print("Missing values before:\\n", df.isnull().sum())

# Fill numeric with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after:\\n", df.isnull().sum())"""

code_transform = """# Data Transformation
# Normalize pollution indices
scaler = StandardScaler()
pollution_indices = ['Air_Pollution_Index', 'Water_Pollution_Index', 'Soil_Pollution_Index']
df[pollution_indices] = scaler.fit_transform(df[pollution_indices])

# Encode categorical features
encoder = LabelEncoder()
df['Country_Encoded'] = encoder.fit_transform(df['Country'])
# Year is numeric but could be treated as ordinal or categorical if needed, 
# for now keeping Year as numeric for trend analysis
"""

text_eda = """### Step 2 - Exploratory Data Analysis (EDA)"""

code_eda_stats = """# Descriptive Statistics
print("Descriptive Stats:\\n", df[['CO2_Emissions', 'Industrial_Waste_in_tons']].describe())"""

code_eda_corr = """# Correlation Analysis
plt.figure(figsize=(12, 10))
# Select numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()"""

code_eda_viz = """# Visualizations
plt.figure(figsize=(14, 6))
# Top 10 Countries by CO2 Emissions
top_countries = df.groupby('Country')['CO2_Emissions'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')
plt.title("Top 10 Countries by Average CO2 Emissions")
plt.xticks(rotation=45)
plt.show()

# Trend of Air Pollution over Time
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Year', y='Air_Pollution_Index', estimator='mean')
plt.title("Global Average Air Pollution Index Trend")
plt.show()"""

text_feature_eng = """### Step 3 - Feature Engineering"""

code_feature = """# Yearly Trends (Already visualized, but explicitly ensuring Year is valid)
# Energy Consumption per Capita
# Assuming Population is in millions
df['Energy_Consumption_Per_Capita'] = df['Energy_Consumption'] / df['Population (in millions)']
print("Feature Engineering complete. New head:\\n", df.head())"""

text_phase2 = """## Phase 2: Predictive Modeling
### Step 4 - Linear Regression Model (for Pollution Prediction of Energy Recovery)"""

code_linear = """# Predict Energy Recovery (in GWh)
# Features: Air_Pollution_Index, CO2_Emissions, Industrial_Waste_in_tons
X_lin = df[['Air_Pollution_Index', 'CO2_Emissions', 'Industrial_Waste_in_tons', 'Energy_Consumption']]
y_lin = df['Energy_Recovery (in GWh)']

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)
y_pred_lin = lin_model.predict(X_test_lin)

print("Linear Regression Performance:")
print(f"MSE: {mean_squared_error(y_test_lin, y_pred_lin):.2f}")
print(f"MAE: {mean_absolute_error(y_test_lin, y_pred_lin):.2f}")
print(f"R2: {r2_score(y_test_lin, y_pred_lin):.2f}")"""

text_logistic = """### Step 5 - Logistic Regression Model (for Categorization of Pollution Levels)"""

code_logistic = """# Classify Pollution Severity
# Create bins for Air_Pollution_Index: Low, Medium, High
# Using quantiles
df['Pollution_Severity'] = pd.qcut(df['Air_Pollution_Index'], q=3, labels=['Low', 'Medium', 'High'])
print("Pollution Severity Distribution:\\n", df['Pollution_Severity'].value_counts())

# Features for classification
X_log = df[['CO2_Emissions', 'Industrial_Waste_in_tons', 'Energy_Consumption']]
y_log = df['Pollution_Severity']

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

log_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
log_model.fit(X_train_log, y_train_log)
y_pred_log = log_model.predict(X_test_log)

print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test_log, y_pred_log):.2f}")
print("Classification Report:\\n", classification_report(y_test_log, y_pred_log))"""

text_phase3 = """## Phase 3: Reporting and Insights
### Step 6 - Model Evaluation and Comparison"""

code_viz_res = """# Confusion Matrix for Logistic Regression
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_log, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=log_model.classes_, yticklabels=log_model.classes_)
plt.title("Confusion Matrix - Pollution Severity Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Comparison discussion (Text)
print("Linear Regression R2 indicates how well we explain Energy Recovery variance.")
print("Logistic Regression Accuracy indicates how well we categorize pollution levels.")
"""

text_final = """### Step 7 - Actionable Insights
1. **Pollution vs Energy Recovery**: Higher pollution often correlates with industrial activity, which might provide more waste for energy recovery, but the trade-off needs management.
2. **Severity Classification**: Identifying 'High' severity regions helps target immediate interventions.
3. **Recommendations**:
    - Invest in waste-to-energy technologies in high industrial waste zones.
    - Implement stricter CO2 caps in identified high-pollution countries.
"""

nb.cells = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_missing),
    nbf.v4.new_code_cell(code_transform),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda_stats),
    nbf.v4.new_code_cell(code_eda_corr),
    nbf.v4.new_code_cell(code_eda_viz),
    nbf.v4.new_markdown_cell(text_feature_eng),
    nbf.v4.new_code_cell(code_feature),
    nbf.v4.new_markdown_cell(text_phase2),
    nbf.v4.new_code_cell(code_linear),
    nbf.v4.new_markdown_cell(text_logistic),
    nbf.v4.new_code_cell(code_logistic),
    nbf.v4.new_markdown_cell(text_phase3),
    nbf.v4.new_code_cell(code_viz_res),
    nbf.v4.new_markdown_cell(text_final)
]

with open(r"c:\Users\yuvra\OneDrive - st.niituniversity.in\Personal Growth\Machine Learning(ML)\Assignment-2\Global_Pollution_Analysis.ipynb", 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
