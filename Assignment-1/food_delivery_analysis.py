
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ast import literal_eval
import warnings

warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")

# File Path
DATA_PATH = "Food_Delivery_Time_Prediction.csv"

def load_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    if 'Order_ID' in df.columns:
        print("Dropping Order_ID column")
        df = df.drop(columns=['Order_ID'])
    return df

def clean_data(df):
    print("Checkpoint: Entering clean_data")
    # print("Original dtypes:\n", df.dtypes)
    # print("DataFrame Head:\n", df.head())
    
    # Explicitly force numeric columns
    numeric_features = ['Distance', 'Delivery_Person_Experience', 'Restaurant_Rating', 'Customer_Rating', 'Delivery_Time', 'Order_Cost', 'Tip_Amount']
    print("Checkpoint: Starting coercion loop")
    for col in numeric_features:
        # print(f"Processing column for coercion: {col}")
        try:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Error coercing column {col}: {e}")
    print("Checkpoint: Finished coercion loop")

            
    # print("Dtypes after coercion:\n", df.dtypes)
    
    # Check current missing values
    print("Checking missing values iteratively...")
    for col in df.columns:
        try:
            n_missing = df[col].isnull().sum()
            print(f"{col}: {n_missing}")
        except Exception as e:
            print(f"Error checking {col}: {e}")
    
    # Handle missing values
    # For numeric columns, fill with median
    try:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        print(f"Numeric columns identified: {numeric_cols.tolist()}")
        for col in numeric_cols:
            print(f"Filling median for: {col}")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    except Exception as e:
        print(f"Error in numeric filling: {e}")

    # For categorical columns, fill with mode
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        print(f"Categorical columns identified: {categorical_cols.tolist()}")
        for col in categorical_cols:
            print(f"Filling mode for: {col}")
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                   df[col] = df[col].fillna(mode_val[0])
                else:
                   df[col] = df[col].fillna("Unknown")
    except Exception as e:
        print(f"Error in categorical filling: {e}")
            
    print("Missing values after cleaning:\n", df.isnull().sum())
    return df

def parse_coordinates(coord_str):
    try:
        # Check if it's already a tuple or needs parsing
        if isinstance(coord_str, str):
            # Remove parentheses if they exist and split
            coord_str = coord_str.strip('()')
            parts = coord_str.split(',')
            return float(parts[0]), float(parts[1])
        return np.nan, np.nan
    except:
        return np.nan, np.nan

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def perform_eda(df):
    print("Performing EDA...")
    
    # Descriptive Statistics
    print("Descriptive Statistics:\n", df.describe())
    
    # Correlation Analysis
    plt.figure(figsize=(12, 8))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.close()
    
    # Outlier Detection (Boxplots)
    features_to_check = ['Distance', 'Delivery_Time', 'Order_Cost']
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(features_to_check):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
    plt.savefig("outliers.png")
    plt.close()

def feature_engineering(df):
    print("Checkpoint: Feature Engineering (swapped order)")
    
    # Calculate Distances
    # Extract lat/long
    # Assuming format "(lat, long)"
    cust_coords = df['Customer_Location'].apply(parse_coordinates)
    rest_coords = df['Restaurant_Location'].apply(parse_coordinates)
    
    df['Cust_Lat'] = cust_coords.apply(lambda x: x[0])
    df['Cust_Long'] = cust_coords.apply(lambda x: x[1])
    df['Rest_Lat'] = rest_coords.apply(lambda x: x[0])
    df['Rest_Long'] = rest_coords.apply(lambda x: x[1])
    
    df['Calculated_Distance_km'] = haversine(df['Cust_Lat'], df['Cust_Long'], df['Rest_Lat'], df['Rest_Long'])
    
    # Drop original location columns to avoid string issues
    print("Dropping original location columns")
    df = df.drop(columns=['Customer_Location', 'Restaurant_Location'])

    # Time-based features
    rush_hours = ['Evening', 'Afternoon'] 
    df['Is_Rush_Hour'] = df['Order_Time'].apply(lambda x: 1 if x in rush_hours else 0)
    
    return df

def preprocess_for_modeling(df):
    print("Preprocessing for Modeling...")
    
    # dropping columns not needed (ID already dropped, Locations dropped)
    # Cust_Lat, Cust_Long... kept for now? Maybe drop them if distance is usage.
    # User didn't ask to remove them, but regression might not use raw lat/long well. 
    # Let's keep them but maybe tree models use them. Linear regression? Not useful.
    
    drop_cols = ['Order_ID', 'Customer_Location', 'Restaurant_Location', 'Cust_Lat', 'Cust_Long', 'Rest_Lat', 'Rest_Long']
    # Filter only existing
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df_model = df.drop(columns=cols_to_drop)
    else:
        df_model = df.copy()
    
    # Label Encoding
    encoder = LabelEncoder()
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_model[col] = encoder.fit_transform(df_model[col].astype(str))
        
    print("Data Columns for modeling:", df_model.columns.tolist())
    
    # Normalize features
    scaler = StandardScaler()
    target = 'Delivery_Time'
    # Delivery_Class might assume to be there? No, that's logistic regression specific.
    
    features = [c for c in df_model.columns if c != target and c != 'Delivery_Class']
    
    # Handle NaNs before scaling if any remained (clean_data should have caught them)
    # But clean_data is now AFTER feature_engineering. 
    # clean_data handles numeric. Lat/Long are numeric.
    
    df_model[features] = scaler.fit_transform(df_model[features])
    
    return df_model

def train_linear_regression(df):
    print("Training Linear Regression...")
    X = df.drop('Delivery_Time', axis=1)
    y = df['Delivery_Time']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Linear Regression Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    
    return model, X_test, y_test, y_pred

def train_logistic_regression(df):
    print("Training Logistic Regression...")
    # Create Binary Target
    # "Fast" vs "Delayed". Let's use Median as split point.
    median_time = df['Delivery_Time'].median()
    print(f"Median Delivery Time for classification threshold: {median_time}")
    
    df['Delivery_Class'] = (df['Delivery_Time'] > median_time).astype(int) # 1 for Delayed (High time), 0 for Fast
    
    X = df.drop(['Delivery_Time', 'Delivery_Class'], axis=1)
    y = df['Delivery_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Logistic Regression Results:")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    
    # Swapped order: Feature Engineering first to remove problematic strings
    try:
        df = feature_engineering(df)
    except Exception as e:
        print(f"Feature Engineering Failed: {e}")
        
    try:
        df = clean_data(df)
    except Exception as e:
        print(f"Clean Data Failed: {e}")
    
    # EDA now
    try:
        perform_eda(df)
    except Exception as e:
        print(f"EDA Failed: {e}")
    
    try:
        df_model = preprocess_for_modeling(df)
        train_linear_regression(df_model)
        train_logistic_regression(df_model) # Passed df_model which has numerical columns
    except Exception as e:
        print(f"Modeling Failed: {e}")
