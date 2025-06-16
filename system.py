import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data Loading and Preprocessing Functions
def load_data(file_path):
    """Load the fire detection dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    df_processed = df.copy()
    
    # Convert date to datetime and extract features
    df_processed['acq_date'] = pd.to_datetime(df_processed['acq_date'], format='%Y-%m-%d')
    print(df_processed['acq_date'].head())
    df_processed['day_of_year'] = df_processed['acq_date'].dt.dayofyear
    df_processed['month'] = df_processed['acq_date'].dt.month
    
    # Create fire risk categories based on brightness and FRP (Fire Radiative Power)
    df_processed['fire_risk'] = create_fire_risk_categories(df_processed)
    
    # Handle categorical variables
    le_confidence = LabelEncoder()
    df_processed['confidence_encoded'] = le_confidence.fit_transform(df_processed['confidence'])
    
    le_daynight = LabelEncoder()
    df_processed['daynight_encoded'] = le_daynight.fit_transform(df_processed['daynight'])
    
    # Create weather severity index
    df_processed['weather_severity'] = calculate_weather_severity(df_processed)
    
    # Create temperature difference
    df_processed['temp_range'] = df_processed['tmax'] - df_processed['tmin']
    
    return df_processed, le_confidence, le_daynight

def create_fire_risk_categories(df):
    """Create fire risk categories based on brightness and FRP"""
    conditions = [
        (df['brightness'] < 300) & (df['frp'] < 1.0),
        (df['brightness'] < 320) & (df['frp'] < 2.0),
        (df['brightness'] < 340) & (df['frp'] < 4.0),
        (df['brightness'] >= 340) | (df['frp'] >= 4.0)
    ]
    choices = [0, 1, 2, 3]  # Low, Medium, High, Very High
    return np.select(conditions, choices, default=1)

def calculate_weather_severity(df):
    """Calculate weather severity index based on temperature, precipitation, and wind"""
    # Normalize components
    temp_factor = (df['tmax'] - df['tmax'].min()) / (df['tmax'].max() - df['tmax'].min())
    wind_factor = (df['wspd'] - df['wspd'].min()) / (df['wspd'].max() - df['wspd'].min())
    # Inverse precipitation factor (less rain = higher severity)
    precip_factor = 1 - ((df['prcp'] - df['prcp'].min()) / (df['prcp'].max() - df['prcp'].min() + 0.1))
    
    return (temp_factor * 0.4 + wind_factor * 0.3 + precip_factor * 0.3)

def prepare_features(df):
    """Prepare features for machine learning"""
    feature_columns = [
        'latitude', 'longitude', 'brightness', 'scan', 'track',
        'bright_t31', 'frp', 'tavg', 'tmin', 'tmax', 'prcp', 'wspd',
        'confidence_encoded', 'daynight_encoded', 'day_of_year', 'month',
        'weather_severity', 'temp_range'
    ]
    
    X = df[feature_columns].copy()
    y = df['fire_risk'].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X, y, feature_columns

# Model Training Functions
def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train Random Forest model"""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=10,
        min_samples_split=5
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        multi_class='ovr'
    )
    lr_model.fit(X_train_scaled, y_train)
    return lr_model, scaler

def train_svm(X_train, y_train, random_state=42):
    """Train SVM model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm_model = SVC(
        kernel='rbf',
        random_state=random_state,
        probability=True
    )
    svm_model.fit(X_train_scaled, y_train)
    return svm_model, scaler

# Model Evaluation Functions
def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """Evaluate model performance"""
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_pred_proba, accuracy

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title(f'Top 15 Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    return None

# Prediction Functions
def predict_fire_risk(model, input_data, scaler=None, feature_columns=None):
    """Predict fire risk for new data"""
    if isinstance(input_data, dict):
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Ensure all required features are present
    if feature_columns:
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing features
        input_df = input_df[feature_columns]
    
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)
        else:
            probabilities = None
    else:
        prediction = model.predict(input_df)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)
        else:
            probabilities = None
    
    return prediction, probabilities

def get_risk_description(risk_level):
    """Get description for risk level"""
    descriptions = {
        0: "Low Risk - Minimal fire activity expected",
        1: "Medium Risk - Moderate fire activity possible",
        2: "High Risk - Significant fire activity likely",
        3: "Very High Risk - Extreme fire activity expected"
    }
    return descriptions.get(risk_level, "Unknown risk level")

# Data Analysis Functions
def analyze_fire_patterns(df):
    """Analyze fire patterns in the data"""
    print("Fire Risk Distribution:")
    risk_counts = df['fire_risk'].value_counts().sort_index()
    for risk, count in risk_counts.items():
        print(f"Risk Level {risk}: {count} incidents ({count/len(df)*100:.1f}%)")
    
    print(f"\nAverage brightness by risk level:")
    brightness_by_risk = df.groupby('fire_risk')['brightness'].mean()
    for risk, brightness in brightness_by_risk.items():
        print(f"Risk Level {risk}: {brightness:.2f}")
    
    print(f"\nAverage FRP by risk level:")
    frp_by_risk = df.groupby('fire_risk')['frp'].mean()
    for risk, frp in frp_by_risk.items():
        print(f"Risk Level {risk}: {frp:.2f}")

def plot_risk_distribution(df):
    """Plot fire risk distribution"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    df['fire_risk'].value_counts().sort_index().plot(kind='bar')
    plt.title('Fire Risk Distribution')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    df.groupby('fire_risk')['brightness'].mean().plot(kind='bar')
    plt.title('Average Brightness by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Brightness')
    
    plt.subplot(1, 3, 3)
    df.groupby('fire_risk')['frp'].mean().plot(kind='bar')
    plt.title('Average FRP by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('FRP')
    
    plt.tight_layout()
    plt.show()

# Main execution function
def run_fire_prediction_system(file_path):
    """Run the complete fire prediction system"""
    print("=== Fire Prediction System ===")
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = load_data(file_path)
    if df is None:
        return
    
    df_processed, le_confidence, le_daynight = preprocess_data(df)
    
    # Analyze patterns
    print("\n2. Analyzing fire patterns...")
    analyze_fire_patterns(df_processed)
    plot_risk_distribution(df_processed)
    
    # Prepare features
    print("\n3. Preparing features...")
    X, y, feature_columns = prepare_features(df_processed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    print("\n4. Training models...")
   
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
    
    # SVM
    print("Training SVM...")
    svm_model, svm_scaler = train_svm(X_train, y_train)
    
    # Evaluate models
    print("\n5. Evaluating models...")
    
    lr_pred, lr_proba, lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression", lr_scaler)
    svm_pred, svm_proba, svm_acc = evaluate_model(svm_model, X_test, y_test, "SVM", svm_scaler)
    
    # Plot confusion matrices
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
    
        
    # Best model selection
    best_acc = max(lr_acc, svm_acc)
    if best_acc == lr_acc:
        best_model = lr_model
        best_scaler = lr_scaler
        best_name = "Logistic Regression"
    else:
        best_model = svm_model
        best_scaler = svm_scaler
        best_name = "SVM"
    
    print(f"\nBest model: {best_name} with accuracy: {best_acc:.4f}")
    
    # Example prediction
    print("\n6. Example prediction...")
    sample_data = {
        'latitude': 38.13,
        'longitude': 23.52,
        'brightness': 320.0,
        'scan': 0.4,
        'track': 0.37,
        'bright_t31': 290.0,
        'frp': 2.0,
        'tavg': 25.0,
        'tmin': 18.0,
        'tmax': 32.0,
        'prcp': 0.0,
        'wspd': 8.0,
        'confidence_encoded': 1,
        'daynight_encoded': 0,
        'day_of_year': 150,
        'month': 5,
        'weather_severity': 0.7,
        'temp_range': 14.0
    }
    
    prediction, probabilities = predict_fire_risk(
        best_model, sample_data, best_scaler, feature_columns
    )
    
    print(f"Predicted fire risk level: {prediction[0]}")
    print(f"Risk description: {get_risk_description(prediction[0])}")
    if probabilities is not None:
        print("Probabilities for each risk level:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  Risk Level {i}: {prob:.3f}")
    
    return {
        'best_model': best_model,
        'best_scaler': best_scaler,
        'feature_columns': feature_columns,
        'encoders': {'confidence': le_confidence, 'daynight': le_daynight},
        'accuracies': {'lr': lr_acc, 'svm': svm_acc}
    }

# Example usage
if __name__ == "__main__":
    # Replace 'paste.txt' with your actual file path
    file_path = 'cleaned.csv'
    
    # Run the fire prediction system
    results = run_fire_prediction_system(file_path)
    
    if results:
        print("\nSystem trained successfully!")
       