import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib

# Load and prepare the data (same as in your notebook)
df = pd.read_csv('Customer Churn.csv')
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
df["TotalCharges"] = df["TotalCharges"].astype("float")

# Select features
features = [
    'tenure',            # How long customer stayed
    'MonthlyCharges',    # Monthly payment
    'TotalCharges',      # Total payment
    'Contract',          # Contract type
    'InternetService',   # Internet type
    'TechSupport',       # Tech support
    'PaperlessBilling'   # Paperless billing
]

X = df[features]
y = df['Churn']

# Encode categorical variables
X_encoded = X.copy()

# Contract mapping
X_encoded['Contract'] = X_encoded['Contract'].map({
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
})

# InternetService mapping
X_encoded['InternetService'] = X_encoded['InternetService'].map({
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2
})

# TechSupport mapping
X_encoded['TechSupport'] = X_encoded['TechSupport'].map({
    'No': 0,
    'Yes': 1,
    'No internet service': 2
})

# PaperlessBilling mapping
X_encoded['PaperlessBilling'] = X_encoded['PaperlessBilling'].map({
    'No': 0,
    'Yes': 1
})

# Target variable
y_encoded = y.map({'No': 0, 'Yes': 1})

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y_encoded)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split data
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X_scaled, y_resampled))

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

# Train the three models with the same parameters as in your notebook

# Logistic Regression with hyperparameter tuning
param_grid_extended = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear'],
    'max_iter': [2000]
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_extended,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Create final model with best parameters
lr_model = LogisticRegression(**best_params, random_state=42)
lr_model.fit(X_train, y_train)

# SVM Model
svm_model = SVC(
    random_state=42,
    probability=True,
    class_weight='balanced',
    C=1.0,
    kernel='rbf'
)
svm_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
rf_model.fit(X_train, y_train)

# Save all models and the scaler
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_encoded.columns.tolist(), 'feature_columns.pkl')

print("Models saved successfully!")
print(f"Logistic Regression Best Params: {best_params}")