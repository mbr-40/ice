import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
df = pd.read_csv("diesel_engine_data.csv")

# Separate features (X) and target (y)
X = df.drop("Performance", axis=1)
y = df["Performance"]

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=le.classes_))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# --- Optional: Feature Importance (simplified) ---
try:
    feature_importance = model.coef_[0]
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:")
    print(importance_df)
except:
    print("Could not calculate feature importance. Model was not fitted.")