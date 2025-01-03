import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ... (load data and separate X and y as before)

# One-hot encode the target variable
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') #sparse=False for dense array
y_encoded = ohe.fit_transform(y.values.reshape(-1, 1))

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Create and train the Logistic Regression model (multiclass)
model = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr') #ovr: One-vs-rest strategy
model.fit(X_train, y_train)

def predict_performance(new_data_array):
    try:
        new_data = np.array(new_data_array)
        if new_data.shape[1] != X.shape[1]:
            print(f"Error: Input data must have {X.shape[1]} features.")
            return None
        new_data_scaled = scaler.transform(new_data)
        predictions_encoded = model.predict(new_data_scaled)
        predictions_numerical = np.argmax(predictions_encoded, axis=1) #get the index of the highest probability
        #Reverse the one hot encoding
        original_labels = ohe.inverse_transform(predictions_encoded)
        return original_labels.flatten().tolist() #return as a list
    except ValueError as e:
        print(f"Error processing input data: {e}")
        return None
    except IndexError as e:
        print(f"Error processing input data: {e}. Check if the array has the correct dimensions.")
        return None

#Example usage
new_data = np.array([
    [2000, 12, 450, 30, 60],  # Example 1: Best
    [1500, 18, 480, 22, 55],  # Example 2: Good
    [1000, 25, 550, 15, 45],  # Example 3: Bad
    [2500, 10, 380, 35, 70]   #Example 4: Best
])

predictions = predict_performance(new_data)

if predictions is not None:
    for i, prediction in enumerate(predictions):
        print(f"Example {i+1}: Predicted Performance: {prediction}")