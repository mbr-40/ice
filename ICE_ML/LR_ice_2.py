import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data (you only need to do this once)
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

# Split data (you only need to train the model once)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the Logistic Regression model (do this only once)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


# --- Prediction with New Data ---

def predict_performance(new_data_array):
    """Predicts diesel engine performance for a new data array.

    Args:
        new_data_array: A NumPy array or list of lists representing the new data.
                        Should have the same number of features as the training data.

    Returns:
        A list of predicted performance categories (strings).
        Returns None if there's an error in the input data format.
    """
    try:
        new_data = np.array(new_data_array)  # Convert to NumPy array for consistency

        # Check if the input array has the correct number of features
        if new_data.shape[1] != X.shape[1]:
            print(f"Error: Input data must have {X.shape[1]} features.")
            return None

        new_data_scaled = scaler.transform(new_data)  # Scale the new data using the SAME scaler
        predictions_numerical = model.predict(new_data_scaled)
        predictions_categorical = le.inverse_transform(predictions_numerical)  # Decode predictions
        return predictions_categorical.tolist()  # return as a list
    except ValueError as e:
        print(f"Error processing input data: {e}")
        return None
    except IndexError as e:
        print(f"Error processing input data: {e}. Check if the array has the correct dimensions.")
        return None


# Example usage:
new_data = np.array([
    [2000, 12, 450, 30, 60],  # Example 1: Best
    [1500, 18, 480, 22, 55],  # Example 2: Good
    [1000, 25, 550, 15, 45],  # Example 3: Bad
    [2500, 10, 380, 35, 70]  # Example 4: Best
])

predictions = predict_performance(new_data)

if predictions is not None:
    for i, prediction in enumerate(predictions):
        print(f"Example {i + 1}: Predicted Performance: {prediction}")

# Example with a single data point
new_data_single = np.array([[2200, 15, 420, 28, 62]])
predictions_single = predict_performance(new_data_single)
if predictions_single is not None:
    print(f"Single Example: Predicted Performance: {predictions_single[0]}")

# Example with a list of lists
new_data_list = [[1500, 14, 430, 29, 61]]
predictions_list = predict_performance(new_data_list)
if predictions_list is not None:
    print(f"List Example: Predicted Performance: {predictions_list[0]}")

# Example with wrong number of features
new_data_wrong = np.array([[1900, 14, 430, 29]])
predictions_wrong = predict_performance(new_data_wrong)