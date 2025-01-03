import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv("diesel_engine_regression_data_1_100_int.csv")

# Separate features (X) and target (y)
X = df.drop("Performance_Score", axis=1)
y = df["Performance_Score"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

#Predicting with new data
def predict_performance(new_data_array):
    try:
        new_data = np.array(new_data_array)
        if new_data.shape[1] != X.shape[1]:
            print(f"Error: Input data must have {X.shape[1]} features.")
            return None
        new_data_scaled = scaler.transform(new_data)
        predictions = model.predict(new_data_scaled)
        return predictions.tolist()
    except ValueError as e:
        print(f"Error processing input data: {e}")
        return None
    except IndexError as e:
        print(f"Error processing input data: {e}. Check if the array has the correct dimensions.")
        return None

new_data = np.array([
    [500, 12, 450, 30, 60],
    [1500, 18, 480, 22, 55],
    [1000, 25, 550, 15, 45],
    [2500, 10, 380, 35, 70]
])

predictions = predict_performance(new_data)

if predictions is not None:
    for i, prediction in enumerate(predictions):
        print(f"Example {i+1}: Predicted Performance Score: {prediction}")