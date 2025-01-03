import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1) LOAD THE DATA
data = pd.read_csv('diesel_engine_performance.csv')

# 2) SEPARATE FEATURES AND TARGET
#    Our features are the first 5 columns, and the target is "Engine_Performance".
feature_cols = [
    "Engine_Speed",
    "Fuel_Flow",
    "Intake_Manifold_Pressure",
    "Coolant_Temp",
    "Ambient_Temp"
]
X = data[feature_cols]
y = data["Engine_Performance"]

# 3) TRAIN/TEST SPLIT
#    We'll use 80% of data for training, 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4) (OPTIONAL) SCALE FEATURES
#    Scaling is often helpful for neural networks, especially with varying feature ranges.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) BUILD A SIMPLE KERAS MODEL
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(5,)),    # We have 5 features
    tf.keras.layers.Dense(64, activation='relu'),    # Hidden layer with 64 neurons
    tf.keras.layers.Dense(32, activation='relu'),    # Another hidden layer
    tf.keras.layers.Dense(1)                         # Output layer (1 neuron for regression)
])

# 6) COMPILE THE MODEL
#    - Loss function: MSE (mean squared error) for regression
#    - Optimizer: Adam
#    - Metrics: MAE (mean absolute error) just for readability
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 7) TRAIN THE MODEL
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,  # 20% of the training set will be used for validation
    epochs=5,             # Number of passes through the training data
    batch_size=4,          # Adjust batch size if you want
    verbose=1
)

# 8) EVALUATE ON TEST SET
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test MSE: {loss:.2f}")
print(f"Test MAE: {mae:.2f}")

# 9) PREDICT ON A NEW SAMPLE
#    Suppose we have a new diesel engine data point:
#    Engine_Speed=1200 rpm, Fuel_Flow=15 g/s, Intake_Manifold_Pressure=1.4 bar,
#    Coolant_Temp=75 °C, Ambient_Temp=25 °C

# 2715,16.55,2.66,108,15,209.91
# 1661,46.5,2.96,109,16,267.51
# 1123,32.32,1.56,103,26,158.05

new_sample = np.array([[1661,46.5,2.96,109,16]])

# Scale it with the SAME scaler used for training
new_sample_scaled = scaler.transform(new_sample)

# Predict
predicted_performance = model.predict(new_sample_scaled)
print(f"Predicted Engine Performance: {predicted_performance[0][0]:.2f}")
