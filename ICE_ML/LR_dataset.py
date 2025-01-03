import pandas as pd
import numpy as np
import random

# Number of data points
n_samples = 1000

# Feature ranges
feature_ranges = {
    "Engine_Speed_RPM": (800, 3000),
    "Fuel_Consumption_LPH": (5, 30),
    "Exhaust_Temperature_C": (200, 600),
    "Boost_Pressure_PSI": (10, 40),
    "Oil_Pressure_PSI": (40, 80)
}

# Coefficients for performance score calculation
coefficients = {
    "Engine_Speed_RPM": 0.02,
    "Fuel_Consumption_LPH": -1.5,
    "Exhaust_Temperature_C": -0.05,
    "Boost_Pressure_PSI": 2.0,
    "Oil_Pressure_PSI": 0.1
}

# Calculate min/max possible scores
min_possible_score = sum(coefficients[feature] * feature_ranges[feature][0] for feature in feature_ranges)
max_possible_score = sum(coefficients[feature] * feature_ranges[feature][1] for feature in feature_ranges)

# Generate data
data = []
for _ in range(n_samples):
    features = {}
    for feature_name, (min_val, max_val) in feature_ranges.items():
        features[feature_name] = random.randint(min_val, max_val)

    # Calculate Performance Score
    performance_score = sum(coefficients[feature] * features[feature] for feature in features)

    # Normalize performance score to 1-100
    performance_score_normalized = (performance_score - min_possible_score) / (max_possible_score - min_possible_score) * 99 + 1

    # Add integer noise and clip to 1-100 range
    noise = np.random.randint(-5, 6)
    performance_score_normalized += noise
    performance_score_normalized = np.clip(performance_score_normalized, 1, 100)

    features["Performance_Score"] = int(performance_score_normalized)
    data.append(features)

# Create Pandas DataFrame
df = pd.DataFrame(data)

# Introduce integer noise to features and clip to range
for feature in feature_ranges:
    df[feature] = df[feature] + [random.randint(-2, 2) for _ in range(n_samples)]
    df[feature] = np.clip(df[feature], feature_ranges[feature][0], feature_ranges[feature][1])

# Save to CSV
df.to_csv("diesel_engine_regression_data_1_100_int.csv", index=False)

print(f"Generated {n_samples} samples and saved to diesel_engine_regression_data_1_100_int.csv")