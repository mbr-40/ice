import pandas as pd
import numpy as np
import random

# Number of data points
n_samples = 1000

# Feature ranges (adjust these based on realistic diesel engine values)
feature_ranges = {
    "Engine_Speed_RPM": (800, 3000),
    "Fuel_Consumption_LPH": (5, 30),
    "Exhaust_Temperature_C": (200, 600),
    "Boost_Pressure_PSI": (10, 40),
    "Oil_Pressure_PSI": (40, 80)
}

# Possible target categories
target_categories = ["Best", "Good", "Bad"]

# Generate data
data = []
for _ in range(n_samples):
    features = {}
    for feature_name, (min_val, max_val) in feature_ranges.items():
        features[feature_name] = np.random.uniform(min_val, max_val)

    # Simple rule-based target generation (you'll likely want a more sophisticated approach)
    if (features["Fuel_Consumption_LPH"] < 15 and features["Exhaust_Temperature_C"] < 400 and features["Boost_Pressure_PSI"] > 25):
        target = "Best"
    elif (features["Fuel_Consumption_LPH"] < 20 and features["Exhaust_Temperature_C"] < 500):
        target = "Good"
    else:
        target = "Bad"

    features["Performance"] = target
    data.append(features)

# Create Pandas DataFrame
df = pd.DataFrame(data)

# Introduce some random noise (important for realistic data)
for feature in feature_ranges:
    df[feature] = df[feature] + np.random.normal(0, df[feature]*0.05, n_samples) #adjust the 0.05 to control the noise level.

# Save to CSV
df.to_csv("diesel_engine_data.csv", index=False)

print(f"Generated {n_samples} samples and saved to diesel_engine_data.csv")


# --- Optional: Basic EDA (Exploratory Data Analysis) ---
import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot to visualize relationships between features and target
sns.pairplot(df, hue="Performance")
plt.savefig("pairplot.png") #save the plot
plt.show()

# Correlation Matrix
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.savefig("correlation_matrix.png") #save the plot
plt.show()

# Count of each target class
print(df["Performance"].value_counts())