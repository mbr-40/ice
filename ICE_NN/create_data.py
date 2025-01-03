import csv
import random


def create_synthetic_diesel_data(num_rows=2000, output_file='diesel_engine_performance.csv'):
    """
    Generate a synthetic diesel engine dataset with 5 features and 1 numeric target (Engine_Performance).
    The CSV will have the following columns:
        1. Engine_Speed (rpm)
        2. Fuel_Flow (g/s)
        3. Intake_Manifold_Pressure (bar)
        4. Coolant_Temp (°C)
        5. Ambient_Temp (°C)
        6. Engine_Performance (numeric)

    :param num_rows: Number of data rows to generate.
    :param output_file: Name of the CSV file to create.
    """

    # Define plausible ranges for each feature
    engine_speed_range = (600, 3000)  # rpm
    fuel_flow_range = (10.0, 50.0)  # g/s
    intake_pressure_range = (1.0, 3.0)  # bar
    coolant_temp_range = (70, 110)  # °C
    ambient_temp_range = (10, 40)  # °C

    # Open the CSV file for writing
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow([
            "Engine_Speed",
            "Fuel_Flow",
            "Intake_Manifold_Pressure",
            "Coolant_Temp",
            "Ambient_Temp",
            "Engine_Performance"
        ])

        # Generate synthetic data rows
        for _ in range(num_rows):
            engine_speed = random.randint(*engine_speed_range)
            fuel_flow = random.uniform(*fuel_flow_range)
            intake_press = random.uniform(*intake_pressure_range)
            coolant_temp = random.randint(*coolant_temp_range)
            ambient_temp = random.randint(*ambient_temp_range)

            # Create a synthetic "Engine_Performance" measure
            # This is just an example formula combining the features + some noise
            engine_performance = (
                    0.02 * engine_speed
                    + 1.5 * fuel_flow
                    + 50.0 * intake_press
                    + 0.1 * coolant_temp
                    - 0.2 * ambient_temp
                    + random.uniform(-10, 10)  # random noise
            )

            # Write a row to the CSV
            writer.writerow([
                engine_speed,
                round(fuel_flow, 2),
                round(intake_press, 2),
                coolant_temp,
                ambient_temp,
                round(engine_performance, 2)
            ])


if __name__ == "__main__":
    # Generate 2000 rows by default
    create_synthetic_diesel_data(num_rows=20000, output_file='diesel_engine_performance.csv')
    print("Synthetic diesel engine data has been generated and saved to 'diesel_engine_performance.csv'.")
