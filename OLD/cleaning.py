import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path, save_scaler=True):
    """
    Preprocesses real estate data for machine learning without loops.

    Args:
        data_path: Path to the CSV file containing the data.
        save_scaler: Boolean flag indicating whether to save the scaler object.

    Returns:
        A pandas DataFrame containing the preprocessed data.
    """

    # Read data
    data = pd.read_csv(data_path)

    # Define categorical and numerical columns directly
    categorical_feature = "subproperty_type"
    numerical_features = [
    'price',
    'total_area_sqm',
    'nbr_bedrooms',
    'latitude',
    'longitude',
    'fl_garden',
    'fl_terrace',
    'primary_energy_consumption_sqm'
    ]

    # One-hot encode categorical features with pd.get_dummies
    encoded_data = pd.get_dummies(data[[categorical_feature]], drop_first=True)

    # Create and fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(data[numerical_features])
    scaled_numerical_data = scaler.transform(data[numerical_features])

    # Detect and remove outliers
    outlier_threshold = 3  # Adjust this threshold as needed
    filtered_data = pd.DataFrame(scaled_numerical_data, columns=numerical_features)
    for col in numerical_features:
        z_scores = (data[col] - np.mean(data[col])) / np.std(data[col]) # Calculate z-scores using NumPy functions
        filtered_data = filtered_data[np.abs(z_scores) <= outlier_threshold]
        print(f"Removed {len(filtered_data) - len(filtered_data)} outliers from column {col}.")

    # Combine scaled numerical data and encoded categorical features
    preprocessed_data = pd.concat([filtered_data, encoded_data], axis=1)

    # Handling missing values (consider imputation techniques)
    preprocessed_data = preprocessed_data.dropna()

    # Save scaler if needed
    if save_scaler:
        joblib.dump(scaler, "scaler.pkl")

    return preprocessed_data, scaler


# Example usage
preprocessed_data,scaler = preprocess_data("data\\properties.csv")
print(preprocessed_data.info)
