import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy 
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor # NEW
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression # OLD
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = [
        'total_area_sqm',
        'nbr_bedrooms',
        'latitude',
        'longitude',
        'zip_code',
        'primary_energy_consumption_sqm',
        'surface_land_sqm',
        ]
    fl_features = [
        'fl_garden',
        'fl_terrace',
        'fl_swimming_pool',
        'fl_floodzone'
        ]
    cat_features = [
        'property_type'
        ]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model using HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Test R² score: {test_score}")
    print(f"Train R² score: {train_score}")
    
    # Prédire les valeurs
    y_pred = model.predict(X_test)
    
    # Calculer les erreurs résiduelles
    residuals = y_test - y_pred

    # Générer les quantiles théoriques de la distribution normale
    qq_norm = numpy.linspace(0, 1, len(residuals))

    # Calculer les quantiles des erreurs résiduelles
    qq_res = numpy.quantile(residuals, numpy.linspace(0, 1, len(residuals)))

    # Créer le graphique
    def create_circle(x, y, radius, color, edgecolor='white', linewidth=1):
        """Creates a circle patch object for plotting."""
        return plt.Circle((x, y), radius, color=color, edgecolor=edgecolor, linewidth=linewidth)

    # Define circle properties
    circle_radius = 0.005
    circle_color = 'blue'

    # Create circles and add them to the plot
    circles = [create_circle(x, y, circle_radius, circle_color) for x, y in zip(qq_norm, qq_res)]
    fig, ax = plt.subplots()
    for circle in circles:
        ax.add_artist(circle)

    # Calculate theoretical quantiles (alternative method)
    sorted_x = numpy.sort(qq_norm)
    theoretical_quantiles = numpy.linspace(0, 1, len(sorted_x))

    # Add theoretical quantile line (red)
    ax.plot(theoretical_quantiles, sorted_x, color='red', linestyle='dashed', label='Theoretical Quantiles')

    # Remove default axes and add custom labels
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Residual error quantiles")
    
    # Add legend (optional)
    # ax.legend()

    # Adjust axis limits based on data spread
    ax.set_xlim(0, 1)
    ax.set_ylim(min(qq_res) - 0.1, max(qq_res) + 0.1)

    # Use grid styling similar to ggplot2 (optional)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    train()