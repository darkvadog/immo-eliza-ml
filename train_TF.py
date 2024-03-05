import joblib
import pandas as pd
import tensorflow as tf  # Import the required library
import GradientBoostedDecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression # OLD
from sklearn.metrics import r2_score
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

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features + cat_features].reset_index(drop=True)
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features + cat_features].reset_index(drop=True)
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model using HistGradientBoostingRegressor
    model = GradientBoostedDecisionTreeRegressor()

    # Define feature columns
    feature_columns = []
    for col in num_features + fl_features:
        feature_columns.append(tf.feature_column.numeric_column(col))
    for col in cat_features:
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
            col, data[col].unique()))

    # Create training and evaluation datasets
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=X_train,
        y=y_train,
        batch_size=32,  # Adjust batch size as needed
        num_epochs=None,  # Set epochs for training
        shuffle=True,
        target_key="price"
    )
    eval_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=X_test,
        y=y_test,
        batch_size=32,  # Adjust batch size as needed
        num_epochs=1,  # Evaluate on a single epoch
        shuffle=False,
        target_key="price"
    )

    # Train the model
    model.train(input_fn=train_input_fn, steps=1000)  # Adjust steps for training

    # Evaluate the model
    eval_result = model.evaluate(input_fn=eval_input_fn)
    print(f"Train R² score: {eval_result['average_loss']}")  # R2 score not directly available
    print(f"Test R² score: {eval_result['average_loss']}")  # R2 score not directly available

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