# MODELSCARD for House Price Prediction
## Project context
This project predicts house prices in a specific region based on various property features. The goal is to develop a model that can accurately estimate the market value of a house given its characteristics.

### Data:
- Input dataset: properties.csv containing information about individual houses;
- Target variable: price: the actual selling price of the house.

### Features:
- Numerical features: total area, number of bedrooms, latitude, longitude, zip code, primary energy consumption, surface land area;
- Boolean features: presence of garden, terrace, swimming pool, and flood zone;
- Categorical feature: property type.

## Model details
### Models tested:
- LinearRegression (baseline);
- HistGradientBoostingRegressor (final chosen).

### Performance
#### Metrics:
- Train R² score: 0.82 (represents 82% of the variance in the target variable explained by the model);
- Test R² score: 0.78 (generalizes well to unseen data).

#### Visualizations:
Consider including visualizations of feature importance, residual plots, or other relevant diagnostics.

### Limitations
- The model is trained on a specific dataset and may not generalize well to other geographical regions or market conditions;
- The model relies on the accuracy and completeness of the input data. Errors or missing values in the features can affect the predictions;
- The R² score indicates a good fit, but there may still be outliers or individual predictions with significant errors.

## Usage
### Dependencies:
pandas, sklearn, joblib

### Scripts:
- train.py: script to train the model and save artifacts;
- Prediction:
* Load the saved artifacts (artifacts.joblib) using joblib.load;
* Preprocess new data using the same imputer and one-hot encoder (imputer and enc objects);
* Make predictions using the trained model (model).

## Maintainers
Philippe Montel (philippe.montel75@gmail.com)

## Additional notes
Consider adding information about hyperparameter tuning, if applicable.