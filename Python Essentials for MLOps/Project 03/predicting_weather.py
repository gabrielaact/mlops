"""
Bike Rentals Prediction

This Python script analyzes bike rental data and predicts the number of bike rentals for different 
regression models. It includes data preprocessing, model training, and evaluation of the models.

The script uses various machine learning models for regression, including Linear Regression, 
Decision Tree Regression, and Random Forest Regression.

Usage:
- Run this script to perform the analysis and predictions on bike rental data.

Requirements:
- Python 3.x
- pandas
- numpy
- scikit-learn

The bike rental data is expected to be in a CSV file ('data/hour.csv').

"""

import logging
import numpy
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def read_data(file_path):
    """
    Read data from a CSV file.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame or None: A DataFrame containing the data, or None if an error occurred.
    """
    try:
        return pandas.read_csv(file_path)
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        print("File not found!")
        return None

def assign_label(hour):
    """
    Assign time labels based on the hour of the day.

    Args:
    - hour (int): The hour of the day (0-23).

    Returns:
    - int: The time label (1-4).
    """
    if hour >= 0 and hour < 6:
        return 4
    if hour >= 6 and hour < 12:
        return 1
    if hour >= 12 and hour < 18:
        return 2
    return 3

def split_data(data, train_fraction=0.8):
    """
    Split the data into training and test sets.

    Args:
    - data (pd.DataFrame): The input DataFrame.
    - train_fraction (float): The fraction of data to use for training (default is 0.8).

    Returns:
    - pd.DataFrame, pd.DataFrame or None, None: The training and test DataFrames, 
    or None, None if an error occurred.
    """
    train, test = None, None
    try:
        train = data.sample(frac=train_fraction)
        test = data.loc[~data.index.isin(train.index)]
    except ValueError as e:
        logging.error("Error splitting data into training and test sets: %s", str(e))
    return train, test

def train_and_evaluate_regression_model(model, train_data, test_data, predictors, target_column):
    """
    Train a regression model and calculate Mean Squared Error.

    Args:
    - model: The regression model to train.
    - train_data (pd.DataFrame): The training data.
    - test_data (pd.DataFrame): The test data.
    - predictors (list): A list of predictor columns.
    - target_column (str): The target column.

    Returns:
    - numpy.ndarray, float or None, None: The model's predictions and Mean Squared Error, 
    or None, None if an error occurred.
    """
    predictions, mse = None, None
    try:
        model.fit(train_data[predictors], train_data[target_column])
        predictions = model.predict(test_data[predictors])
        mse = numpy.mean((predictions - test_data[target_column]) ** 2)
    except (ValueError, TypeError) as e:
        logging.error("Error training and evaluating the model: %s", str(e))
    return predictions, mse

bike_rentals = read_data("data/hour.csv")
train, test = split_data(bike_rentals)

if train is not None and test is not None:
    predictors = list(train.columns)
    predictors.remove("cnt")
    predictors.remove("casual")
    predictors.remove("registered")
    predictors.remove("dteday")

    reg = LinearRegression()
    linear_predictions, linear_regression_mse = train_and_evaluate_regression_model(
        reg, train, test, predictors, "cnt")

    if linear_predictions is not None and linear_regression_mse is not None:
        print("Linear Regression Predictions:", linear_predictions)
        print("Mean Squared Error (Linear Regression):", linear_regression_mse)

    reg = DecisionTreeRegressor(min_samples_leaf=5)
    dt_predictions_5, decision_tree_mse_5 = train_and_evaluate_regression_model(
        reg, train, test, predictors, "cnt")
    if dt_predictions_5 is not None and decision_tree_mse_5 is not None:
        print("Decision Tree Predictions (min_samples_leaf=5):", dt_predictions_5)
        print("Mean Squared Error (Decision Tree, min_samples_leaf=5):", decision_tree_mse_5)

    reg = DecisionTreeRegressor(min_samples_leaf=2)
    dt_predictions_2, decision_tree_mse_2 = train_and_evaluate_regression_model(
        reg, train, test, predictors, "cnt")
    if dt_predictions_2 is not None and decision_tree_mse_2 is not None:
        print("Decision Tree Predictions (min_samples_leaf=2):", dt_predictions_2)
        print("Mean Squared Error (Decision Tree, min_samples_leaf=2):", decision_tree_mse_2)

    reg = RandomForestRegressor(min_samples_leaf=5)
    rf_predictions, random_forest_mse = train_and_evaluate_regression_model(
        reg, train, test, predictors, "cnt")
    if rf_predictions is not None and random_forest_mse is not None:
        print("Random Forest Predictions:", rf_predictions)
        print("Mean Squared Error (Random Forest, min_samples_leaf=5):", random_forest_mse)
