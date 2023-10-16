# Bike Rentals Prediction

This Python script analyzes bike rental data and predicts the number of bike rentals using different regression models. It includes data preprocessing, model training, and evaluation of the models.

The script uses various machine learning models for regression, including Linear Regression, Decision Tree Regression, and Random Forest Regression.

## Usage

1. **Requirements:**

   Make sure you have the following dependencies installed:
   - Python 3.x
   - pandas
   - numpy
   - scikit-learn

   The bike rental data is expected to be in a CSV file ('data/hour.csv').

2. **Running the Script:**

   To perform the analysis and predictions on bike rental data, run the script.

   ```
   python rental_prediction.py
   ```

3. **Data Preprocessing:**

The script performs data preprocessing tasks, including splitting the data into training and test sets, selecting predictor columns, and preparing the data for model training.

4. **Regression Models:**

- **Linear Regression:** This model is trained and evaluated for bike rental predictions.
- **Decision Tree Regression:** Decision tree models are trained with different `min_samples_leaf` values (5 and 2), and their performance is evaluated.
- **Random Forest Regression:** A random forest model is trained with a `min_samples_leaf` value of 5, and its performance is evaluated.

5. **Model Evaluation:**

The script calculates Mean Squared Error (MSE) for each model to assess prediction accuracy.

## Linting and Pylint

Linting is the process of checking your code for potential issues, style violations, and errors. To ensure that the code follows good coding practices, we can use Pylint, a Python code linter, by running the following command:

```
pylint rental_prediction.py
```

The script rental_prediction.py received a linting score of 9.09.

![Pylint result](https://github.com/gabrielaact/mlops/blob/main/Python%20Essentials%20for%20MLOps/Project%2001/images/pylint.png)

## References
[Dataquest - Predicting Bike Rentals](https://github.com/dataquestio/solutions/blob/master/Mission213Solution.ipynb)