"""
house_price_model.py

Reusable functions for data cleaning, feature engineering,
model training, and evaluation.

Author: Naresh Kumar
"""

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def remove_outliers_zscore(df: pd.DataFrame, columns, threshold: float = 3.0):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(df[columns]))
    return df[(z_scores < threshold).all(axis=1)]


def engineer_features(df: pd.DataFrame):
    """
    Create interaction features.
    """
    df = df.copy()

    df["Income_Rooms_Interaction"] = (
        df["Avg. Area Income"] * df["Avg. Area Number of Rooms"]
    )

    df["Age_Income_Interaction"] = (
        df["Avg. Area House Age"] * df["Avg. Area Income"]
    )

    return df


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions, squared=False),
        "R2": r2_score(y_test, predictions),
    }

    return metrics

