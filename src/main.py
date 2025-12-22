"""
main.py

Entry point for running the house price prediction pipeline.

Author: Naresh Kumar
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from house_price_model import (
    remove_outliers_zscore,
    engineer_features,
    train_linear_regression,
    evaluate_model
)


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv("data/USA_Housing.csv")

    # Drop non-numeric column
    df = df.drop(columns=["Address"])

    numeric_cols = df.columns.drop("Price")

    # -----------------------------
    # Clean data
    # -----------------------------
    df_cleaned = remove_outliers_zscore(df, numeric_cols)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df_features = engineer_features(df_cleaned)

    selected_features = [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Area Population",
        "Income_Rooms_Interaction",
        "Age_Income_Interaction"
    ]

    X = df_features[selected_features]
    y = df_features["Price"]

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Train model
    # -----------------------------
    model = train_linear_regression(X_train, y_train)

    # -----------------------------
    # Evaluate model
    # -----------------------------
    metrics = evaluate_model(model, X_test, y_test)

    print("\nModel Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:,.2f}")


if __name__ == "__main__":
    main()
