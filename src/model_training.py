# model_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_model():

    # Load dataset
    data = pd.read_csv("insurance.csv")

    # Encode smoker column (yes=1, no=0)
    data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

    # Define features and target
    X = data[['age', 'bmi', 'smoker']]
    y = data['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

print("Model training completed successfully.")

