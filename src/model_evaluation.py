# model_evaluation.py

import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

def evaluate_model():

    # Load dataset
    data = pd.read_csv("../data/insurance.csv")

    # Encode smoker column (yes=1, no=0)
    data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

    # Define features and target
    X = data[['age', 'bmi', 'smoker']]
    y = data['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load trained model
    model = joblib.load("health_model.pkl")

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Model Performance Metrics")
    print("--------------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

if __name__ == "__main__":
    evaluate_model()

