import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("insurance.csv")

def evaluate_model():
    # Fill any NaN values in 'smoker' column with 'no' as a default before mapping
    data['smoker'] = data['smoker'].fillna('no')
    # Convert 'smoker' column to numerical representation
    data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

    # Define features and target
    X = data[['age', 'bmi', 'smoker']]
    y = data['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple Linear Regression model (since no model was found)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'health_model.pkl')

    # Load trained model (now it should exist)
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

