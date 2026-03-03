# ==============================
# 1. Import Libraries
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================
# 2. Data Collection
# ==============================

df = pd.read_csv("insurance.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nInfo:\n")
print(df.info())
print("\nStatistical Summary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# ==============================
# 3. Data Visualization
# ==============================

sns.set()

plt.figure()
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure()
sns.countplot(x='sex', data=df)
plt.title("Sex Distribution")
plt.show()

plt.figure()
sns.histplot(df['bmi'], kde=True)
plt.title("BMI Distribution")
plt.show()

plt.figure()
sns.countplot(x='smoker', data=df)
plt.title("Smoker Distribution")
plt.show()

plt.figure()
sns.countplot(x='region', data=df)
plt.title("Region Distribution")
plt.show()

plt.figure()
sns.histplot(df['charges'], kde=True)
plt.title("Charges Distribution")
plt.show()

# ==============================
# 4. Data Preprocessing
# ==============================

# Encoding categorical columns
df.replace({
    'sex': {'male': 0, 'female': 1},
    'smoker': {'no': 0, 'yes': 1},
    'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
}, inplace=True)

# ==============================
# 5. Split Features & Target
# ==============================

X = df.drop(columns='charges', axis=1)
Y = df['charges']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==============================
# 6. Model Training
# ==============================

model = LinearRegression()
model.fit(X_train, Y_train)

# ==============================
# 7. Model Evaluation
# ==============================

# Training Data Prediction
train_pred = model.predict(X_train)

r2_train = r2_score(Y_train, train_pred)
mae_train = mean_absolute_error(Y_train, train_pred)
rmse_train = np.sqrt(mean_squared_error(Y_train, train_pred))

print("\n===== Training Performance =====")
print("R2 Score :", r2_train)
print("MAE      :", mae_train)
print("RMSE     :", rmse_train)

# Testing Data Prediction
test_pred = model.predict(X_test)

r2_test = r2_score(Y_test, test_pred)
mae_test = mean_absolute_error(Y_test, test_pred)
rmse_test = np.sqrt(mean_squared_error(Y_test, test_pred))

print("\n===== Testing Performance =====")
print("R2 Score :", r2_test)
print("MAE      :", mae_test)
print("RMSE     :", rmse_test)

# ==============================
# 8. Making a Prediction
# ==============================

# Example Input:
# (age, sex, bmi, children, smoker, region)
# sex: male=0 female=1
# smoker: no=0 yes=1
# region: southeast=0 southwest=1 northeast=2 northwest=3

input_data = (30, 0, 25.3, 1, 0, 2)

input_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_array)

print("\nPredicted Insurance Cost: USD", prediction[0])
