# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Input & Dataset Preparation
  Collect the dataset containing house features (area in sq.ft and number of rooms) as input X, and define two target variables: house price and number of occupants.

2.Data Splitting & Feature Scaling
  Split the dataset into training and testing sets using train_test_split, then standardize the input features using StandardScaler to improve SGD convergence.

3.Model Training using SGD Regressor
  Train two separate SGDRegressor models—one for predicting house price and another for predicting occupants—using the scaled training data.

4.Prediction & Evaluation
  Predict house price and occupants for test data and new inputs, and evaluate model performance using Mean Squared Error (MSE). 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: S.Manigandan
RegisterNumber:  25004934
*/
```
```
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X = np.array([
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
    [2000, 5]
])

y_price = np.array([30, 40, 45, 60, 75, 90])      
y_occupants = np.array([2, 3, 3, 4, 5, 6])

X_train, X_test, y_price_train, y_price_test, y_occ_train, y_occ_test = train_test_split(
    X, y_price, y_occupants, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

price_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
occupant_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)

price_model.fit(X_train_scaled, y_price_train)
occupant_model.fit(X_train_scaled, y_occ_train)

price_pred = price_model.predict(X_test_scaled)
occ_pred = occupant_model.predict(X_test_scaled)


print("House Price MSE:", mean_squared_error(y_price_test, price_pred))
print("Occupants MSE:", mean_squared_error(y_occ_test, occ_pred))

new_house = np.array([[1600, 4]])
new_house_scaled = scaler.transform(new_house)

predicted_price = price_model.predict(new_house_scaled)
predicted_occupants = occupant_model.predict(new_house_scaled)

print("\nPrediction for New House:")
print("Predicted House Price (in lakhs):", predicted_price[0])
print("Predicted Number of Occupants:", round(predicted_occupants[0]))


```

## Output:
<img width="406" height="62" alt="Screenshot 2026-02-03 140951" src="https://github.com/user-attachments/assets/ec7842e4-4841-445e-a781-f5c064cd85bd" />
<img width="534" height="101" alt="Screenshot 2026-02-03 141002" src="https://github.com/user-attachments/assets/acb5280e-544a-4abe-bb36-6ff7e67a91d2" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
