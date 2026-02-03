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
from sklearn.multioutput import MultiOutputRegressor
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
y = np.array([
    [30, 2],
    [45, 3],
    [55, 4],
    [75, 5],
    [90, 6],
    [110, 7]
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    learning_rate='optimal',
    random_state=42
)

model = MultiOutputRegressor(sgd)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

new_house = np.array([[1600, 4]])
new_house_scaled = scaler.transform(new_house)

prediction = model.predict(new_house_scaled)

print("\nPredicted House Price (in lakhs):", round(prediction[0][0], 2))
print("Predicted Number of Occupants:", round(prediction[0][1]))


```

## Output:
<img width="634" height="90" alt="Screenshot 2026-02-03 130737" src="https://github.com/user-attachments/assets/0dda676c-a01f-46df-8826-ff12b49258f3" />
<img width="486" height="68" alt="Screenshot 2026-02-03 130751" src="https://github.com/user-attachments/assets/b0e24b2d-1939-4529-b6ed-68cc99c0cd41" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
