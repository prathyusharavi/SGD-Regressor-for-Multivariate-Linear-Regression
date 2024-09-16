# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California housing dataset and preprocess the data.
2. Split the dataset into training and testing sets.
3. Scale the features using `StandardScaler` and train the `SGDRegressor` using `MultiOutputRegressor`.
4. Evaluate the model using metrics like mean squared error and R-squared score. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MOPURI ANKITHA
RegisterNumber: 212223040117
*/
```
```
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import SGDRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputRegressor

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df["HousingPrice"]=dataset.target
df.head()
```
## Output:
![image](https://github.com/user-attachments/assets/eb0deeeb-9bb3-4297-ac0c-e0cdd32c52e6)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
