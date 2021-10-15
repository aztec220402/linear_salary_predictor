
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# linear regression on the training set
LR = LinearRegression()
LR.fit(X_train, y_train)

joblib.dump(LR, "LR1.pkl")

# making predictions
y_pred = LR.predict(X_test)
print(X_test)

# training set results
# plt.scatter(X_train, y_train, color='blue')
# plt.plot(X_train, LR.predict(X_train), color='red')

# # test set results
# plt.scatter(X_test, y_test, color='blue')
# plt.plot(X_train, LR.predict(X_train), color='red')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
