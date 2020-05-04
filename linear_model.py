# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
data = pd.read_excel('ANZ synthesised transaction dataset.xlsx')
X = data.iloc[:,12:14].values # Gender and age
y = data.iloc[:,10:11].values # Salary

# Encoding Categorical(String) data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Male coulmn and age column
X= X[:, 1:]

# splitting in training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting mul linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set results
y_pred = regressor.predict(X_test)

# Calculating MSE
from sklearn.metrics import mean_squared_error
ms = mean_squared_error(y_test, y_pred)

# Plottting Age vs Salary 
plt.scatter(X_train[:,1:], y_train)
plt.plot(X_train[:,1:], regressor.predict(X_train), color='red')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Training Dataset')
plt.show()

# Plottting Age vs Salary on test dataset
plt.scatter(X_test[:,1:], y_test)
plt.plot(X_test[:,1:],regressor.predict(X_test), color='red')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Test Dataset')
plt.show()

# ---- Salary vs amount spent ----
X_amt = data.iloc[:,17:18].values

# splitting in training and test set
from sklearn.model_selection import train_test_split
X_amt_train, X_amt_test, y_train, y_test = train_test_split(X_amt, y, test_size = 0.2, random_state =0)

# Fitting mul linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_amt_train, y_train)

# Calculating MSE
from sklearn.metrics import mean_squared_error
ms = mean_squared_error(y_test, y_pred)

# Predicting Test set results
y_pred = regressor.predict(X_amt_test)

# Plottting Amount  vs Salary 
plt.scatter(X_amt_train, y_train)
plt.plot(X_amt_train,regressor.predict(X_amt_train), color='red')
plt.xlabel('Amount')
plt.ylabel('Salary')
plt.xlim(0,10000)
plt.ylim(0, 300000)
plt.title('Training Dataset')
plt.show()

# Plottting Amount  vs Salary Test dataset
plt.scatter(X_amt_test, y_test)
plt.plot(X_amt_test, y_pred, color='red')
plt.xlabel('Amount')
plt.ylabel('Salary')
plt.xlim(0,10000)
plt.ylim(0, 300000)
plt.title('Training Dataset')
plt.show()