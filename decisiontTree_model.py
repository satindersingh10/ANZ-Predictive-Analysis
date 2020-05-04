# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing data
data = pd.read_excel('ANZ synthesised transaction dataset.xlsx')

# --- Balance vs Age ---
y = data.iloc[:,10:11].values # Balance
X_age = data.iloc[:, 13:14].values # Age

# Spliting the dataset
from sklearn.model_selection import train_test_split
X_age_train, X_age_test, y_train, y_test = train_test_split(X_age, y, test_size =0.2, random_state =0)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_age_train,y_train)

# predicting a new result
y_pred = regressor.predict(X_age_test)

# Calculating MSE
from sklearn.metrics import mean_squared_error
ms = mean_squared_error(y_test, y_pred)


# Visualizing Decision Tree Regression in High resolution
X_grid = np.arange(min(X_age_train), max(X_age_train), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X_age_train, y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Salary vs Age (Decision Tree Regression Training Dataset)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Visualizing Training Dataset
X_grid1 = np.arange(min(X_age_test), max(X_age_test), 0.001)
X_grid1 = X_grid1.reshape(len(X_grid1), 1)
plt.scatter(X_age_test, y_test, color = 'red')
plt.plot(X_grid1, regressor.predict(X_grid1), color = 'blue')
plt.title('Salary vs Age (Decision Tree Regression Testing Dataset)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()






