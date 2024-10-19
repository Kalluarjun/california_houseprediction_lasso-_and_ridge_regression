import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
df = fetch_california_housing()
#print(df) 
dataset= pd.DataFrame(df.data)
dataset.columns= df.feature_names
dataset["price"]= df.target
#print(dataset.head())
X= dataset.iloc[:, : -1] #or X= df.data and y= df.target
y= dataset.iloc[:, -1]
# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear regression for comparison
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
linear_regression_mse = -cross_val_score(linear_regressor, X, y, scoring='neg_mean_squared_error', cv=5).mean()
print("Linear Regression MSE:", linear_regression_mse)

#Ridge regression
ridge= Ridge()
parameters= {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
ridge_regressor= GridSearchCV(ridge, parameters, scoring= 'neg_mean_squared_error', cv= 5)
ridge_regressor.fit(X, y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#lasso regression
lasso= Lasso()
parameters= {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor= GridSearchCV(lasso, parameters, scoring= 'neg_mean_squared_error', cv= 5)
lasso_regressor.fit(X, y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

prediction_lasso= lasso_regressor.predict(X_test)
prediction_ridge= ridge_regressor.predict(X_test)

sns.displot(y_test-prediction_lasso)
sns.displot(y_test- prediction_ridge)
plt.show()





