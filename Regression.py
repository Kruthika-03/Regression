import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
%matplotlib inline
from sklearn.linear_model import LinearRegression,Ridge 
from sklearn.metrics import mean_squared_error

#Loading dataset
boston_data = datasets.load_boston()

#the dataset is explored
keys = boston_data.keys()
print(keys)

#display the parameters
print(boston_data.feature_names)

#set the dataframe
boston_df = pd.DataFrame(boston_data.data)
boston_df.head()

#display the dataset shape
print(boston_df.shape)

boston_df.columns = boston_data.feature_names
boston_df.head()

#the target values are displayed
y = boston_data.target
print(y)

import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
sns.set(style='whitegrid', context='notebook')
features_plot = boston_data.feature_names

sns.pairplot(boston_df[features_plot], size=2.0);
plt.tight_layout()
plt.show()

#initialize the variables by preprocessing
X = boston_df.values
y = y

#select the train and test portions
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train shape -> {}".format(X_train.shape))
print("y_train shape -> {}".format(y_train.shape))
print("X_test shape -> {}".format(X_test.shape))
print("y_test shape -> {}".format(y_test.shape))

#linera regression on boston dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#display the plot without regularization
pred = regressor.predict(X_test)
plt.scatter(y_test, pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], c='r', lw=2)
plt.show()

#regression score
print(regressor.score(X_test, y_test))

#regularization is applied using ridge
model = Ridge(alpha = 9000)
model.fit(X_train, y_train)

#print the intercept and slope
predictionCV = model.predict(X_test)
predictionTestSet = model.predict(X_test)
errorCV = mean_squared_error(y_test, predictionCV)
errorTestSet = mean_squared_error(y_test, predictionTestSet)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

#linear regression with regularization
plt.scatter(y_test, predictionCV, c = 'green')
plt.xlabel("Parameters")
plt.ylabel("Price of Boston Housing")
plt.title("Predicted value VS True value: Linear Regression with regularisation")
plt.show()

#the intercept and slope are printed
linear=LinearRegression()
linear.fit(X_train, y_train)
linearpredictionCV = linear.predict(X_test)
linearpredictionTestSet = linear.predict(X_test)
errorCV = mean_squared_error(y_test, linearpredictionCV)
errorTestSet = mean_squared_error(y_test, linearpredictionTestSet)
print('intercept:', linear.intercept_)
print('slope:', linear.coef_)
