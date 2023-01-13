#importing the dependencies
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

#loading and splitiing data
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

#defining the scalers and scaling the data
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
X_test = X_scaler.fit_transform(X_test)
y_test = y_scaler.fit_transform(y_test.reshape(-1, 1))

#creating SGD regression instance
regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print("Cross validation r-squared scores: \n%s"%scores)
print("Average cross validation r-squared score: %s"%np.mean(scores))
regressor.fit(X_train, y_train)
print('Test set r-squared scores: %s'%regressor.score(X_test, y_test))
