#importing dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#training data
X_train = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
y_train = [7,9,13,17.5,18]

X_test = np.array([8,9,11,16,12]).reshape(-1,1)
y_test = [11,8.5,15,18,11]

#calculating the mean of X and y
x_bar = np.mean(X_train)
y_bar = np.mean(y_train)
print("the mean is :%.2f"%x_bar)

#calculating the variance and covariance
variance = ((X_train - x_bar) **2).sum() / (np.shape(X_train)[0] - 1)
#covariance = ((X - x_bar) * (y - y_bar)).sum() / (np.shape(X)[0] - 1)
covariance = (np.multiply((X_train - x_bar).transpose(),(y_train - y_bar)).sum() / (np.shape(X_train)[0] - 1))
print("the variance is: %.2f"%variance)
print("the covariance is: %.2f"%covariance)
print("another method for covariance : %.2f"%np.cov(X_train.transpose(),y_train)[0][1])

#create an instance of the estimator
model = LinearRegression()
model.fit(X_train,y_train) #fit the data on the training data

test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]
print("A 12 inch pizza should cost : $%.2f"%predicted_price)
print("Residual sum of squares: %.2f"%np.mean((model.predict(X_train) - y_train)**2))

#r_squared scoring
r_squared=model.score(X_test,y_test)
print('r_squared value: %.2f'%r_squared)
#plotting data
plt.figure()
plt.title("Pizza prices agains diameter")
plt.xlabel("Diameter in inches")
plt.ylabel("Price in dollars")
plt.plot(X_train,y_train,'k.') #k. for points (kx can be used for x marks too)
plt.plot(X_train, model.predict(X_train),color='r') #plotting the fitting line
plt.axis([0,25,0,25])
plt.grid(True) #adding lines to the plot
plt.show()
