#importing dependincies
import numpy as np
from sklearn.linear_model import LinearRegression

#the training data
X = [[1, 6, 2],
     [1, 8, 1],
     [1, 10, 0],
     [1, 14, 2],
     [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]
#solving for Beta with mathematical formula
print("Using the mathematical formula:\n%s"%np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y)))
#solving for Beta with numpy function
print("Using Numpy Least sqares function:\n%s"%np.linalg.lstsq(X,y,rcond=None)[0])
