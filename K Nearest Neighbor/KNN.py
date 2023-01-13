#importing the dependencies
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


#training data
X_train = np.array([[158, 64],
           [170, 86],
           [183, 84],
           [191, 80],
           [155, 49],
           [163, 59],
           [180, 67],
           [158, 54],
           [170, 67]])
y_train = ['male','male','male','male','female','female','female','female','female']

x = np.array([155, 70])

#Euclidean distance
distances= np.sqrt(np.sum((X_train - x)**2, axis=1))

#sorting the distances and return their indexes
nearest_neighbor_indices = distances.argsort()[:3]

#find the nearest class for x
nearest_neighbor_genders = np.take(y_train,nearest_neighbor_indices)
b=Counter(nearest_neighbor_genders)
print(b.most_common(1)[0][0])


plt.figure()
plt.title("Human Heights and Weights by Sex")
plt.xlabel("Height in cm")
plt.ylabel("Weight in kg")

for i,x in enumerate(X_train):
    plt.scatter(x[0], x[1], c="k", marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()
