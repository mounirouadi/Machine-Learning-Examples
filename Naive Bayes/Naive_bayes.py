#importing dependencies
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=31)
lr = LogisticRegression()
nb = GaussianNB()

lr_scores = []
nb_scores = []

train_sizes = range(10,len(X_train), 25)

#Fit and score the models for different train set sizes
for train_size in train_sizes:
  X_train_subset = X_train[:train_size]
  y_train_subset = y_train[:train_size]
  lr.fit(X_train_subset, y_train_subset)
  lr_scores.append(lr.score(X_test, y_test))
  nb.fit(X_train_subset, y_train_subset)
  nb_scores.append(nb.score(X_test, y_test))

#Plot the results
plt.plot(train_sizes, lr_scores, label='Logistic Regression')
plt.plot(train_sizes, nb_scores, label='Naive Bayes')
plt.xlabel('Train set size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
