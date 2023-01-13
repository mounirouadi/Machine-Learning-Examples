#importing the dependencies
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report

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

#KNN using scikit learn
lb=LabelBinarizer()
y_train_binarized=lb.fit_transform(y_train)

K=3
clf=KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))
prediction_binarized = clf.predict(np.array([155, 70]).reshape(1, -1))[0]
predicted_label = lb.inverse_transform(prediction_binarized)
print(predicted_label)

X_test = np.array([[168, 65],
                   [180, 96],
                   [160, 52],
                   [169, 67]])
y_test = ['male', 'male', 'female', 'female']

y_test_binarized = lb.transform(y_test)
print("binarized labels :%s"%y_test_binarized.T[0])
predictions_binarized = clf.predict(X_test)
print("binarized predictions: %s"%predictions_binarized)
print("predicted labels: %s"%lb.inverse_transform(predictions_binarized))
print("accuracy : %s"%accuracy_score(y_test_binarized,predictions_binarized))
print("precision : %s"%precision_score(y_test_binarized,predictions_binarized))
print("recall : %s"%recall_score(y_test_binarized,predictions_binarized))
print("F1 score : %s"%f1_score(y_test_binarized,predictions_binarized))
print("Matthews correlation coefficient: %s "%matthews_corrcoef(y_test_binarized,predictions_binarized))
print("classification report :\n%s"%classification_report(y_test_binarized,predictions_binarized, target_names="male", labels=[1]))
