
"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
from typing import ByteString
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

def hyper_accuracy(digits, para):

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    clf = svm.SVC(gamma=para)

    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)

    clf.fit(X_train, y_train)
    print("Train ", len(X_train))
    print("Val ", len(X_val))
    print("Test ", len(X_test))
    print("Train accuracy",clf.score(X_train, y_train))
    print("Val accuracy",clf.score(X_val, y_val))
    print("Test accuracy",clf.score(X_test, y_test))

def hyper(digits, para):

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    clf = svm.SVC(gamma=para)

    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=0.15, shuffle=False)

    clf.fit(X_train, y_train)

    predicted = clf.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, predicted)
    return accuracy

parames = [1e-7,1e-5,1e-3,0.01,0.1,1]
acc = []
for i in range(len(parames)):
    acc.append(hyper(digits, parames[i]))

print("Gamma    ","Accuracy")
for i in range(len(parames)):
    print(parames[i],"     ",acc[i])

plt.plot(parames, acc)
plt.xlabel('gamma')
plt.ylabel('accuracy')
#plt.show()

best_gamma = parames[acc.index(max(acc))]
print("optimal gamma ",best_gamma)

hyper_accuracy(digits, best_gamma)
#print("Train accuracy ",clf.score(X_train,y_train))
