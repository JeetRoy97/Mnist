from typing import ByteString
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import os
import shutil
from joblib import dump, load
import numpy as np
import math
from utils import create_splits, preprocess, report, classification_model



digits = datasets.load_digits()
X,Y = digits.images, digits.target

train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= round((1 - train_ratio),2), shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), shuffle=False)

#print(y_test.shape, y_val.shape)



mydir = "/home/jeet/MLOPs/Mnist/mnist/models_train/"
if os.path.exists(mydir):
    shutil.rmtree(mydir)

def validate(X_val, y_val, i, s, clf):
    predicted = clf.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, predicted)
    #print(accuracy)
    f1 = metrics.f1_score(y_val, predicted, average = 'macro' )
    candidate = {
        'acc_valid' : accuracy,
        'f1_valid' : f1,
        'gamma' : i,
        'split' : s,
        }
    return candidate

def classification_model(X_train, y_train, X_val, y_val, gamma, split,  model_path):
    #model_candidate = []
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)
    model_candidate = validate(X_val, y_val,gamma, split,clf) # validation function
    #print(model_candidate)
    output = model_path+"split_{}_gamma{}".format(split, gamma)
    os.makedirs(output)
    dump(clf, os.path.join(output,"model.joblib"))
    return model_candidate

def test(X_test, y_test, best_model_folder):
    clf = load(os.path.join(best_model_folder,"model.joblib"))
    X_test =  X_test.reshape((len(X_test), -1))
    predicted = clf.predict(X_test)
    #report(clf, y_test, predicted) # classification report function
    f1 = metrics.f1_score(y_test,predicted, average = 'macro')
    #print(confusion_matrix(y_test, predicted))

    return f1


parames = [1e-7,1e-5,1e-3,0.01,0.1,1]
#parames = [0.01,0.1]
f1_macro = []
checkpoint = []


split = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
for i in (split):
    os.makedirs(os.path.join("/home/jeet/MLOPs/Mnist/mnist/models_train/", str(i)))

for s in split:
    model_candidate =[]
    for i in parames:
        sp = int(s*len(X_train))
        X_train_sample = X_train[:sp]
        Y_train_sample = y_train[:sp]
        mydir = "/home/jeet/MLOPs/Mnist/mnist/models_train/{}/".format(s)
        X_train_sample = X_train_sample.reshape((len(X_train_sample), -1))
        X_val =  X_val.reshape((len(X_val), -1))
        model_candidate.append(classification_model(X_train_sample, Y_train_sample, X_val, y_val, i, s,  mydir))


    max_valid = max(model_candidate, key = lambda x: x['f1_valid']) # extract the model based on max accuracy

    best_model_folder = "/home/jeet/MLOPs/Mnist/mnist/models_train/{}/split_{}_gamma{}".format(s, max_valid['split'], max_valid['gamma'])

    f1_macro.append(test(X_test, y_test,best_model_folder)) # test function
#print(f1_macro)

plt.plot(split, f1_macro)
plt.xlabel("Training size")
plt.ylabel("F1 scores")
plt.title("Training set vs F1 scores on test data")
plt.show()