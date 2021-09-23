from typing import ByteString
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import sys
import os
import shutil
from joblib import dump, load
import numpy as np
import math
from utils import create_splits, preprocess, report


digits = datasets.load_digits()

def validate(X_val, y_val, i, s, h):
    predicted = clf.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, predicted)
    if accuracy < 0.11:
        pass
        print("Skipping for rescale {}x{} testsize {} gamma {}". format(int(math.sqrt(data.shape[1])), int(math.sqrt(data.shape[1])), s, i))
    candidate = {
        'acc_valid' : accuracy,
        'gamma' : i,
        'split' : s,
        'shape' : h
        }
    return candidate


def test(X,best_model_folder, shape,split):
    data = preprocess(X, shape)
    X_train, X_test, y_train, y_test, X_val, y_val = create_splits(data, Y, split)
    clf = load(os.path.join(best_model_folder,"model.joblib"))
    predicted = clf.predict(X_test)
    report(clf, y_test, predicted) # classification report function



mydir = "/home/jeet/MLOPs/Mnist/mnist/models/"
if os.path.exists(mydir):
    shutil.rmtree(mydir)
X = digits.images
Y = digits.target
split = [0.25, 0.5, 0.75]
shape = [0.5, 1, 2, 4]
parames = [1e-7,1e-5,1e-3,0.01,0.1,1]
acc = []
checkpoint = []
model_candidate =[]
for h in shape:
    for s in split:
        for i in parames:
            data = preprocess(X, h)
            clf = svm.SVC(gamma=i)
            X_train, X_test, y_train, y_test, X_val, y_val = create_splits(data, Y,s) # data split function
            clf.fit(X_train, y_train) 
            model_candidate.append(validate(X_val, y_val,i, s, int(math.sqrt(data.shape[1])))) # validation function
            mydir = "/home/jeet/MLOPs/Mnist/mnist/models/" 
            output = mydir+"s_{}_tt_{}_val_{}_gamma{}".format(int(math.sqrt(data.shape[1])),s, s, i)
            os.makedirs(output)
            dump(clf, os.path.join(output,"model.joblib"))


max_valid = max(model_candidate, key = lambda x: x['acc_valid']) # extract the model based on max accuracy

best_model_folder = "/home/jeet/MLOPs/Mnist/mnist/models/s_{}_tt_{}_val_{}_gamma{}".format(max_valid['shape'],max_valid['split'], max_valid['split'], max_valid['gamma'])

test(X,best_model_folder, max_valid['shape']/X.shape[1],max_valid['split']) # test function
