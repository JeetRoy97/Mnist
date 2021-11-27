from typing import ByteString
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import datasets, svm, metrics, tree
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

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= round((1 - train_ratio),2), shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), shuffle=False)

#print(y_test.shape, y_val.shape)



mydir = "/home/jeet/MLOPs/Mnist/mnist/models_train/"
if os.path.exists(mydir):
    shutil.rmtree(mydir)

val_acc = []
def validate(X_val, y_val, i, m, clf):
    predicted = clf.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, predicted)
    val_acc = []
    #print(accuracy)
    f1 = metrics.f1_score(y_val, predicted, average = 'macro' )
    candidate = {
        'acc_valid' : accuracy,
        'f1_valid' : f1,
        'depth' : i,
        'max_features' : m,
        }
    return candidate, f1

def classification_model(X_train, y_train, X_val, y_val, depth,  split, m, model_path):
    #model_candidate = []
    #print(m)
    clf = tree.DecisionTreeClassifier(max_depth = depth, max_features = m)
    clf.fit(X_train, y_train)
    train = clf.score(X_train, y_train)
    model_candidate, val = validate(X_val, y_val,depth, m,clf) # validation function
    #print(model_candidate)
    output = model_path+"maxfeatures_{}_depth{}".format(m, depth)
    os.makedirs(output)
    dump(clf, os.path.join(output,"model.joblib"))
    return clf, train, val

def test_f(X_test, y_test, clf):
    #clf = load(os.path.join(best_model_folder,"model.joblib"))
    X_test =  X_test.reshape((len(X_test), -1))
    predicted = clf.predict(X_test)
    #report(clf, y_test, predicted) # classification report function
    f1 = metrics.f1_score(y_test,predicted, average = 'macro')
    #print(f1)
    #print(confusion_matrix(y_test, predicted))

    return f1


#parames = [1e-7,1e-5,1e-3,0.01,0.1,1]
leaf = [1,5,10]
max_features = ['auto', 'sqrt', 'log2']
#parames = [0.01,0.1]
f1_macro = []
checkpoint = []


#split = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
for i in range(3):
    os.makedirs(os.path.join("/home/jeet/MLOPs/Mnist/mnist/models_train/", str(i)))


final_train = []
final_val = []
final_test = []
for s in range(3):
    model_candidate =[]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= round((1 - train_ratio),2), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), shuffle=False)
    for i in leaf:
        for m in max_features:
            sp = int(s*len(X_train))
            # X_train_sample = X_train[:sp]
            # Y_train_sample = y_train[:sp]
            mydir = "/home/jeet/MLOPs/Mnist/mnist/models_train/{}/".format(s)
            X_train = X_train.reshape((len(X_train), -1))
            X_val =  X_val.reshape((len(X_val), -1))
            #print(i,m)
            clf, train, val = classification_model(X_train, y_train, X_val, y_val, i, s, m,  mydir)
            #print(i,m,train,test,val)
            final_train.append(train)
            final_val.append(val)
            test = test_f(X_test, y_test, clf)
            final_test.append(test)
            print(i,m,round(train,4),round(val,4),round(test,4))

#print(len(final_train), len(final_test))


#print(f1_macro)

