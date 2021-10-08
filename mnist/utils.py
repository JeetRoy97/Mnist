from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import math
import os
from joblib import dump, load


def create_splits(data, Y, split):
    train_ratio = 0.70
    validation_ratio = 0.10
    test_ratio = 0.20
    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size= round((1 - train_ratio),2), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), shuffle=False)
    return X_train, X_test, y_train, y_test, X_val, y_val


def preprocess(X, shape):
    resized_images = []
    for d in X:
        resized_images.append(rescale(d, shape, anti_aliasing=False))
    resized_images = np.asarray(resized_images)
    resized_images = resized_images.reshape((len(X), -1))
    return resized_images

def report(clf, y_test, predicted):
    print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

def validate(data, X_val, y_val, i, s, h,clf):
    predicted = clf.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, predicted)
    f1 = metrics.f1_score(y_val, predicted, average = 'macro' )
    if accuracy < 0.11:
        pass
        print("Skipping for rescale {}x{} testsize {} gamma {}". format(int(math.sqrt(data.shape[1])), int(math.sqrt(data.shape[1])), s, i))
    candidate = {
        'acc_valid' : accuracy,
        'f1_valid' : f1,
        'gamma' : i,
        'split' : s,
        'shape' : h
        }
    return candidate

def classification_model(data, X_train, y_train, X_val, y_val, gamma, split, shape,  model_path):
    #model_candidate = []
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train) 
    model_candidate = validate(data, X_val, y_val,gamma, split, int(math.sqrt(data.shape[1])),clf) # validation function
    #print(model_candidate) 
    output = model_path+"s_{}_tt_{}_val_{}_gamma{}".format(int(math.sqrt(data.shape[1])),split, split, gamma)
    os.makedirs(output)
    dump(clf, os.path.join(output,"model.joblib"))
    return model_candidate

def run_classification_experiment(data, X_train, y_train, X_val, y_val, gamma, split, shape,  model_path, expected_model_file):
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train) 
    model_candidate = validate(data, X_val, y_val,gamma, split, int(math.sqrt(data.shape[1])),clf) # validation function
    #print(model_candidate) 
    output = model_path+str(expected_model_file)
    os.makedirs(output)
    dump(clf, os.path.join(output,"model.joblib"))
