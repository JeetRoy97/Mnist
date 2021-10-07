from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import math
import os
from joblib import dump, load


def create_splits(data, Y, split):
    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size = split, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(data, Y, test_size = split, shuffle=False)
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
