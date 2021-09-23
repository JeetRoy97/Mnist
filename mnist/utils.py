from skimage.transform import rescale, resize, downscale_local_mean
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np


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
