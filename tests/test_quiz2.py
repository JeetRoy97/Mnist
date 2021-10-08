import os
import math
from sklearn import datasets, svm, metrics
import sys, os
import shutil
sys.path.insert(1, '/home/jeet/MLOPs/Mnist/mnist')
import utils
digits = datasets.load_digits()
X,Y = digits.images, digits.target

def check_split_tests_1():
    test_split = 0.2
    n = 9 # For 9 samples
    X_train, X_test, y_train, y_test, X_val, y_val = utils.create_splits(X[:n],Y[:n], test_split)
    print(X_train.shape[0], X_test.shape[0], X_val.shape[0])
    assert X_train.shape[0] == 6
    assert X_test.shape[0] == 2
    assert X_val.shape[0] == 1
    assert X_train.shape[0]+ X_test.shape[0]+ X_val.shape[0] == n

def check_split_tests_2():
    test_split = 0.2
    n = 100 # For 100 samples
    X_train, X_test, y_train, y_test, X_val, y_val = utils.create_splits(X[:n],Y[:n], test_split)
    print(X_train.shape[0], X_test.shape[0], X_val.shape[0])
    assert X_train.shape[0] == 70
    assert X_test.shape[0] == 20
    assert X_val.shape[0] == 10
    assert X_train.shape[0]+ X_test.shape[0]+ X_val.shape[0] == n
#check_split_tests()