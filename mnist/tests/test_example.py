import math
from sklearn import datasets, svm, metrics
import sys, os

from joblib import dump, load
import shutil
sys.path.insert(1, '/home/jeet/MLOPs/Mnist/mnist')
import utils
import numpy as np

digits = datasets.load_digits()
X,Y = digits.images, digits.target

test_split = 0.2

X_train, X_test, y_train, y_test, X_val, y_val = utils.create_splits(X[:10],Y[:10], test_split)
best_model_path = "/home/jeet/MLOPs/Mnist/mnist/models/s_8_tt_0.25_val_0.25_gamma0.01/model.joblib"
best_model_path2 = "/home/jeet/MLOPs/Mnist/mnist/models/s_8_tt_0.25_val_0.25_gamma0.1/model.joblib"

def pick_index(y_sample):
    index = np. where(y_test == 0)
    print(X_test[index])

#pick_index(0)

def test_digit_correct_0():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 0]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==0

def test_digit_correct_1():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 1]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==1

def test_digit_correct_2():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 2]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==2

def test_digit_correct_3():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 3]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==3

def test_digit_correct_4():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 4]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==4

def test_digit_correct_5():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 5]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==5

def test_digit_correct_6():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 6]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==6

def test_digit_correct_7():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 7]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==7

def test_digit_correct_8():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 8]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==8

def test_digit_correct_9():
    clf = load(best_model_path)
    images_0 = digits.images[digits.target == 9]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==9


test_digit_correct_0()
def test_equal():
	assert 1 == 1

def test_digit_correct_0():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 0]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==0

def test_digit_correct_1():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 1]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==1

def test_digit_correct_2():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 2]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==2

def test_digit_correct_3():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 3]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==3

def test_digit_correct_4():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 4]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==4

def test_digit_correct_5():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 5]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==5

def test_digit_correct_6():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 6]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==6

def test_digit_correct_7():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 7]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==7

def test_digit_correct_8():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 8]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==8

def test_digit_correct_9():
    clf = load(best_model_path2)
    images_0 = digits.images[digits.target == 9]
    #print(images_0.shape)
    image_s = np.array(images_0[0]).reshape(1,-1)
    predicted_output = clf.predict(image_s)
    #print(predicted_output) 
    expected_output = digits.target[digits.target == 0][0] 
    #print(expected_output)
    assert  predicted_output[0]==9


test_digit_correct_0()

