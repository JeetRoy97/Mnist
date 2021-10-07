import math
from sklearn import datasets, svm, metrics
import sys, os
import shutil
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jeet/MLOPs/Mnist/mnist')
import utils
#from mnist.utils import create_splits, preprocess, report, classification_model
def test_equal():
	assert 1 == 1

def test_model_writing():
	gamma = 0.001
	shape = 0.5
	split = 0.5
	expected_model_file = "dummy_model"
	mydir = "/home/jeet/MLOPs/Mnist/mnist/models/"
	if os.path.exists(mydir):
		shutil.rmtree(mydir)
	digits = datasets.load_digits()
	X,Y = digits.images, digits.target
	data = utils.preprocess(X, shape)
	X_train, X_test, y_train, y_test, X_val, y_val = utils.create_splits(data[:100], Y[:100],split)
	candidate = utils.run_classification_experiment(data, X_train, y_train, X_val, y_val, gamma, split, shape,  mydir, expected_model_file)
	#print(mydir+expected_model_file)
	assert os.path.isfile(mydir+expected_model_file+"/model.joblib")


def test_small_data_overfit_checking():
	gamma = 0.01
	shape = 0.5
	split = 0.25
	mydir = "/home/jeet/MLOPs/Mnist/mnist/models/"
	if os.path.exists(mydir):
		shutil.rmtree(mydir)
	digits = datasets.load_digits()
	X,Y = digits.images, digits.target
	data = utils.preprocess(X, shape)
	X_train, X_test, y_train, y_test, X_val, y_val = utils.create_splits(data[:100], Y[:100],split)
	candidate = utils.classification_model(data, X_train, y_train, X_val, y_val, gamma, split, shape,  mydir)
	assert candidate['acc_valid']	> 0.8 #threshold to check over-fitting
