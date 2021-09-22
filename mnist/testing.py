"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 claus2
# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import warnings
import math
from skimage.transform import resize
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.





digits = datasets.load_digits()
print(digits.images.shape)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images


def fun(digits, shape, split):
    n_samples = len(digits.images)
    X = digits.images
    Y = digits.target
    #print(X.shape, Y.shape)
    #print(type(digits.images))
    #data1 = np.resize(digits.images, (n_samples,shape))
    data1=[]
    for i in range(len(X)):
        data1.append(resize(X[i], (math.sqrt(shape),math.sqrt(shape))))
    data1 = np.array(data1)
    data1 = data1.reshape((n_samples, -1))
    #data1 = data1.reshape(n_samples, -1)
    #print("No. of images ",data1.shape[1])
    #print("data1 =", data1.shape)

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)


    X_train, X_test, y_train, y_test = train_test_split(
        data1, digits.target, test_size=split, shuffle=False)

    # Learn the digits on the train subset
    #print(X_train.shape, y_train.shape)
    y_train = y_train.reshape(-1,1)
    #print(X_train.shape, y_train.shape)
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.
    '''
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(16, 16)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Prediction: {prediction}')
    '''
    accuracy = metrics.accuracy_score(y_test, predicted)
    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    #print(f"Classification report for classifier {clf}:\n"f"{metrics.classification_report(y_test, predicted)}\n")

    return accuracy

shape = 256
split = 0.75
#print("Accuracy ", fun(digits,shape , split))
print("16*16   ","75:25   ", fun(digits,shape , split))

shape = 256
split = 0.5
#print("Accuracy ", fun(digits,shape , split))
print("16*16   ","50:50   ", fun(digits,shape , split))

shape = 256
split = 0.25
#print("Accuracy ", fun(digits,shape , split))
print("16*16   ","25:75   ", fun(digits,shape , split))


shape = 1024    
split = 0.75
print("32*32   ","75:25   ", fun(digits,shape , split))
#print("Accuracy ", fun(digits,shape , split))

shape = 1024    
split = 0.5
#print("Accuracy ", fun(digits,shape , split))
print("32*32    ","50:50   ", fun(digits,shape , split))

shape = 1024
split = 0.25
#print("Accuracy ", fun(digits,shape , split))
print("32*32    ","25:75   ", fun(digits,shape , split))

shape = 4096    
split = 0.75
#print("Accuracy ", fun(digits,shape , split))
print("64*64   ","75:25   ", fun(digits,shape , split))

shape = 4096    
split = 0.5
#print("Accuracy ", fun(digits,shape , split))
print("64*64   ","50:50   ", fun(digits,shape , split))

shape = 4096
split = 0.25
#print("Accuracy ", fun(digits,shape , split))
print("64*64   ","25:75   ", fun(digits,shape , split))

'''
shape= [256, 1024, 144]
split = [0.25, 0.50, 0.75]

for i in shape:
    for j in split:
        print(shape ,split , fun(digits,shape , split))

'''