


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import warnings
#from skimage.transform import rescale, resize, downscale_local_mean
from skimage.transform import rescale
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import math



digits = datasets.load_digits()
print(digits.images.shape)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

def rescale_image(X, shape):
    resized_images = []
    for d in X:
        resized_images.append(rescale(d, shape, anti_aliasing=False))
    resized_images = np.asarray(resized_images)
    resized_images = resized_images.reshape((len(X), -1))
    return resized_images

def fun(digits, shape, split):
    X = digits.images
    Y = digits.target
    data = rescale_image(X, shape)
    #print(data.shape)
    clf = svm.SVC(gamma=0.001)
    X_train, X_test, y_train, y_test = train_test_split(
        data, Y, test_size=split, shuffle=True)
    y_train = y_train.reshape(-1,1)
    #print(X_train.shape, y_train.shape)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("{}x{} {}:{} {}".format(int(math.sqrt(data.shape[1])), int(math.sqrt(data.shape[1])), int((1-split)*100),int(split*100), accuracy))
    return accuracy

shape = [0.5, 1, 2, 4]
split = [0.25, 0.5, 0.75]

for i in shape:
    for j in split:
        fun(digits,i , j)
