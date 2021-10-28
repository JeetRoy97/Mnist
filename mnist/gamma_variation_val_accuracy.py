
from typing import ByteString
import matplotlib.pyplot as plt
from sklearn import tree
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

digits = datasets.load_digits()

def hyper_accuracy(digits, para, split_size, model):

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    if model == "SVC":
        clf = svm.SVC(gamma=para)
    else:
        clf = tree.DecisionTreeClassifier(max_depth = para)


    #clf = svm.SVC(gamma=para)

    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=split_size, shuffle=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=split_size, shuffle=True)

    clf.fit(X_train, y_train)
    """print("Train ", len(X_train))
    print("Val ", len(X_val))
    print("Test ", len(X_test))
    print("Train accuracy",clf.score(X_train, y_train))
    print("Val accuracy",clf.score(X_val, y_val))
    print("Test accuracy",clf.score(X_test, y_test))"""

    return clf.score(X_test, y_test)

def hyper(digits, para, split_size, model):

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    if model == "SVC":
        clf = svm.SVC(gamma=para)
    else:
        clf = tree.DecisionTreeClassifier(max_depth = para)


    #clf = svm.SVC(gamma=para)

    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=split_size, shuffle=False)
    
    X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=split_size, shuffle=False)

    clf.fit(X_train, y_train)

    predicted = clf.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, predicted)
    return accuracy

parames = [1e-7,1e-5,1e-3,0.01,0.1,1]
depth = [5,10,50,100,500]
split = [0.15, 0.25, 0.40, 0.75, 0.1]
#parames = sys.args[0]

"""acc_svm = []
for i in range(len(parames)):
    acc_svm.append(hyper(digits, parames[i], split[j],"SVC"))

acc_dt = []
for i in range(len(depth)):
    acc_dt.append(hyper(digits, depth[i], split[j], "DT"))

print("Gamma    ","Accuracy")
for i in range(len(parames)):
    print(parames[i],"     ",acc_svm[i])

best_gamma = parames[acc_svm.index(max(acc_svm))]
print("optimal gamma ", best_gamma)
hyper_accuracy(digits, best_gamma, "SVC")



print("Depth    ","Accuracy")
for i in range(len(depth)):
    print(depth[i],"     ",acc_dt[i])

best_depth = depth[acc_dt.index(max(acc_dt))]
print("optimal depth ", best_depth)
hyper_accuracy(digits, best_depth, , "DT")"""


best_gamma = []
best_svm_acc = []
best_depth = []
best_dt_acc = []


for j in range(len(split)):
    acc_svm = []

    for i in range(len(parames)):
        acc_svm.append(hyper(digits, parames[i], split[j],"SVC"))
    best_gamma = parames[acc_svm.index(max(acc_svm))]
    best_svm_acc.append(hyper_accuracy(digits, best_gamma,  split[j], "SVC"))

    acc_dt = []
    for i in range(len(depth)):
        acc_dt.append(hyper(digits, depth[i], split[j], "DT"))
    
    best_depth = depth[acc_dt.index(max(acc_dt))]
    best_dt_acc.append(hyper_accuracy(digits, best_depth, split[j] , "DT"))

        

#print(best_dt_acc)
#print(best_svm_acc)

data = {"split":split, "optimal_gamma": best_gamma, "SVM_acc" : best_svm_acc, "optimal_depth": best_depth, "DT_acc": best_dt_acc}
df = pd.DataFrame(data)


#print("Mean", "\u00B1","std SVM ", round(df["SVM_acc"].mean(),6), "\u00B1", round(df["SVM_acc"].std(),6))
#print("Mean", "\u00B1" ,"std Decision Tree", round(df["DT_acc"].mean(),6), "\u00B1", round(df["DT_acc"].std(),6))

ms = round(df["SVM_acc"].mean(),6), "\u00B1" ,round(df["SVM_acc"].std(),6)
md = round(df["DT_acc"].mean(),6), "\u00B1", round(df["DT_acc"].std(),6)
s = pd.Series([' ',' ', ms, ' ', md], index = ['split', 'optimal_gamma','SVM_acc', 'optimal_depth', 'DT_acc'])

df = df.append(s, ignore_index = True)
print(df)