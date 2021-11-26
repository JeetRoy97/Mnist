from flask import Flask
from flask import request
#from mnist.utils import load
from joblib import dump, load
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

best_model_path_svm = "/home/jeet/MLOPs/Mnist/mnist/models_train/1.0/split_1.0_gamma0.001/model.joblib"
best_model_path_decision = "/home/jeet/MLOPs/Mnist/mnist/models_dt/s_8_tt_0.25_val_0.25_depth50/model.joblib"
@app.route("/svm_predict", methods = ['POST'])
def predict():
    clf = load(best_model_path_svm)
    input_json = request.json
    image = input_json['image']
    #print(image)
    #image = np.array(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])

@app.route("/decision_predict", methods = ['POST'])
def decision_predict():
    clf = load(best_model_path_decision)
    input_json = request.json
    image = input_json['image']
    #print(image)
    #image = np.array(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])
