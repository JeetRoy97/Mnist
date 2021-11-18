from flask import Flask
from flask import request
#from mnist.utils import load
from joblib import dump, load
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

best_model_path = "/home/jeet/MLOPs/Mnist/mnist/models/s_8_tt_0.25_val_0.25_gamma0.01/model.joblib"

@app.route("/predict", methods = ['POST'])
def predict():
    clf = load(best_model_path)
    input_json = request.json
    image = input_json['image']
    #print(image)
    #image = np.array(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return str(predicted[0])