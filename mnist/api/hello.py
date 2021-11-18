from flask import Flask
from flask import request
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods = ['POST'])
def predict():
    input_json = request.json
    image = input_json['image']
    print(image)
    return '<p>Image printed</p>'