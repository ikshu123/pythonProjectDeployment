import flask
import numpy as np
import pandas as pd
from flask import Flask,request, url_for, redirect, render_template
from flask import jsonify
import pickle
# from geopy.geocoders import Nominatim

app = flask.Flask(__name__,template_folder="Templates")

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def main():
    return render_template('form.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    features=[[int(x) for x in request.form.getlist('name')]]
    # return render_template('form.html',pred=)
    # final=[np.array(int_features)]
    test=pd.DataFrame(features)
    prediction=model.predict(test)
    if prediction==0:
        return render_template('form.html',pred='Class is: Type 1')
    elif prediction==1:
        return render_template('form.html', pred='Class is: Type 2')
    elif prediction==2:
        return render_template('form.html', pred='Class is: Type 3')


if __name__ == '__main__':
    app.run(debug=True)


