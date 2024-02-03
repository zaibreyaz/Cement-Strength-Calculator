from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

## Route for homepage


## Route for Single data point prediction
@app.route('/',methods=['GET','POST'])
def predict_datapoint():

    if request.method=='POST':

        cement=int(request.form.get("cement"))
        blast_furnace_slag = float(request.form.get('blast_furnace_slag'))
        fly_ash = float(request.form.get('fly_ash'))
        water = float(request.form.get('water'))
        superplasticizer = float(request.form.get('superplasticizer'))
        coarse_aggregate = float(request.form.get('coarse_aggregate'))
        fine_aggregate = float(request.form.get('fine_aggregate'))
        Age = float(request.form.get('age'))

        new_data=scaler.transform([[cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, Age]])
        result=model.predict(new_data)
       
            
        return render_template('result.html',result=result[0])

    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")