from flask import Flask ,request,jsonify,render_template

import pickle

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

app=Flask(__name__)


#import ridge repressor and standard scaler pickel

ridge_model=pickle.load(open('/Users/nithinkumar/Desktop/Learning_projects/Medical_Cost_Prediction/ridge.pkl','rb'))
                             
Standard_Scaler=pickle.load(open('/Users/nithinkumar/Desktop/Learning_projects/Medical_Cost_Prediction/scaler.pkl','rb'))

@app.route('/')

def hello_world():
    #return "fjcnkfcnkjdfnckjdnckjdnckjnckd"
    return render_template('index.html')

gender_status={'male': 0, 'female': 1}

region_status={'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}

smoking={'yes': 1, 'no': 0}

@app.route('/predict',methods=['GET','POST'])
def predict_charge():
    
    if request.method == 'POST':
        region = request.form.get('region')
        gender = request.form.get('gender')
        smoker = request.form.get('smoker')
        
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        children = request.form.get('children')
        
        # Convert categorical variables to numerical
        gender_numeric = gender_status[gender]
        region_numeric = region_status[region]
        smoker_numeric = smoking[smoker]
       # print(gender_numeric,region_numeric,smoker_numeric)

        
        Scaled=Standard_Scaler.transform([[region_numeric,gender_numeric,smoker_numeric,age,bmi,children]])
        result=ridge_model.predict(Scaled)
        output=f"Please Check the amount estimated for the charges for the condtions  Rs.{round(result[0][0]/12)} Per month"
       # output=region+gender+smoker
        return  render_template('results.html' , result = output)
        #return render_template('results.html' , result = result)
        #return render_template('home.html')

    else:
        return render_template('home.html')

    
if __name__=="__main__":
    app.run(host='0.0.0.0',port=5001)