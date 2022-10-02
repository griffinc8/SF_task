from flask import Flask, jsonify, request
import joblib
import pickle
from pathlib import Path
import os
import traceback
import pandas as pd


app = Flask(__name__)

dir_path = Path(__file__).parent
final_logit = pickle.load(open(os.path.join(dir_path, 'model.pkl'), 'rb'))


@app.route('/predict', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():

    ##list of parameters data comes through here, looping through list of request json
    ## create a loop
    json_ = request.get_json(force = "True")
    df = pd.DataFrame(json_, index = [0])
    print(df)

    ##analyze and write comments

    prediction = list(final_logit.predict(df))
    print(prediction)

    final = 0
    if prediction[0] < .5: 
        final = 0
    else:
        final = 1

        ##prediciton will be an arry of predictions 

    return jsonify({'probability_of_class_one': prediction, "predicted-class": final})

if __name__ == '__main__':
    app.run(port = 1313, debug = True)