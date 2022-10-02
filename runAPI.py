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
    ##if final_logit exists
    if final_logit:
        try:

            ##list of parameters data comes through here, looping through list of request json
            ## create a loop
            json_ = request.json
            ## print(json_)

            ##analyze and write comments

            query = pd.get_dummies(pd.DataFrame([json_]))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(final_logit.predict(query))

            final = 0
            if prediction[0] < .5: 
                final = 0
            else:
                final = 1

            ##prediciton will be an arry of predictions 

            return jsonify({'probability_of_class_one': prediction, "predicted-class": final})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 1313 # If you don't provide any port then the port will be set to 12345
    logit = joblib.load('model.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)
