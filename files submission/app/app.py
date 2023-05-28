import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    prediction = FloatField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model



with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Begin field verification
def verify_data_types(data):
    expected_types = {
        "observation_id": [str],
        "Type": [str],
        "Date": [str],
        "Part of a policing operation": [bool],
        "Latitude": [float, int],
        "Longitude": [float, int],
        "Gender": [str],
        "Age range": [str],
        "Officer-defined ethnicity": [str],
        "Legislation": [str],
        "Object of search": [str],
        "station": [str]
    }
    selected_columns = [
        'Type',
        'Gender',
        'Age range',
        'Officer-defined ethnicity',
        'Object of search',
        'station',
        ]
    
    for key in data.keys():
        if key not in expected_types.keys():
            return True, {'error': f"Unexpected key: {key}"}

    for col, expected_type in expected_types.items():
        if col not in data:
            return (True, {'error': f"{col} column not found"})
        actual_type = type(data[col])
        if actual_type not in expected_type and col in selected_columns:
            return (True, {'error': f"{col} column has wrong data type. Expected {expected_type}, got {actual_type}"})
    return (False, "All data types are correct")

# End field verification
########################################

########################################
# Begin webserver 

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def predict():

    try:
        obs_dict = request.get_json()
    except:
        response = {'error': 'Could not parse the request'}
        return jsonify(response), 405

    is_error, error_msg = verify_data_types(obs_dict)
    if is_error:
        response = {'error': error_msg}
        return jsonify(response), 405
    
    _id = obs_dict['observation_id']
    observation = obs_dict

    try:
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    except Exception as e:
        response = {'error': f'error malformed request {e.message}'}
        return jsonify(response), 405
    except:
        response = {'error': f'error malformed request'}
        return jsonify(response), 405
    # Now get ourselves an actual prediction of the positive class.
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = int(proba >= 0.3)  # apply the threshold 
    response = {'outcome': bool(prediction)}
    
    p = Prediction(
        observation_id=_id,
        proba=proba,
        prediction= prediction,
        observation=request.data
    )

    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
        return jsonify(response), 405
    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['outcome']
        p.save()
        return jsonify(
                {
                    "observation_id": p.observation_id,
                    "outcome": p.true_class,
                    "predicted_outcome": bool(p.prediction)
                }
            )
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg}), 405
    except:
        error_msg = 'error malformed request'
        return jsonify({'error': error_msg}), 405


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
