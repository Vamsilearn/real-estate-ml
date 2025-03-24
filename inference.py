import json
import pandas as pd
import joblib

# Global variable to store the model
model = None

def init():
    global model
    # Load the model once when the container starts
    model = joblib.load("model/model.joblib")

def run(raw_data):
    try:
        # Parse the incoming JSON string
        data = json.loads(raw_data)
        # Convert the data to a DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        # Make prediction
        prediction = model.predict(df)
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
