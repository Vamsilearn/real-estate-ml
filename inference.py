import json
import pandas as pd
import joblib

# Global model variable to be loaded once
model = None

def init():
    global model
    # Load the model once when the container starts
    model = joblib.load("model/model.joblib")

def run(raw_data):
    try:
        # Parse the incoming JSON data
        data = json.loads(raw_data)
        
        # Convert to DataFrame: assume the data is either a dict or a list of dicts
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Make predictions
        prediction = model.predict(df)
        
        # Convert prediction to list and return as JSON string
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
