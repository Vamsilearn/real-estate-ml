# inference.py
import pandas as pd
import joblib

def run_inference(input_data):
    model = joblib.load("model/model.joblib")
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame(input_data)
    return model.predict(df)
