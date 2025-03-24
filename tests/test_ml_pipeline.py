import os
import json
import pytest
import joblib

from train import train_model
from inference import init, run  # Updated import

def test_train_model():
    """
    Test that train_model() runs successfully and returns a model object.
    Also checks if the model file is created.
    """
    # Remove old model file if it exists (to ensure we're testing fresh output)
    if os.path.exists("model/model.joblib"):
        os.remove("model/model.joblib")

    model = train_model("./data/real_estate_data.csv")
    assert model is not None, "train_model() returned None"
    assert hasattr(model, "predict"), "Trained model does not have predict method"
    # Check that the model file was indeed saved
    assert os.path.exists("model/model.joblib"), "model.joblib was not created"

def test_inference():
    """
    Test that run() returns predictions in the correct format.
    """
    # Ensure the model is available (train it if not present)
    if not os.path.exists("model/model.joblib"):
        train_model("./data/real_estate_data.csv")
    
    # Initialize the inference script (load the model once)
    init()

    # Create a sample input
    sample_input = {
        "Square_Feet": 1200,
        "Num_Bedrooms": 3,
        "Num_Bathrooms": 2,
        "Num_Floors": 1,
        "Year_Built": 2000,
        "Has_Garden": 1,
        "Has_Pool": 0,
        "Garage_Size": 1,
        "Location_Score": 80,
        "Distance_to_Center": 5
    }
    # Convert sample input to JSON string (as run() expects a JSON-formatted string)
    raw_data = json.dumps(sample_input)
    
    # Call run() and parse the result
    output = run(raw_data)
    output_dict = json.loads(output)
    
    # Verify the output contains a prediction
    assert "prediction" in output_dict, "Expected 'prediction' key in output"
    prediction = output_dict["prediction"]
    assert isinstance(prediction, list) and len(prediction) == 1, "Expected a single prediction in a list"

