import os
import json
import pytest
import joblib

from train import train_model
from inference import init, run  # Updated import: now import init and run instead of run_inference

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
    Test that the run() function returns predictions in the correct format.
    """
    # Ensure the model is available (train it if not present)
    if not os.path.exists("model/model.joblib"):
        train_model("./data/real_estate_data.csv")
    
    # Initialize the inference environment (load the model)
    init()
    
    # Create a sample input dictionary
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
    # Convert sample input to JSON string, as run() expects raw JSON input
    raw_data = json.dumps(sample_input)
    
    # Call the run() function
    result = run(raw_data)
    
    # Parse the JSON result
    result_dict = json.loads(result)
    assert "prediction" in result_dict, "Prediction key missing in output"
    # Optionally, check the prediction value type if needed:
    # assert isinstance(result_dict["prediction"][0], (float, int))
