# tests/test_ml_pipeline.py
import os
import pytest
import joblib

from train import train_model
from inference import run_inference

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
    Test that run_inference() returns predictions in the correct format.
    """
    # Ensure the model is available (train it if not present)
    if not os.path.exists("model/model.joblib"):
        train_model("./data/real_estate_data.csv")

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

    preds = run_inference(sample_input)
    assert len(preds) == 1, "Expected a single prediction"
    assert isinstance(preds[0], float), "Prediction should be a float (or numpy float)"
