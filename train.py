# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(data_path="./data/real_estate_data.csv"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["ID", "Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Trained model R^2: {score:.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.joblib")
    return model

def main():
    train_model()  # If you just run `python train.py`, it calls this

if __name__ == "__main__":
    main()

