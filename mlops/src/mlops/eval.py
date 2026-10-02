import pickle
import pandas as pd
from sklearn.metrics import accurancy_score
from sklearn.model_selection import  train_test_split
from mlops.train import model_path


def evaluate(model_name,input_path):
    with open(model_path(model_name), "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Survived"]), df["Survived"], test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accurancy_score(y_test, y_pred)
    print(f"[evaluate] {model_name} accuracy: {accuracy:.4f}")