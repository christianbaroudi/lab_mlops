import json
import pickle
import pandas as pd
from sklearn.metrics import accurancy_score
from sklearn.model_selection import  train_test_split
from mlops.train import model_path
from pathlib import Path


def evaluate(model_name,input_path,output_path):
    with open(model_path(model_name), "rb") as f:
        pipeline = pickle.load(f)
    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Survived"]), df["Survived"], test_size=0.2, random_state=42)
    y_pred = pipeline.predict(X_test)
    accuracy = accurancy_score(y_test, y_pred)
    print(f"[evaluate] {model_name} accuracy: {accuracy:.4f}")
    
    metrics = {"model": model_name, "accuracy": accuracy, "n_samples": len(y_test)}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] metrics -> {output_path}")