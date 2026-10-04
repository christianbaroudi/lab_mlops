import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlops.train import MODELS, model_path


def evaluate(input_path, output_path):
    for model_name in MODELS:
        evaluate_model(model_name, input_path, output_path)


def evaluate_model(model_name, input_path, output_path):
    # Load trained model
    with open(model_path(model_name), "rb") as f:
        pipeline = pickle.load(f)

    # Load data
    df = pd.read_csv(input_path)

    # Use Titanic training rows
    train = df.loc[:890]

    X_train, X_test, y_train, y_test = train_test_split(
        train.drop(columns=["Survived"]),
        train["Survived"],
        test_size=0.2,
        random_state=42,
        stratify=train["Survived"],
    )

    # Predict
    y_pred = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[evaluate] {model_name} accuracy: {accuracy:.4f}")

    # Save one JSON per model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "n_samples": len(y_test),
    }

    metrics_path = output_dir / f"{model_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[evaluate] metrics -> {metrics_path}")