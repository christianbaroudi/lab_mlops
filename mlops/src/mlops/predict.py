import pickle
from pathlib import Path

import pandas as pd

from mlops.train import MODELS, model_path


def predict(input_path, output_path):
    df = pd.read_csv(input_path)

    # Test rows are the rows without a known Survived value
    test = df[df["Survived"].isna()].copy()
    
    # Keep the same features used during training
    X = test.drop(columns=["Survived"])

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        print(f"[predict] loading {model_name}...")

        # Load trained model
        with open(model_path(model_name), "rb") as f:
            pipeline = pickle.load(f)

        # Make predictions
        predictions = pipeline.predict(X)

        # Create prediction file
        result = pd.DataFrame({
            "PassengerId": test["PassengerId"],
            "Survived": predictions,
        })

        prediction_path = output_dir / f"{model_name}.csv"
        result.to_csv(prediction_path, index=False)

        print(
            f"[predict] {len(result)} predictions "
            f"-> {prediction_path}"
        )