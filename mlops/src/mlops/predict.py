import pickle
from pathlib import Path

import pandas as pd

from mlops.train import model_path



def predict(model_name,input_path, output_path):
    with open(model_path(model_name), "rb") as f:
        pipeline = pickle.load(f)
    df = pd.read_csv(input_path)
    
    test = df[df["Survived"].isna()].drop(columns=["Survived"])
    X = test.drop("PassengerId", axis=1)

    predictions = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pipeline.predict(X)})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"[predict] {len(predictions)} predictions -> {output_path}")