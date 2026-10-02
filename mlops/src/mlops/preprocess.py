import pandas as pd
from pathlib import Path


def preprocess(train_path, test_path, output_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Handling null values: Cabin has too many, so drop it.
    train = train.drop(columns=["Cabin"])
    test = test.drop(columns=["Cabin"])
    train["Embarked"] = train["Embarked"].fillna("S")
    test["Fare"] = test["Fare"].fillna(test["Fare"].mean())

    # One DataFrame for easier manipulation.
    df = pd.concat([train, test], sort=True).reset_index(drop=True)
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Age"] = df["Age"].astype("int64")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] train={len(train)} + test={len(test)} rows -> {output_path}")

