import pandas as pd
from pathlib import Path


def preprocess(input_path, output_path):
    
    df = pd.read_csv(input_path)
    df.drop(columns=['Cabin'],inplace=True)
    df['Embarked'].fillna('S',inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df.to_csv(output_path, index=False)
    
