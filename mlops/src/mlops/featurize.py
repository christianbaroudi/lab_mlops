import pandas as pd
from pathlib import Path

def family_size(number):
    if number==1:
        return "Alone"
    elif number>1 and number <5:
        return "Small"
    else:
        return "Large"

def preprocess(input_path, output_path):
    
    df = pd.read_csv(input_path)
    
    df['Age']=df.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    df['Title']=df['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
    df['Title'] = df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['Family_size']=df['SibSp'] + df['Parch'] + 1
    df['Family_size']=df['Family_size'].apply(family_size)

    
    df.drop(columns=['Name','Parch','SibSp','Ticket'],inplace=True)
    df.to_csv(output_path, index=False)
    
