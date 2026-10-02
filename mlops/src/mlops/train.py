from sklearn import (
    ensemble,
    gaussian_process,
    linear_model,
    naive_bayes,
    neighbors,
    svm,
    tree,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

import pandas as pd
from pathlib import Path
import pickle

MODELS = {
    
    # Ensemble methods
    "ada_boost": ensemble.AdaBoostClassifier(random_state=42),
    "bagging": ensemble.BaggingClassifier(random_state=42),
    "extra_trees": ensemble.ExtraTreesClassifier(random_state=42),
    "gradient_boosting": ensemble.GradientBoostingClassifier(random_state=42),
    "random_forest": ensemble.RandomForestClassifier(random_state=42),
    
    # Gaussian processes
    "gaussian_process": gaussian_process.GaussianProcessClassifier(random_state=42),
    
    # GLM. 
    "logistic_regression": linear_model.LogisticRegressionCV(max_iter=1000, l1_ratios=(0,), scoring="accuracy", use_legacy_attributes=False),
    "passive_aggressive": linear_model.PassiveAggressiveClassifier(random_state=42),
    "ridge": linear_model.RidgeClassifierCV(),
    "sgd": linear_model.SGDClassifier(random_state=42),
    "perceptron": linear_model.Perceptron(random_state=42),
    
    # Naive Bayes
    "bernoulli_nb": naive_bayes.BernoulliNB(),
    "gaussian_nb": naive_bayes.GaussianNB(),
    
    # Nearest neighbors
    "k_neighbors": neighbors.KNeighborsClassifier(),
    
    # SVM
    "svc": svm.SVC(probability=True, random_state=42),
    "nu_svc": svm.NuSVC(probability=True, random_state=42),
    "linear_svc": svm.LinearSVC(random_state=42),
    
    # Trees
    "decision_tree": tree.DecisionTreeClassifier(random_state=42),
    "extra_tree": tree.ExtraTreeClassifier(random_state=42),
}

def model_path(model_name):
    return Path("resources/models") / f"{model_name}.pkl"

def create_pipeline(algo):
    num_cat_transformation = ColumnTransformer([
        ("scaling", MinMaxScaler(), ["Age", "Fare"]),
        ("onehotencoding1", OneHotEncoder(handle_unknown="ignore"), ["Embarked", "Pclass"]),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["Sex"]),
        ("onehotencoding2", OneHotEncoder(handle_unknown="ignore"), ["Title", "Family_size"]),
    ], remainder="passthrough")

    bins = ColumnTransformer([
        ("Kbins", KBinsDiscretizer(n_bins=15, encode="ordinal", strategy="quantile"), [0, 1]),
    ], remainder="passthrough")

    return Pipeline([
        ("num_cat_transformation", num_cat_transformation),
        ("bins", bins),
        ("classifier", algo),
    ])


def train(model_name, input_path):
    df = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Survived"]), df["Survived"], test_size=0.2, random_state=42)
    pipeline = create_pipeline(MODELS[model_name])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv)
    print(f"[train] {model_name} 5-fold CV accuracy: {scores.mean():.4f}")

    pipeline.fit(X_train, y_train)

    path = model_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[train] model -> {path}")