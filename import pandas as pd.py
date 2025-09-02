import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

TARGETS = {
    "age_group": "model_age_group.pkl",
    "gender": "model_gender.pkl",
    "ses_tier": "model_ses.pkl"
}

def train_and_save(df, target, model_path):
    X = df.drop(columns=list(TARGETS.keys()))
    y = df[target]

    num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object"]).columns.tolist()

    preproc = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_feats),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_feats)
    ])

    try:
        model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    except:
        model = RandomForestClassifier()

    pipe = Pipeline([
        ("preproc", preproc),
        ("clf", model)
    ])

    pipe.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, f"models/{model_path}")
    print(f"Saved model for {target} at models/{model_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/synth_edtech.csv")
    for target, filename in TARGETS.items():
        train_and_save(df, target, filename)
