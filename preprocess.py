import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessor():
    # Define numeric and categorical feature columns
    num_features = [
        "login_hour", "typing_wpm",
        "session_duration_min", "sessions_per_day", "attempts_per_quiz", "avg_gap_mins"
    ]
    
    # Build preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_features),
        
    ])
    
    return preprocessor

def preprocess_data(df):
    X = df.drop(columns=["age_group", "gender", "ses_tier"])  # Inputs
    return X
