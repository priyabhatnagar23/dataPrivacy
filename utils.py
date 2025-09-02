import joblib
import pandas as pd

TARGETS = {
    "age_group": "models/model_age_group.pkl",
    "gender": "models/model_gender.pkl",
    "ses_tier": "models/model_ses.pkl"
}

def predict_all(input_dict):
    results = {}
    X = pd.DataFrame([input_dict])
    for target, path in TARGETS.items():
        model = joblib.load(path)
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        pred = classes[probs.argmax()]
        results[target] = {
            "prediction": pred,
            "probabilities": dict(zip(classes, map(float, probs)))
        }
    return results
