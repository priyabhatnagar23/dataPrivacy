import os
import random
import json
import pandas as pd
import numpy as np
from preprocess import build_preprocessor
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap


df = pd.read_csv("data/synth_edtech.csv")

X = df.drop(columns=["age_group", "gender", "ses_tier", "quiz_score", "time_per_question_sec", "device_type", "browser"])  # contributing input features only
y = df["age_group"]  # target

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # e.g., '13-15' -> 0, '16-18' -> 1, etc.

class_counts = np.bincount(y_encoded)
print("\nClass distribution (encoded):", class_counts)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

preprocessor = build_preprocessor()

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    ))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.1],
    "classifier__subsample": [0.8, 1],
    "classifier__reg_alpha": [0, 0.1],
    "classifier__reg_lambda": [1, 2]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy"
)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)

best_pipeline = grid_search.best_estimator_

X_train_preprocessed = best_pipeline.named_steps['preprocessor'].transform(X_train)

feature_names_num = best_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler'].get_feature_names_out()
all_feature_names = list(feature_names_num)

# SHAP explainer
explainer = shap.TreeExplainer(best_pipeline.named_steps['classifier'])
shap_values = explainer.shap_values(X_train_preprocessed)  # list of arrays (one per class)

# Stack SHAP values across classes: shape -> (num_samples, num_features, num_classes)
shap_values_stacked = np.stack(shap_values, axis=-1)

# Mean absolute SHAP values across samples and classes
mean_abs_shap = np.mean(np.abs(shap_values_stacked), axis=(0, 2))  # shape: (num_features,)

# Top 5 features
top_indices = np.argsort(mean_abs_shap)[::-1][:5]
print("Top 5 features based on SHAP values:")
for i in top_indices:
    print(all_feature_names[i], mean_abs_shap[i])

# Plotting SHAP summary (bar plot)
shap.summary_plot(shap_values, X_train_preprocessed, feature_names=all_feature_names, plot_type="bar")
# plt.savefig("outputs/shap_feature_importance.png")

os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/xgb_age_group_pipeline.pkl")
joblib.dump(le, "models/age_group_label_encoder.pkl")

y_pred = best_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



# Testing:
def generate_random_sample():
    # Simulating age group first to maintain correlations
    age_groups = ["13-15", "16-18", "19-22"]
    age_probs = [0.3, 0.4, 0.3]
    selected_age = np.random.choice(age_groups, p=age_probs)

    if selected_age == "13-15":
        typing_wpm = round(np.random.normal(35, 8), 2)
        login_hour = np.random.choice(range(7, 22))
        session_duration = round(np.random.normal(25, 5), 2)
    elif selected_age == "16-18":
        typing_wpm = round(np.random.normal(50, 10), 2)
        login_hour = np.random.choice(range(8, 23))
        session_duration = round(np.random.normal(30, 7), 2)
    else:
        typing_wpm = round(np.random.normal(65, 12), 2)
        login_hour = np.random.choice(range(10, 24))
        session_duration = round(np.random.normal(40, 10), 2)

    sample = {
        "login_hour": login_hour,
        "typing_wpm": typing_wpm,
        "quiz_score": round(np.clip(np.random.normal(75, 10), 0, 100), 2),
        "time_per_question_sec": round(np.random.normal(30, 5), 2),
        "session_duration_min": session_duration,
        "sessions_per_day": round(np.random.uniform(1, 4), 2),
        "attempts_per_quiz": round(np.random.uniform(1, 2), 2),
        "avg_gap_mins": round(np.random.uniform(0.5, 3.0), 2),
        "device_type": np.random.choice(["Android", "iOS", "Desktop", "Shared/Other"]),
        "browser": np.random.choice(["Chrome", "Safari", "Firefox", "Edge", "In-App"])
    }

    return pd.DataFrame([sample])

new_data = generate_random_sample()
print(new_data)

y_pred_num = best_pipeline.predict(new_data)

y_pred_label = le.inverse_transform(y_pred_num)
print("\nPredicted age group for given input:", y_pred_label[0])

y_pred_proba = best_pipeline.predict_proba(new_data)
print("Prediction probabilities:", y_pred_proba)

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
result_summary = {
    "accuracy": acc,
    "classification_report": report
}
os.makedirs("outputs", exist_ok=True)
with open("outputs/model_evaluation.json", "w") as f:
    json.dump(result_summary, f, indent=4)

