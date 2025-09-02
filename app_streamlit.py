import streamlit as st
from utils import predict_all, TARGETS
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Privacy Simulator", layout="centered")
st.title("🔍 Data Privacy Simulator for EdTech")

st.sidebar.header("Enter your EdTech Activity Data")
features = {
    "login_hour": st.sidebar.slider("Login Hour", 0, 23, 10),
    "typing_wpm": st.sidebar.slider("Typing Speed (WPM)", 10, 120, 50),
    "quiz_score": st.sidebar.slider("Quiz Score (%)", 0, 100, 75),
    "time_per_question_sec": st.sidebar.slider("Time per Question (sec)", 5, 120, 30),
    "session_duration_min": st.sidebar.slider("Session Duration (min)", 5, 120, 30),
    "sessions_per_day": st.sidebar.slider("Sessions per Day", 1.0, 5.0, 2.0),
    "attempts_per_quiz": st.sidebar.slider("Attempts per Quiz", 1.0, 3.0, 1.2),
    "avg_gap_mins": st.sidebar.slider("Avg Gap between Actions (min)", 0.5, 10.0, 1.0),
    "device_type": st.sidebar.selectbox("Device Type", ["Android", "iOS", "Desktop", "Shared/Other"]),
    "browser": st.sidebar.selectbox("Browser", ["Chrome", "Safari", "Firefox", "Edge", "In-App"])
}

if st.sidebar.button("Analyze"):
    results = predict_all(features)
    st.subheader("🔮 Predictions & Confidence")
    confidences = []
    
    for trait, res in results.items():
        st.write(f"**{trait.title()}**: {res['prediction']}")
        for cls, prob in res["probabilities"].items():
            st.write(f"- {cls}: {prob:.2f}")
        confidences.append(max(res["probabilities"].values()))
    
    avg_conf = sum(confidences) / len(confidences)
    if avg_conf < 0.5:
        risk = "🟢 Low"
    elif avg_conf < 0.75:
        risk = "🟡 Medium"
    else:
        risk = "🔴 High"
    st.subheader(f"Privacy Risk: {risk}")

    # 📊 SHAP Explanations
    st.subheader("📌 Why These Predictions Were Made")
    X_input = pd.DataFrame([features])

    for trait, model_path in TARGETS.items():
        st.markdown(f"#### 🔍 {trait.title()} Prediction Explanation")
        model = joblib.load(model_path)

        try:
            explainer = shap.TreeExplainer(model.named_steps["clf"])
            shap_values = explainer.shap_values(model.named_steps["preproc"].transform(X_input))

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, 
                              pd.DataFrame(model.named_steps["preproc"].transform(X_input)),
                              feature_names=model.named_steps["preproc"].get_feature_names_out(),
                              plot_type="bar", 
                              show=False)
            st.pyplot(fig)
        except Exception as e:
            st.write(f"(No SHAP plot available for {trait} — {e})")
