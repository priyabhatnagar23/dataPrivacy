import streamlit as st
import pandas as pd
import joblib
import time
import platform
import streamlit.components.v1 as components
import shap
import plotly.express as px
import numpy as np
import streamlit_js_eval
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(page_title="Data Privacy Simulator", layout="wide")
st.title("Data Privacy Simulator for EdTech")


# Load trained model and label encoder
model = joblib.load("models/xgb_age_group_pipeline.pkl")
label_encoder = joblib.load("models/age_group_label_encoder.pkl")

st.markdown("Simulating how your digital behavior reveals private traits...")

# --- Feature Extraction Section ---
st.subheader("Passive Data Capture")

# --- Device and Browser Detection ---
device_type = platform.system()

# Get browser info via JS evaluation
browser_info = streamlit_js_eval(js_expressions="navigator.userAgent", key="browser_info")

# Function to parse and clean the browser name
def detect_browser(user_agent: str) -> str:
    if not user_agent:
        return "Unknown"

    ua = user_agent.lower()
    if "edg/" in ua:
        return "Microsoft Edge"
    elif "opr" in ua or "opera" in ua:
        return "Opera"
    elif "chrome" in ua and "safari" in ua and "chromium" not in ua:
        return "Google Chrome"
    elif "safari" in ua and "chrome" not in ua:
        return "Safari"
    elif "firefox" in ua:
        return "Mozilla Firefox"
    elif "trident" in ua or "msie" in ua:
        return "Internet Explorer"
    else:
        return "Other / Unknown"

# Save browser info to session state
if browser_info:
    parsed_browser = detect_browser(browser_info)
    st.session_state["browser"] = parsed_browser

browser = st.session_state.get("browser", "Unknown")

st.write(f"🖥️ **Device Type:** {device_type}")
st.write(f"🌐 **Detected Browser:** {browser}")


# Record login time
login_time = time.localtime()
login_hour = login_time.tm_hour

# --- Typing Speed Test ---
st.subheader("⌨️ Typing Speed Test (Behavioral Feature Extraction)")

target_text = (
    "Education technology platforms collect various forms of user data to personalize learning. "
    "While this can improve engagement, it can also reveal sensitive traits like age or behavior."
)
st.text_area("Text to type:", value=target_text, height=120, disabled=True)

typed_text = st.text_area("Start typing here:", key="typed_area", height=120)

# Initialize persistent state
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "completed" not in st.session_state:
    st.session_state.completed = False

# Start timer when typing starts
if st.session_state.start_time is None and typed_text.strip() != "":
    st.session_state.start_time = time.time()

# When 90% of the target is typed
if not st.session_state.completed and len(typed_text.strip()) >= len(target_text) * 0.9:
    st.session_state.completed = True
    end_time = time.time()
    elapsed_minutes = (end_time - st.session_state.start_time) / 60
    total_chars = len(typed_text)
    words_typed = total_chars / 5
    wpm = words_typed / elapsed_minutes if elapsed_minutes > 0 else 0
    correct_chars = sum(1 for a, b in zip(typed_text, target_text) if a == b)
    accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    awpm = wpm * (accuracy / 100)

    st.session_state.results = {
        "wpm": round(wpm, 2),
        "accuracy": round(accuracy, 2),
        "awpm": round(awpm, 2),
        "elapsed_minutes": round(elapsed_minutes, 3)
    }

# Display results
if "results" in st.session_state:
    r = st.session_state.results
    st.success(f"✅ Typing Speed: **{r['wpm']} WPM**")
    st.info(f"🎯 Accuracy: **{r['accuracy']}%**")
    st.success(f"⚡ Adjusted WPM: **{r['awpm']} AWPM** (over {r['elapsed_minutes']} min)")
else:
    st.info("Start typing above to measure your speed and accuracy.")

# Reset
if st.button("🔄 Reset Typing Test"):
    for key in ["start_time", "completed", "results"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ===============================
# 🧩 Mini Cognitive Quiz Section
# ===============================
st.subheader("🧠 Quick Cognitive Quiz (Simulated EdTech Interaction)")

quiz_data = [
    {
        "question": "Which of the following words doesn’t belong?",
        "options": ["Apple", "Banana", "Mango", "Chair"],
        "answer": "Chair"
    },
    {
        "question": "What number comes next in the series: 2, 4, 8, 16, ?",
        "options": ["18", "24", "32", "20"],
        "answer": "32"
    },
    {
        "question": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "options": ["Yes", "No", "Cannot be determined"],
        "answer": "Cannot be determined"
    }
]

if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "current_q": 0,
        "score": 0,
        "attempts": 0,
        "total_time": 0.0,
        "start_time": None,
        "completed": False
    }

q_state = st.session_state.quiz_state

if not q_state["completed"]:
    q = quiz_data[q_state["current_q"]]
    st.write(f"**Question {q_state['current_q'] + 1} of {len(quiz_data)}**")
    st.write(q["question"])

    if q_state["start_time"] is None:
        q_state["start_time"] = time.time()

    user_answer = st.radio("Select your answer:", q["options"], key=f"q{q_state['current_q']}_ans")

    if st.button("Submit Answer"):
        q_state["attempts"] += 1
        end_time = time.time()
        elapsed = end_time - q_state["start_time"]
        q_state["total_time"] += elapsed

        if user_answer == q["answer"]:
            q_state["score"] += 1

        q_state["start_time"] = None
        q_state["current_q"] += 1

        if q_state["current_q"] >= len(quiz_data):
            q_state["completed"] = True

        st.rerun()

else:
    avg_time_per_q = q_state["total_time"] / len(quiz_data)
    quiz_score = (q_state["score"] / len(quiz_data)) * 100
    attempts_per_quiz = q_state["attempts"]

    st.success("✅ Quiz Completed!")
    st.metric("Quiz Score (%)", round(quiz_score, 2))
    st.metric("Avg Time per Question (s)", round(avg_time_per_q, 2))
    st.metric("Attempts per Quiz", attempts_per_quiz)

    # Store results for model input
    st.session_state["quiz_score"] = round(quiz_score, 2)
    st.session_state["time_per_question_sec"] = round(avg_time_per_q, 2)
    st.session_state["attempts_per_quiz"] = attempts_per_quiz

    if st.button("🔄 Retake Quiz"):
        st.session_state.quiz_state = {
            "current_q": 0,
            "score": 0,
            "attempts": 0,
            "total_time": 0.0,
            "start_time": None,
            "completed": False
        }
        st.rerun()

# --- Sliders for Other Features ---
quiz_score = st.session_state.get("quiz_score", 75)
time_per_question_sec = st.session_state.get("time_per_question_sec", 30)
attempts_per_quiz = st.session_state.get("attempts_per_quiz", 1)
session_duration_min = st.slider("Session duration (min)", 10, 60, 25)
sessions_per_day = st.slider("Sessions per day", 1, 5, 2)
avg_gap_mins = st.slider("Average gap between sessions (min)", 0.5, 5.0, 2.0)

# --- Create DataFrame ---
typing_wpm_value = (
    st.session_state.results["wpm"]
    if ("results" in st.session_state and st.session_state.results["wpm"])
    else 40  # fallback default
)

user_input = pd.DataFrame({
    "login_hour": [login_hour],
    "typing_wpm": [typing_wpm_value],
    "quiz_score": [quiz_score],
    "time_per_question_sec": [time_per_question_sec],
    "session_duration_min": [session_duration_min],
    "sessions_per_day": [sessions_per_day],
    "attempts_per_quiz": [attempts_per_quiz],
    "avg_gap_mins": [avg_gap_mins],
    "device_type": [device_type],
    "browser": [browser]
})

st.write("### Detected / Simulated Behavioral Data")
st.dataframe(user_input)

# --- Predict ---
if st.button("Predict Age Group"):
    pred_num = model.predict(user_input)
    pred_label = label_encoder.inverse_transform(pred_num)[0]
    st.success(f"Predicted Age Group: {pred_label}")

    # --- SHAP Explainability Section ---

    st.subheader("🔍 Feature Importance via SHAP")

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    X_preprocessed = model.named_steps['preprocessor'].transform(user_input)
    shap_values = explainer.shap_values(X_preprocessed)

    # Handle multiclass vs binary output
    if isinstance(shap_values, list):
        shap_values_stacked = np.stack(shap_values, axis=0)
        mean_abs_shap = np.mean(np.abs(shap_values_stacked), axis=(0, 1))
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # --- Get readable feature names ---
    # Extract the names of numeric features from your preprocessing pipeline
    feature_names = model.named_steps['preprocessor'] \
        .transformers_[0][2]  # retrieves the original numeric feature column names

    # Align dimensions
    mean_abs_shap = np.ravel(mean_abs_shap)
    feature_names = np.ravel(feature_names)
    min_len = min(len(feature_names), len(mean_abs_shap))
    feature_names = feature_names[:min_len]
    mean_abs_shap = mean_abs_shap[:min_len]

    # Create DataFrame for plotting
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": mean_abs_shap
    }).sort_values("SHAP Value", ascending=False)

    # Plot interactive SHAP bar chart
    fig = px.bar(
        shap_df,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title="Feature Importance (SHAP Explanation)",
        color="SHAP Value",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Explain the chart dynamically ---
    top_feature = shap_df.iloc[0]["Feature"]
    bottom_feature = shap_df.iloc[-1]["Feature"]

    st.markdown(f"""
    **How to Read This Chart:**
    - Each bar represents how much a particular feature influenced the model’s prediction.
    - The **longer the bar**, the greater the feature’s impact on the predicted age group.
    - SHAP values represent the *magnitude* of influence, regardless of direction (positive or negative).
    - For this input:
    - The most influential feature was **{top_feature}**, meaning it played the biggest role in determining the age group.
    - The least influential feature was **{bottom_feature}**, meaning it had minimal impact on this prediction.
    """)



    # # ===============================
    # # SHAP Explainability (with shape fix)
    # # ===============================
    # st.subheader("🔍 Feature Importance via SHAP")

    # try:
    #     # Preprocess input
    #     X_preprocessed = model.named_steps['preprocessor'].transform(user_input)
    #     explainer = shap.TreeExplainer(model.named_steps['classifier'])
    #     shap_values = explainer.shap_values(X_preprocessed)

    #     # Handle multi-class or binary outputs
    #     if isinstance(shap_values, list):
    #         shap_values_stacked = np.stack(shap_values, axis=0)
    #         mean_abs_shap = np.mean(np.abs(shap_values_stacked), axis=(0, 1))
    #     else:
    #         mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    #     # --- Build SHAP DataFrame for plotting ---
    #     feature_names = model.named_steps['preprocessor'].transformers_[0][1] \
    #                         .named_steps['scaler'].get_feature_names_out()

    #     # Ensure both arrays are 1D
    #     mean_abs_shap = np.ravel(mean_abs_shap)
    #     feature_names = np.ravel(feature_names)

    #     # Align lengths (truncate or pad)
    #     min_len = min(len(feature_names), len(mean_abs_shap))
    #     feature_names = feature_names[:min_len]
    #     mean_abs_shap = mean_abs_shap[:min_len]

    #     # Build DataFrame safely
    #     shap_df = pd.DataFrame({
    #         "Feature": feature_names,
    #         "SHAP Value": mean_abs_shap
    #     }).sort_values("SHAP Value", ascending=False)

    #     # Plot
    #     fig = px.bar(
    #         shap_df,
    #         x="SHAP Value",
    #         y="Feature",
    #         orientation="h",
    #         title="Feature Importance (SHAP)"
    #     )
    #     st.plotly_chart(fig, use_container_width=True)

    # except Exception as e:
    #     st.error(f"Error generating SHAP visualization: {e}")
