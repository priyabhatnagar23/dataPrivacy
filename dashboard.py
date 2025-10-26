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

# Shorter target text (single sentence)
target_text = "The quick brown fox jumps over the lazy dog near the riverbank."

# Initialize state
if "typing_test" not in st.session_state:
    st.session_state.typing_test = {
        "started": False,
        "start_time": None,
        "completed": False
    }

test_state = st.session_state.typing_test

# Instructions and target text
st.text_area("Text to type:", value=target_text, height=80, disabled=True)

# Start button
if not test_state["started"]:
    st.info("👇 Click 'Start Test' below, then immediately start typing in the text box.")
    if st.button("▶️ Start Test", type="primary"):
        test_state["started"] = True
        test_state["start_time"] = time.time()
        st.rerun()

# Typing area (only shown after start)
if test_state["started"] and not test_state["completed"]:
    typed_text = st.text_area(
        "Type here:", 
        key="typed_area", 
        height=80,
        placeholder="Start typing now..."
    )
    
    current_length = len(typed_text)
    target_length = len(target_text)
    
    # Show progress
    if current_length > 0:
        progress = min(current_length / target_length, 1.0)
        st.progress(progress, text=f"Progress: {current_length}/{target_length} characters ({int(progress*100)}%)")
        
        # Calculate live WPM
        elapsed = time.time() - test_state["start_time"]
        if elapsed > 0:
            live_wpm = (current_length / 5) / (elapsed / 60)
            st.caption(f"⏱️ Current speed: ~{round(live_wpm, 1)} WPM | Time: {round(elapsed, 1)}s")
    
    # Finish button (enabled when 90% complete for shorter text)
    completion_threshold = target_length * 0.90
    
    if current_length >= completion_threshold:
        st.success("✅ You've typed enough! Click 'Finish Test' to see your results.")
        if st.button("🏁 Finish Test", type="primary"):
            # Calculate final metrics
            end_time = time.time()
            elapsed_seconds = end_time - test_state["start_time"]
            elapsed_minutes = elapsed_seconds / 60
            
            # Compare character by character
            min_compare_length = min(len(typed_text), len(target_text))
            correct_chars = sum(1 for i in range(min_compare_length) if typed_text[i] == target_text[i])
            incorrect_chars = min_compare_length - correct_chars
            
            # Accuracy
            accuracy = (correct_chars / min_compare_length) * 100 if min_compare_length > 0 else 0
            
            # WPM (correct characters only)
            wpm = (correct_chars / 5) / elapsed_minutes if elapsed_minutes > 0 else 0
            
            # Raw WPM (all characters)
            raw_wpm = (current_length / 5) / elapsed_minutes if elapsed_minutes > 0 else 0
            
            # Save results
            st.session_state.typing_results = {
                "wpm": round(wpm, 1),
                "raw_wpm": round(raw_wpm, 1),
                "accuracy": round(accuracy, 1),
                "elapsed_seconds": round(elapsed_seconds, 1),
                "correct_chars": correct_chars,
                "incorrect_chars": incorrect_chars,
                "total_chars": current_length
            }
            
            test_state["completed"] = True
            st.rerun()
    else:
        remaining = int(completion_threshold - current_length)
        st.info(f"📝 Type at least {remaining} more characters to finish the test.")

# Display results
if test_state["completed"] and "typing_results" in st.session_state:
    r = st.session_state.typing_results
    
    st.success("🎉 Test Complete!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("WPM", r['wpm'], help="Correct words per minute")
    with col2:
        st.metric("Raw WPM", r['raw_wpm'], help="All typed characters per minute")
    with col3:
        st.metric("Accuracy", f"{r['accuracy']}%", help="Percentage of correct characters")
    
    st.info(f"⏱️ Time taken: {r['elapsed_seconds']} seconds")
    st.info(f"📊 Characters: {r['correct_chars']} correct / {r['incorrect_chars']} incorrect / {r['total_chars']} total")

# Reset button
if test_state["started"]:
    if st.button("🔄 Reset and Try Again"):
        st.session_state.typing_test = {
            "started": False,
            "start_time": None,
            "completed": False
        }
        if "typing_results" in st.session_state:
            del st.session_state.typing_results
        st.rerun()

# ===============================
# 🧩 Mini Cognitive Quiz Section
# ===============================
st.subheader("🧠 Quick Cognitive Quiz (Simulated EdTech Interaction)")

quiz_data = [
    {
        "question": "Which of the following words doesn't belong?",
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

# --- Get typing WPM from test results ---
typing_wpm_value = (
    st.session_state.typing_results["wpm"]
    if "typing_results" in st.session_state
    else 40  # fallback default
)

# --- Create DataFrame ---
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

    # Compute SHAP values once
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    X_preprocessed = model.named_steps['preprocessor'].transform(user_input)
    shap_values = explainer.shap_values(X_preprocessed)

    # Save everything to session state
    st.session_state["prediction"] = pred_label
    st.session_state["pred_num"] = pred_num
    st.session_state["shap_values"] = shap_values
    st.session_state["feature_names"] = model.named_steps['preprocessor'].transformers_[0][2]

# --- Display SHAP Section (if prediction exists) ---
if "prediction" in st.session_state:
    st.subheader("🔍 Feature Importance via SHAP")

    shap_values = st.session_state["shap_values"]
    feature_names = st.session_state["feature_names"]
    pred_label = st.session_state["prediction"]
    pred_num = st.session_state["pred_num"]

    # --- Toggle ---
    show_global = st.toggle("🌍 Show Global Explanation (Across All Classes)", key="show_global", value=False)

    # Handle multiclass/binary
    is_multiclass = isinstance(shap_values, list)

    # --- LOCAL ---
    if not show_global:
        predicted_class = pred_num[0]
        predicted_class_name = pred_label

        if is_multiclass:
            local_shap = shap_values[predicted_class][0]
        else:
            local_shap = shap_values[0]

        local_shap = np.ravel(local_shap)
        feature_names = np.ravel(feature_names)
        min_len = min(len(feature_names), len(local_shap))
        shap_df = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "SHAP Value": np.abs(local_shap[:min_len])
        }).sort_values("SHAP Value", ascending=False)

        fig = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="SHAP Value",
            color_continuous_scale="Blues",
            title=f"Feature Importance for Predicted Age Group: {predicted_class_name}"
        )
        st.plotly_chart(fig, use_container_width=True)

        top_feature = shap_df.iloc[0]["Feature"]
        bottom_feature = shap_df.iloc[-1]["Feature"]
        st.markdown(f"""
        **Interpretation (Local SHAP for Predicted Class — {predicted_class_name}):**
        - Each bar shows how much a feature influenced this *specific* prediction.
        - The **longer the bar**, the stronger the influence.
        - Most influential: **{top_feature}**; Least influential: **{bottom_feature}**.
        """)

    # --- GLOBAL ---
    else:
        if is_multiclass:
            shap_stack = np.stack([np.abs(s) for s in shap_values], axis=0)
            global_shap = np.mean(shap_stack, axis=(0, 1))
        else:
            global_shap = np.mean(np.abs(shap_values), axis=0)

        global_shap = np.ravel(global_shap)
        feature_names = np.ravel(feature_names)
        min_len = min(len(feature_names), len(global_shap))
        shap_df_global = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "Mean |SHAP Value|": global_shap[:min_len]
        }).sort_values("Mean |SHAP Value|", ascending=False)

        fig = px.bar(
            shap_df_global,
            x="Mean |SHAP Value|",
            y="Feature",
            orientation="h",
            color="Mean |SHAP Value|",
            color_continuous_scale="Viridis",
            title="🌍 Global Feature Importance (Across All Age Groups)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretation (Global SHAP):**
        - Aggregates SHAP values across all possible age groups.
        - Shows which features matter **most on average**.
        - The **longer the bar**, the stronger its global influence.
        """)