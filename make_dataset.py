import pandas as pd
import numpy as np
import argparse
import os

def generate_data(n=5000, random_state=42):
    np.random.seed(random_state)

    # Age groups
    age_groups = ["13-15", "16-18", "19-22"]
    age_probs = [0.3, 0.4, 0.3]

    # SES tiers
    ses_tiers = ["low", "mid", "high"]
    ses_probs = [0.3, 0.5, 0.2]

    # Gender
    genders = ["male", "female"]
    gender_probs = [0.5, 0.5]

    data = []
    for _ in range(n):
        age = np.random.choice(age_groups, p=age_probs)
        ses = np.random.choice(ses_tiers, p=ses_probs)
        gender = np.random.choice(genders, p=gender_probs)

        # Feature correlations
        if age == "13-15":
            typing_wpm = np.random.normal(35, 8)
            login_hour = np.random.choice(range(7, 22))
            session_duration = np.random.normal(25, 5)
        elif age == "16-18":
            typing_wpm = np.random.normal(50, 10)
            login_hour = np.random.choice(range(8, 23))
            session_duration = np.random.normal(30, 7)
        else:
            typing_wpm = np.random.normal(65, 12)
            login_hour = np.random.choice(range(10, 24))
            session_duration = np.random.normal(40, 10)

        quiz_score = np.clip(np.random.normal(75, 10), 0, 100)
        time_per_question = np.random.normal(30, 5)
        sessions_per_day = np.random.uniform(1, 4)
        attempts_per_quiz = np.random.uniform(1, 2)
        avg_gap_mins = np.random.uniform(0.5, 3.0)
        device_type = np.random.choice(["Android", "iOS", "Desktop", "Shared/Other"])
        browser = np.random.choice(["Chrome", "Safari", "Firefox", "Edge", "In-App"])

        data.append([
            login_hour, typing_wpm, quiz_score, time_per_question,
            session_duration, sessions_per_day, attempts_per_quiz, avg_gap_mins,
            device_type, browser, age, gender, ses
        ])

    columns = [
        "login_hour", "typing_wpm", "quiz_score", "time_per_question_sec",
        "session_duration_min", "sessions_per_day", "attempts_per_quiz",
        "avg_gap_mins", "device_type", "browser",
        "age_group", "gender", "ses_tier"
    ]

    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    df = generate_data(args.n)
    df.to_csv("data/synth_edtech.csv", index=False)
    print(f"Generated dataset with {args.n} rows at data/synth_edtech.csv")
