import streamlit as st
import pandas as pd
import joblib
import base64
import os

# ===============================
# Function to set background
# ===============================
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        /* Import Orbitron font */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        /* Background Image with Dark Overlay */
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                        url(data:image/png;base64,{encoded});
            background-size: cover;
            background-position: center;
        }}

        /* Title Styling */
        h1 {{
            font-family: 'Orbitron', sans-serif !important;
            color: #FFD700 !important; /* Gold */
            text-align: center;
            font-size: 50px !important;
            font-weight: bold !important;
            text-shadow: 3px 3px 10px black !important;
        }}

        /* Label Styling */
        label, .stSelectbox label, .stNumberInput label {{
            color: #FFFFFF !important;  
            font-size: 18px !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 5px black;
            font-family: 'Orbitron', sans-serif !important;
        }}

        /* Transparent Input Boxes */
        div[data-baseweb="input"], div[data-baseweb="select"] {{
            background-color: rgba(255, 255, 255, 0.7) !important;
            border-radius: 10px;
            padding: 5px;
            border: 1px solid #ccc;
            font-family: 'Orbitron', sans-serif !important;
        }}

        /* Transparent Button Styling */
        button[kind="primary"] {{
            background-color: rgba(0, 123, 255, 0.7) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
            transition: all 0.3s ease-in-out;
            font-family: 'Orbitron', sans-serif !important;
        }}
        button[kind="primary"]:hover {{
            background-color: rgba(0, 123, 255, 0.9) !important;
            transform: scale(1.05);
        }}

        /* Result Box Styling */
        .result-box {{
            background: rgba(0, 0, 0, 0.6);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.8);
        }}
        .win {{
            color: #00FF00 !important; /* Bright Green */
            font-size: 22px;
            font-weight: bold;
            font-family: 'Orbitron', sans-serif !important;
        }}
        .loss {{
            color: #FF4C4C !important; /* Bright Red */
            font-size: 22px;
            font-weight: bold;
            font-family: 'Orbitron', sans-serif !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



# Call background function
set_background("bg.jpg")  # Change bg.jpg to your image name

# ===============================
# Load model and columns
# ===============================
BASE = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE, "ipl_model.pkl"))
model_columns = joblib.load(os.path.join(BASE, "model_columns.pkl"))

teams = [
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", 
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore", 
    "Sunrisers Hyderabad"
]

cities = [
    "Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Hyderabad", 
    "Jaipur", "Kolkata", "Lucknow", "Mumbai", "Pune", "Abu Dhabi", "Dubai"
]

# ===============================
# Streamlit UI
# ===============================
st.markdown('<h1 class="stTitle" style="font-family:Orbitron, sans-serif;">🏏 IPL Winning Team Predictor</h1>', unsafe_allow_html=True)


batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
city = st.selectbox("City", cities)
runs_left = st.number_input("Runs Left", min_value=0, max_value=300, value=50)
balls_left = st.number_input("Balls Left", min_value=0, max_value=300, value=24)
wickets_remaining = st.number_input("Wickets Remaining", min_value=0, max_value=10, value=5)
total_runs = st.number_input("Total Runs (Target)", min_value=0, max_value=300, value=180)
crr = st.number_input("Current Run Rate (CRR)", min_value=0.0, max_value=20.0, value=7.5)
rrr = st.number_input("Required Run Rate (RRR)", min_value=0.0, max_value=20.0, value=10.5)

if st.button("Predict", key="predict_button"):
    input_data = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_remaining": wickets_remaining,
        "total_run_x": total_runs,
        "crr": crr,
        "rrr": rrr
    }

    df = pd.DataFrame([input_data])
    categorical_cols = ["batting_team", "bowling_team", "city"]
    df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_enc = df_enc.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(df_enc)[0]
    loss_prob, win_prob = prob[0], prob[1]

    st.markdown(
    f"""
    <div class="result-box">
        <h3 class="result-title" style="color: white;">📊 Prediction Results</h3>
        <p class="win">Win Probability: {win_prob:.2f}</p>
        <p class="loss">Loss Probability: {loss_prob:.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
)