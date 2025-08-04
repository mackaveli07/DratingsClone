import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65
REVERSION_FACTOR = 0.75

def load_game_data(file_path):
    return pd.read_csv(file_path)

def group_games_by_week(df):
    return df.groupby(["season", "week"])

def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]
    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE

    expected1 = expected_score(r1, r2)
    expected2 = expected_score(r2, r1)

    actual1 = 1 if score1 > score2 else 0
    actual2 = 1 - actual1

    new_r1 = elo_ratings[team1] + K * (actual1 - expected1)
    new_r2 = elo_ratings[team2] + K * (actual2 - expected2)

    elo_ratings[team1] = new_r1
    elo_ratings[team2] = new_r2

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    grouped = group_games_by_week(df)
    for (season, week), games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
    return dict(elo_ratings)



st.title("NFL Bayesian Elo Prediction")

uploaded_file = st.file_uploader("Upload NFL games CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    ratings = run_elo_pipeline(df)

    st.subheader("Final Team Ratings")
    st.dataframe(pd.DataFrame(ratings.items(), columns=["Team", "Rating"]).sort_values(by="Rating", ascending=False))

    # --- Prediction Section ---
    def predict_matchup(team1, team2, home_team, elo_ratings):
        r1, r2 = elo_ratings.get(team1, BASE_ELO), elo_ratings.get(team2, BASE_ELO)
        if home_team == team1:
            r1 += HOME_ADVANTAGE
        elif home_team == team2:
            r2 += HOME_ADVANTAGE
        win_prob1 = expected_score(r1, r2)
        return win_prob1, 1 - win_prob1

    st.markdown("---")
    st.header("Predict a Future Matchup")

    all_teams = sorted(ratings.keys(), key=lambda x: ratings[x], reverse=True)

    team1 = st.selectbox("Team 1", all_teams)
    team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1])
    home_team = st.selectbox("Home Team", [team1, team2])

    if st.button("Predict Winner"):
        prob1, prob2 = predict_matchup(team1, team2, home_team, ratings)
        st.success(f"**{team1}** win probability: **{prob1:.2%}**")
        st.success(f"**{team2}** win probability: **{prob2:.2%}**")
