import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65
REVERSION_FACTOR = 0.75

def load_game_data(file_path):
    return pd.read_excel(file_path)  # Reads Excel on startup

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

def probability_to_moneyline(prob):
    if prob >= 0.5:
        return f"-{round(100 * prob / (1 - prob))}"
    else:
        return f"+{round(100 * (1 - prob) / prob)}"

def get_vegasinsider_odds():
    url = "https://www.vegasinsider.com/nfl/odds/money-line/"
    try:
        tables = pd.read_html(url)
        odds_df = tables[0]
        odds_df.columns = odds_df.columns.droplevel(0)
        odds_df = odds_df.rename(columns={"Matchup": "Matchup", "Consensus": "Consensus"})
        return odds_df
    except Exception as e:
        st.error(f"Error scraping VegasInsider odds: {e}")
        return pd.DataFrame()

def find_team_odds(team_name, odds_df):
    for _, row in odds_df.iterrows():
        matchup = row["Matchup"]
        if team_name in matchup:
            return row["Consensus"]
    return "N/A"

# Streamlit App
st.set_page_config(page_title="NFL Elo Predictor", layout="wide")
st.title("🏈 NFL Bayesian Elo Prediction")
st.caption("Powered by Bayesian Elo ratings based on historical NFL games")

# Load Excel on app start (update the path if needed)
excel_file_path = "games.xlsx"  # <-- Replace with actual filename
try:
    df = load_game_data(excel_file_path)
    ratings = run_elo_pipeline(df)

    st.subheader("📊 Final Team Ratings (Bar Chart)")
    ratings_df = pd.DataFrame(ratings.items(), columns=["Team", "Rating"]).sort_values(by="Rating", ascending=False).set_index("Team")
    st.bar_chart(ratings_df)

    st.subheader("📋 Team Ratings Table")
    st.dataframe(ratings_df)

    with st.expander("📜 View Full Game Data"):
        st.dataframe(df)

    # --- Prediction Section ---
    st.markdown("---")
    st.header("🔮 Predict a Future Matchup")

    all_teams = sorted(ratings.keys(), key=lambda x: ratings[x], reverse=True)

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", all_teams)
    with col2:
        team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1])

    home_team = st.selectbox("🏠 Home Team", [team1, team2])

    def predict_matchup(team1, team2, home_team, elo_ratings):
        r1, r2 = elo_ratings.get(team1, BASE_ELO), elo_ratings.get(team2, BASE_ELO)
        if home_team == team1:
            r1 += HOME_ADVANTAGE
        elif home_team == team2:
            r2 += HOME_ADVANTAGE
        win_prob1 = expected_score(r1, r2)
        return win_prob1, 1 - win_prob1

    if st.button("Predict Winner"):
        prob1, prob2 = predict_matchup(team1, team2, home_team, ratings)
        odds1 = probability_to_moneyline(prob1)
        odds2 = probability_to_moneyline(prob2)

        # Get live odds from VegasInsider
        live_odds_df = get_vegasinsider_odds()
        live_odds_team1 = find_team_odds(team1, live_odds_df)
        live_odds_team2 = find_team_odds(team2, live_odds_df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"{team1} Win Probability", value=f"{prob1:.2%}", delta=f"Pred ML: {odds1} | Live ML: {live_odds_team1}")
        with col2:
            st.metric(label=f"{team2} Win Probability", value=f"{prob2:.2%}", delta=f"Pred ML: {odds2} | Live ML: {live_odds_team2}")

        st.markdown("### 🧠 Confidence Level")
        confidence = abs(prob1 - prob2)
        if confidence > 0.25:
            st.success("🔒 High confidence prediction")
        elif confidence > 0.15:
            st.info("🔍 Moderate confidence prediction")
        else:
            st.warning("⚠️ Low confidence — close matchup")

        # Value bet alerts
        st.markdown("### 💰 Value Bet Analysis")
        if live_odds_team1 != "N/A" and live_odds_team2 != "N/A":
            try:
                pred1_val = int(odds1.replace('+','').replace('-',''))
                live1_val = int(str(live_odds_team1).replace('+','').replace('-',''))
                if (odds1.startswith('-') and pred1_val < live1_val) or (odds1.startswith('+') and pred1_val > live1_val):
                    st.success(f"✅ Value on {team1}")
                pred2_val = int(odds2.replace('+','').replace('-',''))
                live2_val = int(str(live_odds_team2).replace('+','').replace('-',''))
                if (odds2.startswith('-') and pred2_val < live2_val) or (odds2.startswith('+') and pred2_val > live2_val):
                    st.success(f"✅ Value on {team2}")
            except:
                st.warning("⚠️ Could not compare odds numerically")

except FileNotFoundError:
    st.error(f"Excel file not found at `{excel_file_path}`. Please ensure the file exists.")
except Exception as e:
    st.error(f"Error loading data: {e}")
