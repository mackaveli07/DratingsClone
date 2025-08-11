import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

def load_game_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

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

    elo_ratings[team1] = elo_ratings[team1] + K * (actual1 - expected1)
    elo_ratings[team2] = elo_ratings[team2] + K * (actual2 - expected2)

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

def moneyline_to_probability(ml):
    ml_str = str(ml)
    if ml_str.startswith('-'):
        ml_val = int(ml_str.replace('-', ''))
        return ml_val / (ml_val + 100)
    elif ml_str.startswith('+'):
        ml_val = int(ml_str.replace('+', ''))
        return 100 / (ml_val + 100)
    else:
        try:
            ml_val = int(ml_str)
            return 100 / (ml_val + 100)
        except:
            return 0.5  # fallback

def probability_to_spread(prob, team_is_favorite=True):
    b = 0.23
    prob = max(min(prob, 0.999), 0.001)
    spread = np.log(prob / (1 - prob)) / b
    spread = round(spread * 2) / 2
    if not team_is_favorite:
        spread = -spread
    if spread == 0:
        return "PK"
    return spread

def spread_to_probability(spread):
    b = 0.23
    spread_calc = -spread
    prob = 1 / (1 + np.exp(-b * spread_calc))
    return prob

def format_edge_text(edge):
    threshold = 0.05
    if edge > threshold:
        return f'<span style="color:green;">(+{edge:.2%} edge)</span>'
    elif edge < -threshold:
        return f'<span style="color:red;">({edge:.2%} negative edge)</span>'
    return ""

# --- OddsShark integration ---

def get_oddsshark_odds():
    url = "https://www.oddsshark.com/nfl/odds"
    try:
        tables = pd.read_html(url)
        # OddsShark has multiple tables, the first is usually the main odds table
        odds_df = tables[0]
        return odds_df
    except Exception as e:
        st.error(f"Error scraping OddsShark: {e}")
        return pd.DataFrame()

def find_team_line_odds_shark(team_name, odds_df, column):
    # OddsShark uses 'Teams' column with matchups like "TeamA TeamB"
    team_name_lower = team_name.lower()
    for _, row in odds_df.iterrows():
        teams_str = str(row.get("Teams", "")).lower()
        if team_name_lower in teams_str:
            # Return the value from requested column
            val = row.get(column, "N/A")
            return val
    return "N/A"

# --- Streamlit app ---

st.set_page_config(page_title="NFL Elo Predictor with OddsShark", layout="wide")
st.title("🏈 NFL Bayesian Elo Prediction + OddsShark Live Odds")
st.caption("Powered by Bayesian Elo ratings and live odds from OddsShark.com")

excel_file_path = "games.xlsx"
try:
    historical_df = load_game_data(excel_file_path, sheet_name="games")
    schedule_2025_df = load_game_data(excel_file_path, sheet_name="2025 schedule")

    ratings = run_elo_pipeline(historical_df)

    st.subheader("📊 Final Team Ratings (Bar Chart)")
    ratings_df = pd.DataFrame(ratings.items(), columns=["Team", "Rating"]).sort_values(by="Rating", ascending=False).set_index("Team")
    st.bar_chart(ratings_df)

    st.subheader("📋 Team Ratings Table")
    st.dataframe(ratings_df)

    with st.expander("📜 View Historical Game Data"):
        st.dataframe(historical_df)

    st.markdown("---")
    st.header("🔮 Predict a 2025 Regular Season Matchup")

    season_2025_games = schedule_2025_df[schedule_2025_df['week'] <= 18]
    game_options = season_2025_games.apply(
        lambda x: f"Week {x['week']}: {x['team1']} vs {x['team2']}", axis=1
    ).tolist()

    selected_game = st.selectbox("Select Game", game_options)

    if selected_game:
        week_str, teams_str = selected_game.split(": ")
        team1, team2 = teams_str.split(" vs ")
        home_team = season_2025_games[
            (season_2025_games['team1'] == team1) & (season_2025_games['team2'] == team2)
        ]['home_team'].values[0]

        def predict_matchup(team1, team2, home_team, elo_ratings):
            r1, r2 = elo_ratings.get(team1, BASE_ELO), elo_ratings.get(team2, BASE_ELO)
            if home_team == team1:
                r1 += HOME_ADVANTAGE
            elif home_team == team2:
                r2 += HOME_ADVANTAGE
            win_prob1 = expected_score(r1, r2)
            return win_prob1, 1 - win_prob1

        prob1, prob2 = predict_matchup(team1, team2, home_team, ratings)
        odds1 = probability_to_moneyline(prob1)
        odds2 = probability_to_moneyline(prob2)
        spread1 = probability_to_spread(prob1, prob1 > prob2)
        spread2 = probability_to_spread(prob2, prob2 > prob1)

        # Fetch live odds from OddsShark
        odds_df = get_oddsshark_odds()

        live_ml_team1 = find_team_line_odds_shark(team1, odds_df, "Moneyline")
        live_ml_team2 = find_team_line_odds_shark(team2, odds_df, "Moneyline")
        live_spread_team1 = find_team_line_odds_shark(team1, odds_df, "Spread")
        live_spread_team2 = find_team_line_odds_shark(team2, odds_df, "Spread")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{team1} Win Probability",
                value=f"{prob1:.2%}",
                delta=f"Pred ML: {odds1} | Spread: {spread1} | Live ML: {live_ml_team1} | Live Spread: {live_spread_team1}"
            )
        with col2:
            st.metric(
                label=f"{team2} Win Probability",
                value=f"{prob2:.2%}",
                delta=f"Pred ML: {odds2} | Spread: {spread2} | Live ML: {live_ml_team2} | Live Spread: {live_spread_team2}"
            )

        st.markdown("### 🧠 Confidence Level")
        confidence = abs(prob1 - prob2)
        if confidence > 0.25:
            st.success("🔒 High confidence prediction")
        elif confidence > 0.15:
            st.info("🔍 Moderate confidence prediction")
        else:
            st.warning("⚠️ Low confidence — close matchup")

        st.markdown("### 💰 Value Bet Analysis")

        # Moneyline value check
        try:
            vegas_prob1 = moneyline_to_probability(live_ml_team1)
            edge1 = prob1 - vegas_prob1
            edge_text1 = format_edge_text(edge1)
            if edge_text1:
                st.markdown(f"✅ ML Value on {team1} {edge_text1}", unsafe_allow_html=True)

            vegas_prob2 = moneyline_to_probability(live_ml_team2)
            edge2 = prob2 - vegas_prob2
            edge_text2 = format_edge_text(edge2)
            if edge_text2:
                st.markdown(f"✅ ML Value on {team2} {edge_text2}", unsafe_allow_html=True)
        except:
            st.warning("⚠️ Could not compare moneylines numerically")

        # Spread value check
        try:
            def parse_spread(s):
                if isinstance(s, str):
                    s = s.replace('½', '.5').replace('+', '').replace('-', '')
                return float(s)
            vegas_spread1 = parse_spread(live_spread_team1)
            vegas_prob1_spread = spread_to_probability(vegas_spread1)
            spread_edge1 = prob1 - vegas_prob1_spread
            edge_text_s1 = format_edge_text(spread_edge1)
            if edge_text_s1:
                st.markdown(f"📏 Spread Value on {team1} {edge_text_s1}", unsafe_allow_html=True)

            vegas_spread2 = parse_spread(live_spread_team2)
            vegas_prob2_spread = spread_to_probability(vegas_spread2)
            spread_edge2 = prob2 - vegas_prob2_spread
            edge_text_s2 = format_edge_text(spread_edge2)
            if edge_text_s2:
                st.markdown(f"📏 Spread Value on {team2} {edge_text_s2}", unsafe_allow_html=True)
        except:
            st.warning("⚠️ Could not compare spreads numerically")

except FileNotFoundError:
    st.error(f"File not found: {excel_file_path}")
except Exception as e:
    st.error(f"An error occurred: {e}")
