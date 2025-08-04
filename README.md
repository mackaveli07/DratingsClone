# NFL Bayesian Elo Prediction App

A Streamlit web app that calculates and visualizes team ratings using a Bayesian Elo model, predicts game outcomes, and simulates season win totals.

## Features

- Bayesian Elo rating updates based on weekly NFL results
- Win probability and margin prediction for games
- Season simulation with average wins per team
- Interactive dashboard built with Streamlit

## Demo

Upload your own `games.csv` file to see live predictions.

## Data Format

Your CSV should look like this:

```csv
season,week,date,team1,team2,score1,score2,home_team
2022,1,2022-09-11,BUF,LAR,31,10,LAR
```

## Run It Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dependencies

- streamlit
- pandas
- numpy
