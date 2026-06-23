# IPL Analytics Dashboard 🏏

A comprehensive data analytics dashboard analyzing 16 years of IPL cricket (2008–2024), 
built with Python and deployed live on Streamlit Cloud.

🔗 **Live Demo:** https://ipl-analytics-dashboard-utmplw7yunuazbw4jyhdd5.streamlit.app/

## What It Does

Six interactive analytical tabs powered by two datasets 
(1,090 matches + 260,000+ ball-by-ball deliveries):

| Tab | Description |
|-----|-------------|
| 📊 IPL Overview | Season trends, Orange/Purple cap tracker, nail-biter analysis |
| 🏏 Team Intelligence | Franchise health check, home/away splits, Hall of Fame |
| ⚔️ Rivalry Dossier | Head-to-head deep dive with MVP matrix and score distribution |
| 🏟️ Venue Intelligence | Phase-wise run rates, ground advantage, venue kings |
| 🤖 Match Predictor | ML-powered win probability with feature importance |
| 🚀 Impact Player Era | Statistical analysis of IPL's 2023 rule change |

## Technical Highlights

- **Dual-dataset pipeline:** Relational mapping between match-level and 
  ball-by-ball data via `match_id` joins
- **Advanced feature engineering:** Chronological H2H win rates, rolling 
  form metrics, home advantage detection — all computed without data leakage
- **Cricket-accurate statistics:** Proper handling of wides/no-balls in 
  strike rate and economy calculations; byes/leg-byes excluded from 
  bowler economy
- **ML Predictor:** RandomForest classifier (54% accuracy) trained on 
  9 engineered features including venue, era, and form — honest about 
  cricket's inherent unpredictability
- **Impact Player analysis:** Before/after 2023 comparison using wicket 
  rates per 100 balls (not raw counts) to account for unequal sample sizes

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

## Data

- `matches.csv` — 1,090 IPL matches (2008–2024)
- `deliveries.csv` — 260,920 ball-by-ball records
- Source: Kaggle IPL Complete Dataset