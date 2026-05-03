# 🏎️ F1 Race Podium Prediction

A machine learning system that predicts Formula 1 podium finishers
using historical race data and real qualifying results via the FastF1 API.
Built as part of my Data Science learning journey — updated every race weekend.

---

## 📌 Project Overview

This project builds an end-to-end ML pipeline that predicts which drivers
will finish on the podium (Top 3) for each Formula 1 Grand Prix. Rather
than predicting a single winner, the model frames it as a binary
classification problem — Top 3 or not — then ranks all drivers by
podium probability.

The model only makes predictions after official race qualifying is
complete, ensuring real grid positions are used rather than placeholders.

---

## 🎯 How It Works

1. Historical race data is collected via the FastF1 API across 2022-2026
2. The model is trained on 4 seasons of race results
3. On race weekend, qualifying results are loaded automatically
4. Each driver is assigned a podium probability
5. Top 3 probabilities become the predicted podium
6. After the race, predictions are evaluated against actual results

---

## 📊 Model Performance

**Test set — 2026 season (3 races):**
- Accuracy: 89%
- Podium Recall: 100% — caught every podium finisher in test data

**Miami Grand Prix 2026 — Live Prediction vs Actual:**

| Position | Predicted | Actual |
|---|---|---|
| 🥇 | VER | ANT |
| 🥈 | ANT | NOR |
| 🥉 | LEC | PIA |

All 3 actual podium finishers (ANT, NOR, PIA) appeared in our
top 7 probability rankings. The model correctly identified the
right drivers — the order was off due to VER's dominant 2022
historical data outweighing his poor 2026 current form.

---

## 🔑 Features Used

- GridPosition — official qualifying grid position
- GridAdvantage — inverse of grid position (pole = highest advantage)
- RecentForm — rolling average finishing position over last 5 races
- RecentPodiumRate — podium rate over last 5 races
- TeamStrength — average points per race for the constructor
- RaceCount — total career races in dataset (experience proxy)

---

## 🛠️ What Was Done

- Collected 1,904 driver-race records via FastF1 API — 2022 to 2026
- Cleaned and handled missing grid positions and DNF results
- Engineered 4 features from raw race data
- Applied time-based train/test split — no data leakage
- Built Random Forest classifier with class_weight balanced
- Achieved 100% podium recall on 2026 test data
- Built automated qualifying loader — blocks prediction until quali is done
- Extracted real qualifying grid from lap times when official results unavailable
- Made live prediction for Miami 2026 using real pole to P22 grid

---

## 🔮 Upcoming Improvements

- XGBoost model — stronger probability calibration than Random Forest
- Constructor current form — last 3 race average instead of all-time
- Qualifying gap feature — time delta from pole position
- Weather flag — rain probability affects race outcome significantly
- Recency weighting — reduce influence of seasons older than 2 years

---

## 🗓️ Race Predictions Log

| Race | Predicted Podium | Actual Podium | Score |
|---|---|---|---|
| Miami 2026 | VER, ANT, LEC | ANT, NOR, PIA | 1/3 correct, 3/3 in top 7 |
| Monaco 2026 | TBD | TBD | TBD |

---

## 🧰 Tools & Libraries

- Python 3
- Pandas
- NumPy
- Scikit-learn
- FastF1
- Matplotlib
- Jupyter Notebook

---

## 📁 File Structure
F1/
│
├── f1.ipynb                      # Main prediction notebook
├── f1_race_results_full.csv      # Collected race data 2022-2026
├── .gitignore                    # Excludes cache and model files
└── README.md                     # Project documentation

---

## ⚠️ Note

FastF1 cache files and trained model files are excluded from this
repository due to size. Run the data collection cells to rebuild
the dataset locally before making predictions.

---

## 👤 Author

**Lindokuhle Nyoka**
[GitHub](https://github.com/lk-nyoka) · [LinkedIn](https://linkedin.com/in/lindokuhle-nyoka-982019245)