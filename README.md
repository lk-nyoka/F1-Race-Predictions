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

## 🏁 Podium Lab — live predictions (v2)

The notebook grew into **Podium Lab**, an automated pipeline that predicts the
next race and updates itself through the race weekend:

- **Before the weekend** — form-only prediction (recent finishes, podium rate,
  team form, points this season)
- **After FP1, FP2, FP3** — practice best laps project the grid, blended with
  the form view (practice is noisy: fuel loads, test programmes)
- **After qualifying** — the real grid, with confirmed grid penalties applied
- **After the race** — the prediction is scored and logged, the new result is
  added, the model is retrained and the next race begins

Every run also re-tests a set of engineered features with a walk-forward
backtest and keeps only the ones that improve predictions on races they weren't
chosen on.

### Walk-forward backtest (85 races, 2023 → Azerbaijan 2026)

Each race is predicted by a model trained only on the races before it.

| | Podium spots called | Log loss (latest 16 races) |
|---|---|---|
| Front 3 on the grid (baseline) | 171 / 255 | — |
| Notebook features, leak-free | 174 / 255 | 0.2138 |
| **Current model** | **176 / 255** | **0.2087** |

### Feature search results

| Feature | Result |
|---|---|
| Grid × team strength | ✅ kept |
| Points per race this season | ✅ kept |
| Team development trend (upgrade proxy) | ❌ didn't improve predictions |
| Beating the teammate | ❌ |
| Long-run form (10 races) | ❌ |
| Record at this circuit | ❌ |
| Recency weighting (half-life 30 / 60 races) | ❌ |

### Fixes to the original pipeline

- **Calendar order** — sorting by `['Year', 'Race', 'Position']` ordered races
  alphabetically, so "last 5 races" meant the wrong races
- **No look-ahead** — `TeamStrength` averaged points over the whole dataset,
  including future races; replaced with team form over the previous 5 races
- **Team rebrands** — AlphaTauri → RB → Racing Bulls and Alfa Romeo → Sauber →
  Audi are tracked as one team
- **Fairer metric** — ~85% of drivers miss the podium in any race, so predicting
  "no podium" for everyone already scores ~85% accuracy. Podium spots called
  against the grid baseline is a fairer test

### News and upgrades

`intel.json` holds race-week news (upgrades, penalties, weather), refreshed
daily. Confirmed grid penalties are applied to the prediction. Everything else
is shown as context, because there's no history of upgrades in the data for
the model to learn from.

### Run it

```bash
pip install fastf1 pandas numpy scikit-learn
python update.py              # refresh data, sessions, model and page
python update.py --research   # force the feature search to re-run
```

`update.py` builds `podium-lab.html`, a self-contained page you can open in a
browser.

---

## 🎯 How It Works (notebook, v1)

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
- ✅ Constructor current form — done in v2 (team form over last 5 races)
- Qualifying gap feature — time delta from pole position
- Weather flag — rain probability affects race outcome significantly
- ✅ Recency weighting — tested in v2, didn't help yet (re-tested every race)
- Practice pace history — backfill FP lap times to learn how much practice really predicts

---

## 🗓️ Race Predictions Log

| Race | Predicted Podium | Actual Podium | Score |
|---|---|---|---|
| Miami 2026 | VER, ANT, LEC | ANT, NOR, PIA | 1/3 correct, 3/3 in top 7 |
| Bahrain 2026 (Sepang) | ANT, RUS, HAM (pre-weekend) | TBD | Live in Podium Lab |

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
├── f1.ipynb                      # Original prediction notebook (v1)
├── f1_race_results_full.csv      # Race results 2022 → latest race (auto-updated)
├── update.py                     # One command: refresh everything, rebuild the page
├── engine.py                     # Features, walk-forward backtest, feature search, live weekend
├── backfill.py                   # Adds newly finished races from FastF1
├── intel.json                    # Race-week news and confirmed grid penalties
├── state.json                    # Feature search results + score history
├── changelog.json                # Every update the page has shown
├── predictions/                  # Saved live predictions per race weekend
├── page.template.html            # Podium Lab page template
├── podium-lab.html               # Built page (open in a browser)
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