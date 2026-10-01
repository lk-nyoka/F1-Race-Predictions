"""Append any completed races missing from f1_race_results_full.csv (same schema as the notebook)."""
import logging
from datetime import datetime, timezone

import fastf1
import pandas as pd

logging.disable(logging.WARNING)
fastf1.Cache.enable_cache("cache")
CSV = "f1_race_results_full.csv"
COLS = ["DriverNumber", "Abbreviation", "TeamName", "GridPosition", "Position", "Points", "Year", "Race"]


def backfill(years=(2026,)):
    data = pd.read_csv(CSV)
    have = set(zip(data["Year"], data["Race"]))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    added = []
    for year in years:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in sched.iterrows():
            race_start = ev["Session5DateUtc"]
            if (year, ev["EventName"]) in have or pd.isna(race_start) or race_start > now:
                continue
            try:
                s = fastf1.get_session(year, ev["EventName"], "Race")
                s.load(laps=False, telemetry=False, weather=False, messages=False)
                r = s.results
                if r is None or r.empty or r["Position"].isna().all():
                    print(f"  not published yet: {year} {ev['EventName']}")
                    continue
                r = r[COLS[:6]].copy()
                r["Year"], r["Race"] = year, ev["EventName"]
                added.append(r)
                print(f"  added {year} {ev['EventName']}")
            except Exception as e:  # results not available yet
                print(f"  skipped {year} {ev['EventName']}: {e}")
    if added:
        data = pd.concat([data, *added], ignore_index=True)
        data.to_csv(CSV, index=False)
    return len(added)


if __name__ == "__main__":
    print("races added:", backfill())
