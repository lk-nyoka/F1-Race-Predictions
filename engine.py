"""Podium Lab engine: features, walk-forward evaluation, feature search, live race-weekend prediction.

Everything here is leak-free: every feature for a race uses only races that finished before it.
"""
import logging
from datetime import datetime, timedelta, timezone

import fastf1
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logging.disable(logging.WARNING)
fastf1.Cache.enable_cache("cache")

CSV = "f1_race_results_full.csv"
# Rebrands keep one lineage so team form carries over (Toro Rosso family, Sauber family)
LINEAGE = {"AlphaTauri": "RB", "RB": "RB", "Racing Bulls": "RB",
           "Alfa Romeo": "Sauber", "Kick Sauber": "Sauber", "Audi": "Sauber"}

BASE = ["LogGrid", "RecentForm", "RecentPodiumRate", "TeamForm"]
CANDIDATES = ["TeamTrend", "TeammateDelta", "LongForm", "TrackHistory", "SeasonPts", "GridXTeam"]
GRID_FEATURES = {"LogGrid", "GridXTeam"}
LABELS = {
    "LogGrid": "Grid position", "RecentForm": "Recent finishes (5 races)",
    "RecentPodiumRate": "Recent podium rate", "TeamForm": "Team form (5 races)",
    "TeamTrend": "Team development trend", "TeammateDelta": "Beating the teammate",
    "LongForm": "Long-run form (10 races)", "TrackHistory": "Record at this circuit",
    "SeasonPts": "Points per race this season", "GridXTeam": "Grid × team strength",
}
DEFAULTS = {"RecentForm": 13.0, "RecentPodiumRate": 0.0, "TeamForm": 2.0, "TeamTrend": 0.0,
            "TeammateDelta": 0.0, "LongForm": 1.0}


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------- data
_sched_cache = {}


def schedule(year):
    if year not in _sched_cache:
        _sched_cache[year] = fastf1.get_event_schedule(year, include_testing=False)
    return _sched_cache[year]


def load_results():
    d = pd.read_csv(CSV).dropna(subset=["Position"]).copy()
    d["RaceIdx"] = d.groupby(["Year", "Race"], sort=False).ngroup()
    locs = {}
    for y in d["Year"].unique():
        s = schedule(int(y))
        locs.update({(int(y), r.EventName): r.Location for r in s.itertuples()})
    d["Location"] = [locs.get((int(y), r), r) for y, r in zip(d["Year"], d["Race"])]
    return d


def prepare(d):
    """Clean + engineer features. Rows with NaN Position (an upcoming race) get features but no target."""
    d = d.sort_values(["RaceIdx", "Position"], na_position="last").reset_index(drop=True)
    field = d.groupby("RaceIdx")["Abbreviation"].transform("size")
    d["GridPosition"] = d["GridPosition"].fillna(field)
    d.loc[d["GridPosition"] <= 0, "GridPosition"] = field
    d["Top3"] = np.where(d["Position"].notna(), (d["Position"] <= 3).astype(float), np.nan)
    d["Lineage"] = d["TeamName"].map(LINEAGE).fillna(d["TeamName"])

    def roll(g, col, n, fn="mean", shift=1):
        return g[col].transform(lambda s: s.shift(shift).rolling(n, min_periods=1).agg(fn))

    by_drv = d.groupby("Abbreviation")
    d["RecentForm"] = roll(by_drv, "Position", 5)
    d["RecentPodiumRate"] = roll(by_drv, "Top3", 5)
    d["LongForm"] = roll(by_drv, "Points", 10)

    # Team per-race average points, then rolling form and development trend
    tr = d.groupby(["Lineage", "RaceIdx"])["Points"].mean().reset_index().sort_values("RaceIdx")
    g = tr.groupby("Lineage")["Points"]
    tr["TeamForm"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    last3 = g.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    prev3 = g.transform(lambda s: s.shift(4).rolling(3, min_periods=2).mean())
    tr["TeamTrend"] = last3 - prev3
    d = d.merge(tr[["Lineage", "RaceIdx", "TeamForm", "TeamTrend"]], on=["Lineage", "RaceIdx"], how="left")

    # Driver points minus team average in the same race -> rolling (how much they beat the car)
    team_avg = d.groupby(["Lineage", "RaceIdx"])["Points"].transform("mean")
    d["_tm"] = d["Points"] - team_avg
    d = d.sort_values(["RaceIdx", "Position"], na_position="last").reset_index(drop=True)
    d["TeammateDelta"] = d.groupby("Abbreviation")["_tm"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # Circuit record: mean finish at this venue in earlier visits
    d["TrackHistory"] = d.groupby(["Abbreviation", "Location"])["Position"].transform(
        lambda s: s.shift(1).expanding().mean())
    d["SeasonPts"] = d.groupby(["Abbreviation", "Year"])["Points"].transform(lambda s: s.shift(1).expanding().mean())

    for c, v in DEFAULTS.items():
        d[c] = d[c].fillna(v)
    d["TrackHistory"] = d["TrackHistory"].fillna(d["RecentForm"])
    d["SeasonPts"] = d["SeasonPts"].fillna(d["LongForm"])
    d["LogGrid"] = np.log(d["GridPosition"].clip(lower=1))
    d["GridXTeam"] = d["LogGrid"] * d["TeamForm"]
    return d.drop(columns=["_tm"])


# ---------------------------------------------------------------- model
def fit(train, feats, half_life=None, current_idx=None):
    mu, sd = train[feats].mean(), train[feats].std().replace(0, 1)
    w = None
    if half_life:
        w = 0.5 ** ((current_idx - train["RaceIdx"]) / half_life)
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit((train[feats] - mu) / sd, train["Top3"].astype(int), sample_weight=w)
    return {"feats": feats, "mu": mu, "sd": sd, "m": m}


def predict(model, rows):
    X = (rows[model["feats"]] - model["mu"]) / model["sd"]
    return model["m"].predict_proba(X)[:, 1]


def walk_forward(d, feats, half_life=None, first_year=2023, keep_probs=False):
    done = d[d["Top3"].notna()]
    out = []
    for idx in sorted(done.loc[done["Year"] >= first_year, "RaceIdx"].unique()):
        train, test = done[done["RaceIdx"] < idx], done[done["RaceIdx"] == idx].copy()
        test["p"] = predict(fit(train, feats, half_life, idx), test)
        actual = set(test.loc[test["Top3"] == 1, "Abbreviation"])
        pred = set(test.nlargest(3, "p")["Abbreviation"])
        grid = set(test.nsmallest(3, "GridPosition")["Abbreviation"])
        p = test["p"].clip(1e-6, 1 - 1e-6)
        ll = -np.mean(test["Top3"] * np.log(p) + (1 - test["Top3"]) * np.log(1 - p))
        rec = {"idx": int(idx), "hits": len(pred & actual), "grid": len(grid & actual), "ll": float(ll)}
        if keep_probs:
            rec["test"], rec["model"] = test, fit(train, feats, half_life, idx)
        out.append(rec)
    return out


def score(runs, idxs=None):
    rs = [r for r in runs if idxs is None or r["idx"] in idxs]
    return {"races": len(rs), "hits": sum(r["hits"] for r in rs), "grid": sum(r["grid"] for r in rs),
            "logloss": round(float(np.mean([r["ll"] for r in rs])), 4) if rs else None}


def feature_search(d, holdout_n=16):
    """Greedy forward selection judged on log loss over a selection window; reported on a later holdout.

    As races are added, the holdout slides forward, so the search keeps re-testing ideas on fresh races.
    """
    done_idx = sorted(d.loc[d["Top3"].notna() & (d["Year"] >= 2023), "RaceIdx"].unique())
    select_idx, hold_idx = set(done_idx[:-holdout_n]), set(done_idx[-holdout_n:])
    cache = {}

    def run(feats, hl):
        key = (tuple(feats), hl)
        if key not in cache:
            cache[key] = walk_forward(d, list(feats), hl)
        return cache[key]

    trials = []
    best, best_ll = list(BASE), score(run(BASE, None), select_idx)["logloss"]
    trials.append({"change": "Starting point (notebook features, fixed)", "feats": list(BASE), "hl": None,
                   "select": best_ll, "kept": True})
    improved = True
    while improved:
        improved = False
        cand_results = []
        for c in [c for c in CANDIDATES if c not in best]:
            ll = score(run(best + [c], None), select_idx)["logloss"]
            cand_results.append((ll, c))
            trials.append({"change": f"Add {LABELS[c]}", "feats": best + [c], "hl": None, "select": ll,
                           "kept": False})
        if cand_results:
            ll, c = min(cand_results)
            if ll < best_ll - 0.0005:
                best, best_ll, improved = best + [c], ll, True
                trials[-len(cand_results) + [x[1] for x in cand_results].index(c)]["kept"] = True
    best_hl = None
    for hl in (60, 30):
        ll = score(run(best, hl), select_idx)["logloss"]
        kept = ll < best_ll - 0.0005
        trials.append({"change": f"Weight recent races (half-life {hl} races)", "feats": best, "hl": hl,
                       "select": ll, "kept": kept})
        if kept:
            best_ll, best_hl = ll, hl
    base_runs, champ_runs = run(BASE, None), run(best, best_hl)
    return {
        "feats": best, "half_life": best_hl, "trials": trials,
        "select_races": len(select_idx), "holdout_races": len(hold_idx),
        "baseline": {"all": score(base_runs), "holdout": score(base_runs, hold_idx)},
        "champion": {"all": score(champ_runs), "holdout": score(champ_runs, hold_idx)},
    }


# ---------------------------------------------------------------- live weekend
def next_event(d):
    have = set(zip(d["Year"], d["Race"]))
    t = now_utc()
    for y in (t.year, t.year + 1):
        try:
            s = schedule(y)
        except Exception:
            continue
        for ev in s.sort_values("RoundNumber").itertuples():
            if (y, ev.EventName) not in have and pd.notna(ev.Session5DateUtc) and ev.Session5DateUtc > t - timedelta(days=4):
                return y, ev
    return None, None


def session_list(ev):
    return [(getattr(ev, f"Session{i}"), getattr(ev, f"Session{i}DateUtc")) for i in range(1, 6)]


def load_session(year, ev, name):
    try:
        s = fastf1.get_session(year, ev.EventName, name)
        s.load(laps=name != "Qualifying" or True, telemetry=False, weather=False, messages=False)
        return s
    except Exception as e:
        print(f"  {name}: not available ({e.__class__.__name__})")
        return None


def best_laps(s):
    laps = s.laps
    if laps is None or laps.empty:
        return None
    laps = laps[laps["LapTime"].notna()]
    if "Deleted" in laps:
        laps = laps[laps["Deleted"] != True]  # noqa: E712
    b = laps.groupby("Driver")["LapTime"].min().dt.total_seconds().dropna()
    if b.empty:
        return None
    return pd.DataFrame({"code": b.index, "best": b.values, "gap": (b.values / b.min() - 1) * 100})


def fmt_lap(sec):
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:06.3f}"


def live_weekend(d_raw, champion, intel):
    year, ev = next_event(d_raw)
    if ev is None:
        return None
    t = now_utc()
    last_idx = d_raw["RaceIdx"].max()
    last = d_raw[d_raw["RaceIdx"] == last_idx]
    entrants = {r.Abbreviation: r.TeamName for r in last.itertuples()}

    sessions, pace, grid = [], [], None
    for name, start in session_list(ev):
        if not isinstance(name, str) or not name or name == "Race":
            continue
        info = {"name": name, "startUtc": start.isoformat() + "Z" if pd.notna(start) else None, "loaded": False}
        if pd.notna(start) and start + timedelta(minutes=75) < t:
            s = load_session(year, ev, name)
            if s is not None:
                res = s.results
                if name == "Qualifying" and res is not None and res["Position"].notna().any():
                    q = res.dropna(subset=["Position"]).sort_values("Position")
                    grid = list(q["Abbreviation"])
                    entrants = {r.Abbreviation: r.TeamName for r in q.itertuples()}
                    info["loaded"] = True
                    bl = best_laps(s)
                    info["order"] = [{"code": c, "team": entrants.get(c, ""),
                                      "time": fmt_lap(float(bl.set_index("code").loc[c, "best"])) if bl is not None and c in set(bl["code"]) else None}
                                     for c in grid]
                else:
                    bl = best_laps(s)
                    if bl is not None:
                        info["loaded"] = True
                        teams = dict(zip(res["Abbreviation"], res["TeamName"])) if res is not None else {}
                        if name != "Practice 1":  # FP1 often has rookies standing in
                            entrants.update({c: teams.get(c, entrants.get(c, "")) for c in bl["code"] if c in teams})
                        bl = bl.sort_values("best")
                        info["order"] = [{"code": r.code, "team": teams.get(r.code, entrants.get(r.code, "")),
                                          "time": fmt_lap(r.best), "gap": round(r.gap, 3)} for r in bl.itertuples()]
                        pace.append((name, bl))
        sessions.append(info)

    # Upcoming race rows so features are computed from prior races only
    nxt_idx = last_idx + 1
    up = pd.DataFrame([{"DriverNumber": None, "Abbreviation": c, "TeamName": tm, "GridPosition": np.nan,
                        "Position": np.nan, "Points": np.nan, "Year": year, "Race": ev.EventName,
                        "RaceIdx": nxt_idx, "Location": ev.Location} for c, tm in entrants.items()])
    feats_all = prepare(pd.concat([d_raw, up], ignore_index=True))
    hist = feats_all[feats_all["Top3"].notna()]
    rows = feats_all[feats_all["RaceIdx"] == nxt_idx].copy().reset_index(drop=True)

    hl = champion["half_life"]
    full = fit(hist, champion["feats"], hl, nxt_idx)
    form_feats = [f for f in champion["feats"] if f not in GRID_FEATURES]
    form = fit(hist, form_feats, hl, nxt_idx)
    rows["p_form"] = predict(form, rows)

    penalties = {k: int(v) for k, v in (intel.get("grid_penalties") or {}).items()}

    def with_grid(order_codes):
        r = rows.copy()
        pos = {c: i + 1 for i, c in enumerate(order_codes)}
        missing = [c for c in r.sort_values("p_form", ascending=False)["Abbreviation"] if c not in pos]
        for c in missing:
            pos[c] = len(pos) + 1
        r["GridPosition"] = r["Abbreviation"].map(pos).astype(float)
        r["LogGrid"] = np.log(r["GridPosition"])
        r["GridXTeam"] = r["LogGrid"] * r["TeamForm"]
        return r

    def snapshot(key, label, p, r=None):
        r = rows if r is None else r
        out = r.assign(p=p).sort_values("p", ascending=False)
        return {"key": key, "label": label,
                "drivers": [{"code": x.Abbreviation, "team": x.TeamName, "p": round(float(x.p), 4),
                             "grid": int(x.GridPosition) if pd.notna(x.GridPosition) else None}
                            for x in out.itertuples()]}

    stages = [snapshot("pre", "Before the weekend", rows["p_form"].values)]
    blend = {1: 0.35, 2: 0.5, 3: 0.6}
    for k in range(1, len(pace) + 1):
        used = pace[:k]
        gaps = pd.concat([bl.set_index("code")["gap"].rename(n) for n, bl in used], axis=1)
        weights = np.arange(1, k + 1, dtype=float)
        score_ = (gaps * weights).sum(axis=1) / gaps.notna().mul(weights).sum(axis=1)
        order = list(score_.sort_values().index)
        r = with_grid(order)
        a = blend.get(k, 0.6)
        p = a * predict(full, r) + (1 - a) * r["p_form"].values
        short = used[-1][0].replace("Practice ", "FP").replace("Sprint Qualifying", "Sprint Quali")
        stages.append(snapshot(short.lower().replace(" ", ""), f"After {short}", p, r))
    if grid:
        g = list(grid)
        for c, places in penalties.items():
            if c in g:
                i = g.index(c)
                g.insert(min(i + places, len(g) - 1), g.pop(i))
        r = with_grid(g)
        stages.append(snapshot("quali", "After qualifying", predict(full, r), r))

    return {
        "year": year, "round": int(ev.RoundNumber), "name": ev.EventName, "location": ev.Location,
        "country": ev.Country, "raceUtc": ev.Session5DateUtc.isoformat() + "Z", "format": ev.EventFormat,
        "sessions": sessions, "stages": stages, "penalties": penalties,
        "features": {r.Abbreviation: {"teamTrend": round(float(r.TeamTrend), 2), "teamForm": round(float(r.TeamForm), 2),
                                      "form": round(float(r.RecentForm), 2), "track": round(float(r.TrackHistory), 1)}
                     for r in rows.itertuples()},
        "newVenue": bool(not (d_raw["Location"] == ev.Location).any()),
    }


# ---------------------------------------------------------------- facts
def facts(d, runs_champ, live=None):
    out = []
    y = int(d["Year"].max())
    cur = d[d["Year"] == y]
    last_idx = d["RaceIdx"].max()
    recent = d[d["RaceIdx"] > last_idx - 5]

    pods = recent[recent["Top3"] == 1]["Abbreviation"].value_counts()
    if len(pods):
        c, n = pods.index[0], int(pods.iloc[0])
        out.append({"tag": "Form", "title": f"{c} is the form driver",
                    "body": f"{n} podiums in the last 5 races, more than anyone else on the grid."})

    lr = d[d["RaceIdx"] == last_idx].drop_duplicates("Lineage")
    if live:  # trend going into the next race, same numbers the prediction uses
        team_of = {x["code"]: x["team"] for x in live["stages"][0]["drivers"]}
        tt = {}
        for c, f in live["features"].items():
            tt.setdefault(team_of.get(c), f["teamTrend"])
        lr = pd.DataFrame([{"TeamName": t, "TeamTrend": v} for t, v in tt.items() if t])
    lr = lr.dropna(subset=["TeamTrend"])
    if len(lr):
        up_ = lr.loc[lr["TeamTrend"].idxmax()]
        dn = lr.loc[lr["TeamTrend"].idxmin()]
        out.append({"tag": "Development", "title": f"{up_.TeamName} are finding pace",
                    "body": f"Their cars scored {up_.TeamTrend:+.1f} points per race more over the last 3 races than the 3 before, the clearest sign of an upgrade working."})
        if dn.TeamTrend < -0.5:
            out.append({"tag": "Development", "title": f"{dn.TeamName} are going backwards",
                        "body": f"{dn.TeamTrend:+.1f} points per car per race over the last 3 races compared with the 3 before."})

    wins = cur[cur["Position"] == 1]
    if len(wins):
        polewins = int((wins["GridPosition"] == 1).sum())
        out.append({"tag": str(y), "title": f"Pole won {polewins} of {len(wins)} races this season",
                    "body": f"That's {polewins / len(wins):.0%}. The front row matters, but the race is far from decided on Saturday."})
        podium = cur[cur["Top3"] == 1].copy()
        podium["gain"] = podium["GridPosition"] - podium["Position"]
        b = podium.loc[podium["gain"].idxmax()]
        out.append({"tag": str(y), "title": f"Charge of the season: {b.Abbreviation} from P{int(b.GridPosition)}",
                    "body": f"{b.Abbreviation} finished P{int(b.Position)} at the {b.Race.replace(' Grand Prix', '')} GP after starting P{int(b.GridPosition)}."})

    winners = cur[cur["Position"] == 1]["Abbreviation"].value_counts()
    if len(winners):
        out.append({"tag": str(y), "title": f"{len(winners)} different winners in {y}",
                    "body": ", ".join(f"{c} ×{n}" for c, n in winners.items()) + "."})

    tm = d[d["RaceIdx"] == last_idx].copy()
    if len(tm):
        top = tm.loc[tm["TeammateDelta"].idxmax()]
        out.append({"tag": "Teammates", "title": f"{top.Abbreviation} is out-scoring the car",
                    "body": f"Over the last 5 races {top.Abbreviation} has averaged {top.TeammateDelta:+.1f} points per race above {top.TeamName}'s team average."})

    ry = [r for r in runs_champ if r["idx"] in set(cur["RaceIdx"])]
    if ry:
        m, g = sum(r["hits"] for r in ry), sum(r["grid"] for r in ry)
        out.append({"tag": "Model", "title": f"This season the model called {m} of {len(ry) * 3} podium spots",
                    "body": f"The front three on the grid would have called {g}. " +
                            ("The model is ahead." if m > g else "The grid is ahead so far." if g > m else "They're level.")})
    return out
