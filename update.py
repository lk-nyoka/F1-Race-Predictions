"""One command to refresh Podium Lab. Safe to run any time; prints CHANGED or UNCHANGED at the end.

    python update.py            # normal refresh
    python update.py --research # force the feature search to re-run
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import engine
from backfill import backfill

ROOT = Path(__file__).parent
STATE, CHANGELOG, INTEL = ROOT / "state.json", ROOT / "changelog.json", ROOT / "intel.json"
PRED_DIR = ROOT / "predictions"
PRED_DIR.mkdir(exist_ok=True)


def jload(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def jsave(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main():
    state = jload(STATE, {})
    log = jload(CHANGELOG, [])
    intel = jload(INTEL, {"items": [], "grid_penalties": {}})
    stamp = engine.now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")

    def note(kind, text):
        log.append({"ts": stamp, "kind": kind, "text": text})
        print(f"[{kind}] {text}")

    # 1. results
    added = backfill()
    d_raw = engine.load_results()
    d = engine.prepare(d_raw)
    n_done = int(d["RaceIdx"].nunique())
    if added:
        last = d[d["RaceIdx"] == d["RaceIdx"].max()]
        pod = last.sort_values("Position").head(3)["Abbreviation"].tolist()
        note("result", f"{int(last.Year.iloc[0])} {last.Race.iloc[0]} result added: {', '.join(pod)} on the podium.")

    # 2. feature search whenever the data grows
    if "--research" in sys.argv or state.get("races") != n_done or "search" not in state:
        print("running feature search ...")
        s = engine.feature_search(d)
        prev = state.get("search")
        state["search"], state["races"] = s, n_done
        hist = state.setdefault("scoreHistory", [])
        hist.append({"ts": stamp, "races": n_done, "feats": s["feats"], "hl": s["half_life"],
                     "champion": s["champion"], "baseline": s["baseline"]})
        names = ", ".join(engine.LABELS[f] for f in s["feats"] if f not in engine.BASE) or "no new features"
        if not prev or prev["feats"] != s["feats"] or prev["half_life"] != s["half_life"]:
            note("model", f"Feature search on {n_done} races kept: {names}"
                          + (f", recent races weighted (half-life {s['half_life']})" if s["half_life"] else "") + ".")
        else:
            note("model", f"Retrained on {n_done} races. Feature set unchanged.")
    s = state["search"]
    champion = {"feats": s["feats"], "half_life": s["half_life"]}

    # 3. replay of every race with the champion model
    runs = engine.walk_forward(d, champion["feats"], champion["half_life"], keep_probs=True)
    replay = []
    for r in runs:
        t, m = r["test"], r["model"]
        rw = t.iloc[0]
        x_feats = [f for f in m["feats"] if f not in engine.GRID_FEATURES]
        replay.append({
            "year": int(rw.Year), "name": rw.Race.replace(" Grand Prix", ""), "location": rw.Location,
            "model": {"feats": m["feats"], "mu": [round(float(m["mu"][f]), 5) for f in m["feats"]],
                      "sd": [round(float(m["sd"][f]), 5) for f in m["feats"]],
                      "w": [round(float(v), 5) for v in m["m"].coef_[0]], "b": round(float(m["m"].intercept_[0]), 5)},
            "hits": {"model": r["hits"], "grid": r["grid"]},
            "drivers": [{"code": x.Abbreviation, "team": x.TeamName, "grid": int(x.GridPosition),
                         "finish": int(x.Position), "teamForm": round(float(x.TeamForm), 4),
                         "x": {f: round(float(getattr(x, f)), 4) for f in x_feats}}
                        for x in t.sort_values("GridPosition").itertuples()],
        })

    # 4. live weekend
    live = engine.live_weekend(d_raw, champion, intel)
    if live:
        key = f"{live['year']}-{live['round']:02d}"
        pfile = PRED_DIR / f"{key}.json"
        before = jload(pfile, {})
        seen = {st["key"] for st in before.get("stages", [])}
        for st in live["stages"]:
            if st["key"] not in seen:
                top = ", ".join(x["code"] for x in st["drivers"][:3])
                if st["key"] == "pre":
                    note("prediction", f"{live['year']} {live['name']} ({live['location']}): opening prediction {top}.")
                else:
                    sess = next((x for x in live["sessions"] if x.get("order") and st["label"].endswith(
                        x["name"].replace("Practice ", "FP"))), None)
                    lead = f" {sess['order'][0]['code']} fastest." if sess and st["key"] != "quali" else ""
                    if st["key"] == "quali" and live["sessions"]:
                        q = next((x for x in live["sessions"] if x["name"] == "Qualifying" and x.get("order")), None)
                        lead = f" {q['order'][0]['code']} on pole." if q else ""
                    note("session", f"{st['label']}:{lead} Predicted podium now {top}.")
        jsave(pfile, {"key": key, "year": live["year"], "name": live["name"], "location": live["location"],
                      "stages": live["stages"], "saved": stamp})

    # 5. prediction log: past weekends predicted live, scored once the result is in
    plog = []
    for pf in sorted(PRED_DIR.glob("*.json")):
        p = jload(pf, {})
        res = d[(d["Year"] == p.get("year")) & (d["Race"] == p.get("name"))]
        actual = res.sort_values("Position").head(3)["Abbreviation"].tolist() if len(res) else None
        entry = {"key": p["key"], "name": p["name"], "location": p["location"], "year": p["year"], "actual": actual,
                 "stages": [{"key": st["key"], "label": st["label"], "top3": [x["code"] for x in st["drivers"][:3]],
                             "hits": len(set(actual) & {x["code"] for x in st["drivers"][:3]}) if actual else None}
                            for st in p["stages"]]}
        plog.append(entry)
        if actual and not any(e.get("scored") == p["key"] for e in log):
            final = entry["stages"][-1]
            log.append({"ts": stamp, "kind": "scored", "scored": p["key"],
                        "text": f"{p['name']} scored: final prediction got {final['hits']}/3 ({', '.join(final['top3'])} vs {', '.join(actual)})."})

    # 6. intel changes
    if intel.get("updated") and intel.get("updated") != state.get("intelSeen"):
        state["intelSeen"] = intel["updated"]
        note("news", f"Paddock news refreshed: {len(intel.get('items', []))} items.")

    payload = {
        "live": live, "replay": replay, "search": s, "scoreHistory": state.get("scoreHistory", []),
        "predLog": plog, "facts": engine.facts(d, runs, live), "intel": intel,
        "labels": engine.LABELS, "changelog": log[-60:],
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    version = hashlib.sha1(body.encode()).hexdigest()[:10]
    payload["version"], payload["built"] = version, stamp

    changed = version != state.get("version")
    state["version"] = version
    jsave(STATE, state)
    jsave(CHANGELOG, log)

    tpl = (ROOT / "page.template.html").read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
                       .replace("</", "<\\/"))
    (ROOT / "podium-lab.html").write_text(html, encoding="utf-8")
    print("CHANGED" if changed else "UNCHANGED", version)


if __name__ == "__main__":
    main()
