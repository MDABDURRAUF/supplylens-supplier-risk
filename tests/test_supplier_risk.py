from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import supplier_risk as sr
from supplier_risk import evaluate as ev
from supplier_risk.cli import main
from supplier_risk.features import DIRECTION, NO_INCIDENT_MONTHS


@pytest.fixture(scope="module")
def data():
    return sr.generate_sample_data()


@pytest.fixture(scope="module")
def model(data):
    return sr.IncidentModel().fit(data)


@pytest.fixture(scope="module")
def watch(data, model):
    return sr.watchlist(data, model=model)


GOOD = {"otif_rate": 0.99, "defect_ppm": 400, "lead_time_cv": 0.08, "current_ratio": 2.5, "debt_to_equity": 0.3,
        "operating_margin": 0.12, "days_payable": 40}
MIDDLE = {"otif_rate": 0.915, "defect_ppm": 2750, "lead_time_cv": 0.25, "current_ratio": 1.4, "debt_to_equity": 1.75,
          "operating_margin": 0.04, "days_payable": 82.5}


def tiny(months=6, a=None, b=None, incidents=None):
    """Two hand-made suppliers: A in Ohio (single site, one of two items sole-sourced) and B in Germany.

    ``a`` and ``b`` map a month number (0, 1, ...) to the values that differ from the defaults for that month.
    """
    index = pd.date_range("2025-01-01", periods=months, freq="MS")
    suppliers = pd.DataFrame([
        {"supplier_id": "A", "supplier_name": "Alpha (made up)", "country": "US", "state": "OH", "single_site": 1, "annual_spend_usd": 300_000},
        {"supplier_id": "B", "supplier_name": "Beta (made up)", "country": "Germany", "state": "", "single_site": 0, "annual_spend_usd": 700_000}])
    rows = []
    for t, month in enumerate(index):
        rows.append({"supplier_id": "A", "month": month, **MIDDLE, "ownership_change": 0, "audit_findings": 0, "compliance_flag": 0, **(a or {}).get(t, {})})
        rows.append({"supplier_id": "B", "month": month, **GOOD, "ownership_change": int(t == 1), "audit_findings": 2.5, "compliance_flag": 1,
                     **(b or {}).get(t, {})})
    items = pd.DataFrame([{"item_id": "A1", "supplier_id": "A", "annual_spend_usd": 100_000, "sole_sourced": 1},
                          {"item_id": "A2", "supplier_id": "A", "annual_spend_usd": 200_000, "sole_sourced": 0},
                          {"item_id": "B1", "supplier_id": "B", "annual_spend_usd": 700_000, "sole_sourced": 0}])
    events = pd.DataFrame([{"supplier_id": s, "month": index[t], "incident_type": kind} for s, t, kind in (incidents or [])],
                          columns=["supplier_id", "month", "incident_type"])
    return sr.validate_data(suppliers, pd.DataFrame(rows), items, events)


# --------------------------------------------------------------------- data
def test_sample_data_is_reproducible_and_sensible(data):
    again = sr.generate_sample_data()
    for name in ("suppliers", "history", "items", "incidents"):
        pd.testing.assert_frame_equal(getattr(data, name), getattr(again, name))
    s, h, inc = data.suppliers, data.history, data.incidents
    assert len(s) == 80 and s["supplier_name"].is_unique and len(data.months) == 36
    assert set(s["region"]) == {"Northeast", "Midwest", "South", "West", "International"}
    assert 8 <= s["cross_border"].sum() <= 30 and (s.loc[s["cross_border"] == 1, "state"] == "").all()
    assert (s["region"].map(sr.REGION_HAZARD) == s["hazard_index"]).all()  # the illustrative index, by region
    assert h["otif_rate"].between(0, 1).all() and (h["defect_ppm"] >= 0).all() and not h.duplicated(["supplier_id", "month"]).any()
    np.testing.assert_allclose(data.items.groupby("supplier_id")["annual_spend_usd"].sum().reindex(s["supplier_id"]), s["annual_spend_usd"])
    assert 0 < s["sole_sourced_spend_usd"].sum() < s["annual_spend_usd"].sum()
    assert set(inc["incident_type"]) == set(sr.data.INCIDENT_TYPES) and 60 < len(inc) < 200
    # An insolvent supplier's history stops the month before the insolvency.
    gone = inc[inc["incident_type"] == "insolvency"]
    last = h.groupby("supplier_id")["month"].max()
    assert len(gone) >= 2 and (last[gone["supplier_id"]].to_numpy() == (gone["month"] - pd.DateOffset(months=1)).to_numpy()).all()
    assert (last.drop(gone["supplier_id"]) == data.last_month).all()
    # Deterioration starts before the incident: delivery in the three months before a failure is worse than the supplier's usual.
    usual = h.groupby("supplier_id")["otif_rate"].median()
    drops = []
    for sid, month in inc.loc[inc["incident_type"] == "delivery_failure", ["supplier_id", "month"]].itertuples(index=False):
        before = h[(h["supplier_id"] == sid) & (h["month"] < month) & (h["month"] >= month - pd.DateOffset(months=3))]
        if len(before) == 3:
            drops.append(usual[sid] - before["otif_rate"].mean())
    assert np.mean(drops) > 0.03 and np.mean(np.array(drops) > 0) > 0.7


def test_validation_reports_problems():
    good = tiny()
    assert list(good.suppliers["region"]) == ["Midwest", "International"] and list(good.suppliers["cross_border"]) == [0, 1]
    assert list(good.suppliers["hazard_index"]) == [0.35, 0.45] and list(good.suppliers["sole_sourced_spend_usd"]) == [100_000, 0]
    assert good.month("2025-03-17") == pd.Timestamp("2025-03-01") and good.month() == pd.Timestamp("2025-06-01")
    with pytest.raises(ValueError, match="No history for"):
        good.month("2030-01")
    s, h = good.suppliers[["supplier_id", "supplier_name", "country", "state", "annual_spend_usd"]], good.history
    with pytest.raises(sr.SchemaError, match="Missing required column.*history: otif_rate"):
        sr.validate_data(s, h.drop(columns="otif_rate"))
    with pytest.raises(sr.SchemaError, match="not a US state"):
        sr.validate_data(s.assign(state="ZZ"), h)
    with pytest.raises(sr.SchemaError, match="duplicate supplier and month"):
        sr.validate_data(s, pd.concat([h, h]))
    with pytest.raises(sr.SchemaError, match="not in suppliers"):
        sr.validate_data(s[s["supplier_id"] == "A"], h)
    with pytest.raises(sr.SchemaError, match="between 0 and 1"):
        sr.validate_data(s, h.assign(otif_rate=97.0))
    with pytest.raises(sr.SchemaError, match="gap in their monthly history"):
        sr.validate_data(s, h[h["month"] != "2025-03-01"])
    with pytest.raises(sr.SchemaError, match="unreadable 'month'"):
        sr.validate_data(s, h.assign(month="later"))
    bare = sr.validate_data(s, h.drop(columns=["ownership_change", "audit_findings", "compliance_flag"]))  # optional parts left out
    assert bare.items.empty and bare.incidents.empty and bare.suppliers["sole_sourced_spend_usd"].sum() == 0
    assert (bare.history["compliance_flag"] == 0).all() and list(bare.suppliers["single_site"]) == [0, 0]
    with pytest.raises(ValueError, match="incidents table is empty"):
        sr.IncidentModel().fit(bare)


def test_save_and_load_round_trip(tmp_path, data):
    loaded = sr.load_data(sr.save_data(data, tmp_path / "d"))
    for name in ("suppliers", "history", "items", "incidents"):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(data, name), check_dtype=False)
    with pytest.raises(FileNotFoundError):
        sr.load_data(tmp_path / "nowhere")


# -------------------------------------------------------------------- score
def test_scale_and_slope_by_hand():
    assert sr.scale(0.915, 0.98, 0.85) == pytest.approx(50)       # halfway from good to bad
    assert sr.scale(0.99, 0.98, 0.85) == 0 and sr.scale(0.5, 0.98, 0.85) == 100  # capped at both ends
    assert sr.scale(1250, 500, 5000) == pytest.approx(100 * 750 / 4500)
    assert sr.slope([10, 12, 14, 16]) == pytest.approx(2.0) and sr.slope([5, 5, 5]) == 0
    assert sr.slope([1, 2, 4]) == pytest.approx(1.5) and np.isnan(sr.slope([1, 2]))


def test_score_matches_a_hand_worked_example():
    scores = sr.score_as_of(tiny()).set_index("supplier_id")
    a, b = scores.loc["A"], scores.loc["B"]
    # A: every delivery, quality and financial measure sits halfway between its good and bad value.
    assert a["operational"] == pytest.approx(50) and a["financial"] == pytest.approx(50)
    assert a["geographic"] == pytest.approx(0.5 * 35 + 0.3 * 100 + 0.2 * 0)       # hazard 0.35, single site, in the US
    assert a["dependency"] == pytest.approx(0.4 * 100 + 0.6 * 100 / 3)            # 30% of spend (capped), a third sole-sourced
    assert a["compliance"] == 0
    assert a["overall"] == pytest.approx(0.30 * 50 + 0.25 * 50 + 0.10 * 47.5 + 0.15 * 60 + 0.20 * 0) == pytest.approx(41.25)
    # B: clean operations and finances, but abroad, large, with audit findings, a compliance flag and a recent change of owner.
    assert b["operational"] == 0 and b["financial"] == 0
    assert b["geographic"] == pytest.approx(0.5 * 45 + 0.2 * 100) and b["dependency"] == pytest.approx(0.4 * 100)
    assert b["compliance"] == pytest.approx(0.4 * 50 + 0.4 * 100 + 0.2 * 100)
    assert b["overall"] == pytest.approx(0.10 * 42.5 + 0.15 * 40 + 0.20 * 80) == pytest.approx(26.25)
    assert a["trend"] == pytest.approx(0) and not a["deteriorating"] and list(scores.index) == ["A", "B"]  # riskiest first


def test_weights_and_thresholds_are_configurable():
    d = tiny()
    only_ops = sr.score_as_of(d, config=sr.ScoreConfig(weights={"operational": 1})).set_index("supplier_id")
    assert only_ops.loc["A", "overall"] == pytest.approx(50) and only_ops.loc["B", "overall"] == 0
    doubled = sr.score_as_of(d, config=sr.ScoreConfig(weights={k: 2 * v for k, v in sr.DEFAULT_WEIGHTS.items()})).set_index("supplier_id")
    assert doubled.loc["A", "overall"] == pytest.approx(41.25)  # weights are rescaled to sum to 1
    strict = sr.score_as_of(d, config=sr.ScoreConfig(thresholds={"otif": (0.99, 0.915)})).set_index("supplier_id")
    assert strict.loc["A", "operational"] == pytest.approx(0.45 * 100 + 0.30 * 50 + 0.25 * 50)
    for bad in ({"operational": -1}, {"nonsense": 1}, {d: 0 for d in sr.DIMENSIONS}):
        with pytest.raises(ValueError):
            sr.score_as_of(d, config=sr.ScoreConfig(weights=bad))
    assert sum(m.weight for m in sr.MEASURES if m.dimension == "financial") == pytest.approx(1)


def test_every_sub_score_is_explained_in_one_sentence():
    d = tiny()
    why = sr.explain_score(d, "A")
    assert len(why) == len(sr.MEASURES) and why["points"].sum() == pytest.approx(41.25)
    by = why.set_index("measure")
    assert by.loc["otif", "sentence"] == ("On-time-in-full averaged 91.5% over the last 3 months; 98.0% or more scores 0 and "
                                          "85.0% or less scores 100, so this scores 50.")
    assert by.loc["single_site", "sentence"] == "The supplier makes everything at a single site, which scores 100."
    assert by.loc["otif", "points"] == pytest.approx(50 * 0.45 * 0.30)
    assert "No compliance flag" in by.loc["compliance_flag", "sentence"]
    assert "A compliance flag was open" in sr.explain_score(d, "B").set_index("measure").loc["compliance_flag", "sentence"]
    with pytest.raises(ValueError, match="no history"):
        sr.explain_score(d, "nobody")


def test_trend_flags_a_supplier_that_is_still_acceptable_but_falling_fast():
    falling = {t: {**GOOD, "otif_rate": 0.98 - 0.03 * t} for t in range(6)}  # on-time delivery loses three points a month
    d = tiny(a=falling)
    panel = sr.score_panel(d)
    a = panel[panel["supplier_id"] == "A"].reset_index(drop=True)
    otif_3m = [0.98, 0.965, 0.95, 0.92, 0.89, 0.86]
    expected = [0.30 * 0.45 * sr.scale(v, 0.98, 0.85) for v in otif_3m]  # only the on-time sub-score moves
    np.testing.assert_allclose(a["overall"] - a["overall"].iloc[0], expected)
    assert a["trend"].iloc[-1] == pytest.approx(sr.slope(expected)) and a["trend"].iloc[-1] > 2.0
    assert np.isnan(a["trend"].iloc[1]) and a["deteriorating"].iloc[-1] and not a["deteriorating"].iloc[2]
    watch = sr.watchlist(d).set_index("supplier_id")
    assert watch.loc["A", "overall"] < sr.TierRules().elevated_score and watch.loc["A", "tier"] == "Elevated"
    assert "the score has been rising" in watch.loc["A", "reason"].lower() and watch.loc["B", "tier"] == "Normal"
    assert sr.watchlist(d, config=sr.ScoreConfig(deterioration_slope=5.0)).set_index("supplier_id").loc["A", "tier"] == "Normal"


# ----------------------------------------------------------------- features
def test_features_by_hand():
    otif = [0.96, 0.94, 0.98, 0.90, 0.92, 0.94, 0.80]
    d = tiny(months=7, a={t: {"otif_rate": v} for t, v in enumerate(otif)}, incidents=[("A", 2, "quality_escape"), ("A", 5, "delivery_failure")])
    x = sr.build_features(d)
    a = x[x["supplier_id"] == "A"].reset_index(drop=True)
    assert a["otif_3m"].iloc[5] == pytest.approx((0.90 + 0.92 + 0.94) / 3)
    assert a["otif_vs_usual"].iloc[5] == pytest.approx((0.90 + 0.92 + 0.94) / 3 - (0.96 + 0.94 + 0.98) / 3)  # against its own earlier months
    assert a["otif_vs_usual"].iloc[6] == pytest.approx((0.92 + 0.94 + 0.80) / 3 - (0.96 + 0.94 + 0.98 + 0.90) / 4)
    assert a["otif_vs_usual"].iloc[:5].isna().all()  # fewer than three earlier months
    assert a["incidents_12m"].tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert a["months_since_incident"].tolist() == [NO_INCIDENT_MONTHS, NO_INCIDENT_MONTHS, 0, 1, 2, 0, 1]
    assert a["history_months"].tolist() == [1, 2, 3, 4, 5, 6, 7] and (a["single_site"] == 1).all() and (a["hazard_index"] == 0.35).all()
    b = x[x["supplier_id"] == "B"].reset_index(drop=True)
    assert b["ownership_change_12m"].tolist() == [0, 1, 1, 1, 1, 1, 1] and (b["cross_border"] == 1).all()
    assert list(x.columns) == ["supplier_id", "month"] + sr.FEATURES + ["history_months"] and set(DIRECTION) == set(sr.FEATURES)
    assert {f for group in sr.FACTOR_GROUPS.values() for f in group} == set(sr.FEATURES)


def test_outcome_is_known_only_after_the_whole_quarter():
    d = tiny(months=8, incidents=[("A", 4, "delivery_failure")])
    rows = sr.build_features(d)
    label = sr.incident_labels(d, rows, horizon=3)
    a = label[rows["supplier_id"] == "A"].tolist()
    assert a[:5] == [0, 1, 1, 1, 0]            # the three months before the incident; the incident month itself looks ahead
    assert np.isnan(a[5:]).all()               # months 5 to 7 need data to months 8 to 10, which do not exist yet
    assert label[rows["supplier_id"] == "B"].tolist()[:5] == [0, 0, 0, 0, 0]


def test_scores_and_features_do_not_change_when_later_records_change(data, model):
    cut = data.months[20]
    h = data.history.copy()
    later = h["month"] > cut
    h.loc[later, "otif_rate"] = h.loc[later, "otif_rate"] * 0.5
    h.loc[later, "defect_ppm"] = h.loc[later, "defect_ppm"] * 3
    h.loc[later, "current_ratio"] = 0.4
    h.loc[later, ["compliance_flag", "ownership_change"]] = 1
    altered = replace(data, history=h, incidents=data.incidents[data.incidents["month"] <= cut])  # and every later incident erased
    for build in (sr.build_features, sr.score_panel, model.predict_panel):
        before, after = build(data), build(altered)
        early = before["month"] <= cut
        pd.testing.assert_frame_equal(before[early], after[early])
        assert not before[~early].equals(after[~early])
    pd.testing.assert_frame_equal(sr.watchlist(data, cut, model), sr.watchlist(altered, cut, model))


# -------------------------------------------------------------------- model
def test_split_is_by_time_with_no_outcome_used_early(data, model):
    s = model.split_
    assert s["train_until"] + pd.DateOffset(months=3) < s["calibration_from"]      # the last training outcome was known before calibration began
    assert s["calibration_until"] + pd.DateOffset(months=3) < s["test_from"]
    assert s["calibration_from"] <= s["calibration_until"] and s["last_month"] + pd.DateOffset(months=3) == data.last_month
    h = model.holdout_
    assert h["month"].min() == s["test_from"] and len(h) == s["test_rows"] > 300
    assert s["train_rows"] > 800 and s["calibration_rows"] > 250
    rows = sr.build_features(data)
    rows["incident"] = sr.incident_labels(data, rows)
    train = rows[(rows["history_months"] >= 3) & (rows["month"] <= s["train_until"])]
    assert len(train) == s["train_rows"] and model.prior_ == pytest.approx(train["incident"].mean())
    merged = h.merge(rows, on=["supplier_id", "month"], suffixes=("", "_check"))
    assert (merged["incident"] == merged["incident_check"]).all()


def test_model_beats_the_score_and_the_no_model_baseline(model):
    table = ev.compare_models(model).set_index("model")
    boosted, score = table.loc["Gradient boosting, calibrated"], table.loc["Transparent risk score"]
    constant = table.loc["Overall incident rate (no model)"]
    assert boosted["roc_auc"] > score["roc_auc"] + 0.02 and score["roc_auc"] > 0.7 and np.isnan(constant["roc_auc"])
    assert boosted["brier_score"] < score["brier_score"] < constant["brier_score"]
    assert boosted["average_precision"] > score["average_precision"] and boosted["log_loss"] < constant["log_loss"]
    s = ev.summary(model)
    assert s["roc_auc"] > 0.8 and 0.08 < s["skill"] < 1 and s["rows"] == len(model.holdout_) and 0.05 < s["incident_rate"] < 0.2


def test_predictions_are_well_formed_and_survive_saving(tmp_path, data, model):
    pred = model.predict(data)
    assert pred["incident_probability"].between(0, 1).all() and (pred["month"] == data.last_month).all()
    assert set(pred["supplier_id"]) == set(data.history.loc[data.history["month"] == data.last_month, "supplier_id"])
    earlier = model.predict(data, "2025-03")
    assert (earlier["month"] == pd.Timestamp("2025-03-01")).all() and len(earlier) >= len(pred)
    loaded = sr.IncidentModel.load(model.save(tmp_path / "m.joblib"))
    pd.testing.assert_frame_equal(loaded.predict(data), pred)
    # Each input can only move the prediction one way: worse delivery never lowers the risk.
    x = sr.build_features(data)
    x = x[x["month"] == data.last_month]
    worse = x.assign(otif_3m=x["otif_3m"] - 0.10, otif_vs_usual=x["otif_vs_usual"] - 0.10)
    assert (model.proba_from_features(worse) >= model.proba_from_features(x) - 1e-12).all()
    assert model.proba_from_features(worse).mean() > model.proba_from_features(x).mean() + 0.02
    with pytest.raises(RuntimeError, match="fit"):
        sr.IncidentModel().predict(data)
    with pytest.raises(ValueError, match="At least 12 months"):
        sr.IncidentModel().fit(tiny(months=10, incidents=[("A", 3, "quality_escape")]))
    with pytest.raises(ValueError, match="sum to at most 1"):
        sr.IncidentModel(train_fraction=0.9, calib_fraction=0.3).fit(data)


# --------------------------------------------------------------- evaluation
def test_calibration_lift_and_uncertainty(model):
    cal = ev.calibration_table(model)
    assert cal["rows"].sum() == len(model.holdout_) and cal["predicted"].is_monotonic_increasing
    assert cal["observed"].iloc[-1] > 3 * cal["observed"].iloc[:3].mean() and cal["gap"].abs().max() < 0.2
    lift, by_score = ev.lift_table(model), ev.lift_table(model, column="score")
    assert lift["cumulative_capture"].iloc[-1] == pytest.approx(1.0) and lift["cumulative_capture"].is_monotonic_increasing
    assert lift["lift"].iloc[0] > 2.5 and lift["cumulative_capture"].iloc[0] > 0.6  # the riskiest fifth holds most of the incidents
    assert lift["cumulative_capture"].iloc[0] > by_score["cumulative_capture"].iloc[0]
    lo, hi = ev.auc_interval(model, repeats=200)
    assert 0.5 < lo < ev.summary(model)["roc_auc"] < hi <= 1 and hi - lo > 0.05  # a small test period gives a wide range
    seg = ev.by_segment(model, "region")
    assert seg["rows"].sum() == len(model.holdout_) and set(seg["region"]) <= set(sr.REGION_HAZARD)


def test_trend_flag_and_seed_check(data):
    flag = ev.trend_flag_check(data)
    # Among suppliers whose score still looks acceptable, those with a fast-rising score have incidents far more often.
    assert flag["flagged_rows"] > 50 and flag["flagged_incident_rate"] > 2 * flag["other_incident_rate"]
    seeds = ev.seed_check(seeds=[1, 2])
    assert list(seeds["seed"]) == [1, 2] and seeds["auc_model"].between(0.5, 1).all() and (seeds["brier_no_model"] > 0).all()


# -------------------------------------------------------------- explanation
def test_factor_importance_and_local_reasons(data, model, watch):
    imp = sr.factor_importance(model)
    assert imp.iloc[0]["factor"] == "delivery performance" and imp["share"].sum() == pytest.approx(1.0)
    assert set(imp["factor"]) == set(sr.FACTOR_GROUPS) and imp["auc_drop"].is_monotonic_decreasing
    effects = sr.explain(model, data)
    assert effects.shape == (len(watch), len(sr.FACTOR_GROUPS)) and effects.abs().max().max() <= 1
    risky, safe = watch.iloc[0]["supplier_id"], watch.sort_values("incident_probability").iloc[0]["supplier_id"]
    assert effects.loc[risky].max() > 0.10 and effects.loc[safe].max() < 0.02
    assert watch.iloc[0]["main_drivers"].startswith(effects.loc[risky].idxmax())
    flat = pd.DataFrame({"quality": [0.0], "location": [0.01]})
    assert sr.top_reasons(flat).iloc[0] == "no single factor stands out"
    reasons = sr.top_reasons(pd.DataFrame({"quality": [0.30], "location": [0.12], "financial health": [0.05]}), n=2)
    assert reasons.iloc[0] == "quality (+30 pts), location (+12 pts)"


# --------------------------------------------------------------- monitoring
def test_watchlist_tiers_follow_the_rules_and_give_a_reason(data, model, watch):
    rules = sr.TierRules()
    assert len(watch) == (data.history["month"] == data.last_month).sum() and set(watch["tier"]) == set(sr.TIERS)
    critical = (watch["overall"] >= rules.critical_score) | (watch["incident_probability"] >= rules.critical_probability)
    elevated = ~critical & ((watch["overall"] >= rules.elevated_score) | (watch["incident_probability"] >= rules.elevated_probability)
                            | watch["deteriorating"])
    assert (watch.loc[critical, "tier"] == "Critical").all() and (watch.loc[elevated, "tier"] == "Elevated").all()
    assert (watch.loc[~critical & ~elevated, "tier"] == "Normal").all()
    assert watch["tier"].map({t: i for i, t in enumerate(sr.TIERS)}).is_monotonic_increasing  # most serious first
    assert (watch["reason"].str.len() > 20).all() and (watch.loc[watch["tier"] != "Normal", "main_drivers"] != "").all()
    assert watch.loc[watch["tier"] == "Normal", "reason"].str.startswith("Score, probability and trend are all below").all()
    top = watch.iloc[0]
    assert f"incident probability {top['incident_probability']:.0%}" in top["reason"].lower()
    np.testing.assert_allclose(watch["spend_at_risk"], watch["annual_spend_usd"] * watch["incident_probability"])
    # Without a model the watchlist still works, from the score and its trend alone.
    plain = sr.watchlist(data)
    assert plain["incident_probability"].isna().all() and (plain["tier"] == "Critical").sum() <= (watch["tier"] == "Critical").sum()
    lenient = sr.watchlist(data, model=model, rules=sr.TierRules(critical_score=99, elevated_score=98, critical_probability=0.99,
                                                                 elevated_probability=0.98), config=sr.ScoreConfig(deterioration_slope=99))
    assert (lenient["tier"] == "Normal").all()
    headline = sr.watchlist_headline(watch)
    assert "June 2026" in headline and f"{(watch['tier'] == 'Critical').sum()} are Critical" in headline


def test_what_changed_since_last_month():
    # A's delivery collapses in the last month with a failure recorded; B gets a new owner.
    d = tiny(months=8, a={7: {"otif_rate": 0.40, "compliance_flag": 1}}, b={7: {"ownership_change": 1}}, incidents=[("A", 7, "delivery_failure")])
    changes = sr.what_changed(d).set_index("supplier_id")
    a, b = changes.loc["A"], changes.loc["B"]
    assert a["previous_tier"] == "Normal" and a["tier"] == "Elevated" and a["move"] == "moved up" and a["score_change"] > 5
    assert a["new_flags"] == "incident recorded: delivery failure; deterioration flag raised; compliance flag opened"
    assert b["move"] == "unchanged" and b["new_flags"] == "ownership changed" and b["score_change"] == pytest.approx(0, abs=1e-9)
    assert changes.index[0] == "A" and "1 supplier moved to a more serious tier" in sr.changes_headline(changes.reset_index())
    with pytest.raises(ValueError, match="first month"):
        sr.what_changed(d, "2025-01")


def test_what_changed_on_the_sample_base(data, model):
    gone = data.incidents[data.incidents["incident_type"] == "insolvency"].iloc[0]
    changes = sr.what_changed(data, gone["month"], model).set_index("supplier_id")
    left = changes.loc[gone["supplier_id"]]
    assert left["move"] == "left" and left["tier"] == "Left the base" and "incident recorded: insolvency" in left["new_flags"]
    now, before = sr.watchlist(data, gone["month"], model), sr.watchlist(data, gone["month"] - pd.DateOffset(months=1), model)
    assert len(changes) == len(set(now["supplier_id"]) | set(before["supplier_id"]))
    stay = changes[changes["move"].isin(["moved up", "moved down", "unchanged"])]
    np.testing.assert_allclose(stay["score_change"], now.set_index("supplier_id").loc[stay.index, "overall"]
                               - before.set_index("supplier_id").loc[stay.index, "overall"])


# ---------------------------------------------------------------- portfolio
def test_spend_at_risk_and_concentration_by_hand():
    assert sr.hhi([50, 50]) == pytest.approx(5000) and sr.hhi([100]) == pytest.approx(10000) and sr.hhi([25] * 4) == pytest.approx(2500)
    assert sr.hhi([300_000, 700_000]) == pytest.approx(30**2 + 70**2)
    watch = sr.watchlist(tiny(a={t: {**GOOD, "otif_rate": 0.98 - 0.03 * t} for t in range(6)}))  # A is Elevated, B Normal
    s = sr.spend_at_risk(watch)
    assert s["total_spend"] == 1_000_000 and s["watch_spend"] == s["elevated_spend"] == 300_000 and s["critical_spend"] == 0
    assert s["watch_share"] == pytest.approx(0.3) and s["sole_sourced_spend"] == s["sole_sourced_watch_spend"] == 100_000
    assert np.isnan(s["probability_weighted_spend"])  # no model, so no probabilities
    by = sr.concentration(watch, "supplier")
    assert list(by["supplier"]) == ["Beta (made up)", "Alpha (made up)"] and list(by["share"]) == pytest.approx([0.7, 0.3])
    assert list(by["cumulative_share"]) == pytest.approx([0.7, 1.0]) and list(by["on_watch"]) == [0, 1]
    regions = sr.concentration(watch, "region").set_index("region")
    assert regions.loc["International", "spend"] == 700_000 and sr.hhi(regions["spend"]) == pytest.approx(5800)
    with pytest.raises(ValueError, match="Cannot group by"):
        sr.concentration(watch, "planet")
    with pytest.raises(ValueError, match="no incident probabilities"):
        sr.probability_impact(watch)


def test_portfolio_on_the_sample_base(watch):
    s = sr.spend_at_risk(watch)
    assert s["probability_weighted_spend"] == pytest.approx((watch["annual_spend_usd"] * watch["incident_probability"]).sum())
    assert s["critical_spend"] + s["elevated_spend"] == pytest.approx(s["watch_spend"]) and 0 < s["watch_share"] < 1
    by = sr.concentration(watch, "supplier")
    assert by["share"].sum() == pytest.approx(1) and by["spend"].is_monotonic_decreasing and len(by) == len(watch)
    assert 10000 / len(watch) < sr.hhi(by["spend"]) < 2000  # more concentrated than an even split, far from a single supplier
    pi = sr.probability_impact(watch)
    assert set(pi["quadrant"]) == {"act now", "watch closely", "protect", "routine"}
    act = pi[pi["quadrant"] == "act now"]
    assert (act["incident_probability"] >= 0.20).all() and (act["annual_spend_usd"] >= watch["annual_spend_usd"].quantile(0.75)).all()
    assert pi["spend_at_risk"].is_monotonic_decreasing


def test_dual_sourcing_what_if_by_hand(data, model, watch):
    d = tiny()
    table = sr.dual_source_what_if(d, "A", share_moved=0.5).set_index("figure")
    # Half of the 100,000 sole-sourced spend moves to a second source; the item is no longer sole-sourced.
    assert table.loc["spend with supplier", ["before", "after"]].tolist() == [300_000, 250_000]
    assert table.loc["sole-sourced spend", ["before", "after"]].tolist() == [100_000, 0]
    assert table.loc["share of total spend", "after"] == pytest.approx(0.25)
    assert table.loc["dependency score", "before"] == pytest.approx(60) and table.loc["dependency score", "after"] == pytest.approx(40)
    assert table.loc["overall risk score", "before"] == pytest.approx(41.25)
    assert table.loc["overall risk score", "after"] == pytest.approx(41.25 - 0.15 * 20)
    assert table.loc["portfolio HHI", "before"] == pytest.approx(5800) and table.loc["portfolio HHI", "after"] == pytest.approx(25**2 + 5**2 + 70**2)
    assert np.isnan(table.loc["probability-weighted spend", "after"])
    assert sr.dependency_score(300_000, 100_000, 1_000_000) == pytest.approx(60)
    assert (sr.dual_source_what_if(d, "B")["change"].fillna(0) == 0).all()  # nothing sole-sourced, nothing to move
    with pytest.raises(ValueError, match="between 0 and 1"):
        sr.dual_source_what_if(d, "A", share_moved=1.5)
    with pytest.raises(ValueError, match="not active"):
        sr.dual_source_what_if(d, "Z")
    # On the sample base, with a model: the probability stays the same, so exposure falls in step with spend.
    row = watch[watch["sole_sourced_spend_usd"] > 0].iloc[0]
    full = sr.dual_source_what_if(data, row["supplier_id"], 0.5, model=model).set_index("figure")
    assert full.loc["dependency score", "before"] == pytest.approx(row["dependency"])
    assert full.loc["probability-weighted spend", "before"] == pytest.approx(row["spend_at_risk"])
    assert full.loc["probability-weighted spend", "after"] == pytest.approx(row["incident_probability"] * full.loc["spend with supplier", "after"])
    assert full.loc["overall risk score", "change"] < 0 and "$" in sr.what_if_text(full.reset_index()).loc[0, "before"]


# ------------------------------------------------------------ reports and CLI
def test_reports(data, model, watch):
    sid = watch.iloc[0]["supplier_id"]
    profile = sr.build_supplier_profile(data, sid, model=model, note="Simulated data.")
    for needle in (watch.iloc[0]["supplier_name"], "Score by dimension", "Why the score is what it is", "On-time-in-full averaged",
                   "What drives the model", "<svg", "not a credit rating", "Simulated data.", "illustrative"):
        assert needle in profile
    earlier = sr.build_supplier_profile(data, sid, "2025-01")  # replayed, and without a model
    assert "As of January 2025" in earlier and "What drives the model" not in earlier and "Feb 2025" not in earlier
    report = sr.build_portfolio_report(data, model=model, note="Simulated data.")
    for needle in ("Portfolio risk report: June 2026", "Suppliers on watch", "What changed since last month", "Concentration",
                   "Likely and large", f"{ev.summary(model)['roc_auc']:.3f}", "not a credit rating", "Simulated data.", sid):
        assert needle in report
    assert "How far to trust the model" not in sr.build_portfolio_report(data, "2023-07")
    with pytest.raises(ValueError, match="not active"):
        sr.build_supplier_profile(data, "S-999")


def test_cli_end_to_end(tmp_path, capsys):
    folder, mdl = tmp_path / "data", tmp_path / "m.joblib"
    assert main(["sample", "--out", str(folder)]) == 0
    assert "Not real data" in capsys.readouterr().out and (folder / "history.csv").exists()
    assert main(["score", "--data", str(folder), "--as-of", "2026-03", "--out", str(tmp_path / "scores.csv")]) == 0
    assert "Risk scores for March 2026" in capsys.readouterr().out and len(pd.read_csv(tmp_path / "scores.csv")) > 60
    assert main(["train", "--data", str(folder), "--model", str(mdl)]) == 0
    out = capsys.readouterr().out
    assert "Gradient boosting, calibrated" in out and "Transparent risk score" in out and "rough 95% range" in out
    assert main(["watchlist", "--data", str(folder), "--model", str(mdl), "--out", str(tmp_path / "watch.csv")]) == 0
    watch = pd.read_csv(tmp_path / "watch.csv")
    assert "are Critical" in capsys.readouterr().out and {"tier", "reason", "incident_probability", "spend_at_risk"} <= set(watch.columns)
    sid = watch.loc[watch["sole_sourced_spend_usd"] > 0, "supplier_id"].iloc[0]
    assert main(["profile", "--data", str(folder), "--model", str(mdl), "--supplier", sid, "--out", str(tmp_path / "p.html"), "--note", "Simulated."]) == 0
    assert main(["report", "--data", str(folder), "--model", str(mdl), "--out", str(tmp_path / "r.html")]) == 0
    assert (tmp_path / "p.html").stat().st_size > 8000 and (tmp_path / "r.html").stat().st_size > 8000
    capsys.readouterr()
    assert main(["whatif", "--data", str(folder), "--model", str(mdl), "--supplier", sid, "--share", "0.5"]) == 0
    assert "sole-sourced spend" in capsys.readouterr().out
    assert main(["watchlist", "--data", str(folder), "--weights", "operational=1,financial=0,geographic=0,dependency=0,compliance=0"]) == 0  # no model
    capsys.readouterr()
    assert main(["score", "--data", str(folder), "--weights", "luck=1"]) == 1 and "not a dimension" in capsys.readouterr().err
    assert main(["profile", "--data", str(folder), "--supplier", "S-999"]) == 1 and "not active" in capsys.readouterr().err
    assert main(["score", "--data", str(folder), "--as-of", "2031-01"]) == 1 and "No history for" in capsys.readouterr().err
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "suppliers.csv").write_text("a,b\n1,2\n")
    (tmp_path / "bad" / "history.csv").write_text("a,b\n1,2\n")
    assert main(["score", "--data", str(tmp_path / "bad")]) == 1 and "Missing required column" in capsys.readouterr().err
