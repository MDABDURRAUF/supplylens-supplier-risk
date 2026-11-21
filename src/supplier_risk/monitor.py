"""Monitoring: the watchlist for a month, and what changed since the month before."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SupplierData
from .explain import explain_features, top_reasons
from .features import build_features
from .model import IncidentModel
from .score import DIMENSIONS, ScoreConfig, score_panel

TIERS = ["Critical", "Elevated", "Normal"]


@dataclass(frozen=True)
class TierRules:
    """When a supplier goes on the watchlist. The defaults are illustrative; set your own.

    Critical: the risk score is at least ``critical_score`` or the incident
    probability is at least ``critical_probability``. Elevated: the score is
    at least ``elevated_score``, the probability is at least
    ``elevated_probability``, or the deterioration flag is up (so a supplier
    whose score is still acceptable but rising fast is caught). Otherwise
    Normal.
    """

    critical_score: float = 60.0
    elevated_score: float = 48.0
    critical_probability: float = 0.35
    elevated_probability: float = 0.20


def _reason(row: pd.Series, rules: TierRules) -> str:
    """One sentence saying which rule put the supplier in its tier."""
    parts = []
    p, score = row["incident_probability"], row["overall"]
    if not pd.isna(p):
        if p >= rules.critical_probability:
            parts.append(f"incident probability {p:.0%} is at or above {rules.critical_probability:.0%}")
        elif p >= rules.elevated_probability:
            parts.append(f"incident probability {p:.0%} is at or above {rules.elevated_probability:.0%}")
    if score >= rules.critical_score:
        parts.append(f"risk score {score:.0f} is at or above {rules.critical_score:.0f}")
    elif score >= rules.elevated_score:
        parts.append(f"risk score {score:.0f} is at or above {rules.elevated_score:.0f}")
    if row["deteriorating"]:
        parts.append(f"the score has been rising {row['trend']:.1f} points a month")
    worst = max(DIMENSIONS, key=lambda d: row[d])
    if not parts:
        return f"Score, probability and trend are all below the watch thresholds; highest dimension is {worst} ({row[worst]:.0f})."
    text = "; ".join(parts)
    return f"{text[0].upper()}{text[1:]}; highest dimension is {worst} ({row[worst]:.0f})."


def watchlist(data: SupplierData, as_of=None, model: IncidentModel | None = None, config: ScoreConfig | None = None,
              rules: TierRules | None = None, explain: bool = True) -> pd.DataFrame:
    """Every supplier active in the month ``as_of`` (default: the latest), most worrying first.

    ``overall`` and the five dimension columns are risk scores from 0 to 100.
    ``trend`` is the change in the overall score in points per month over
    recent months, and ``deteriorating`` flags a fast rise.
    ``incident_probability`` is the model's calibrated chance of an incident
    in the next quarter (missing when no model is given). ``tier`` follows
    ``TierRules`` and ``reason`` says which rule applied.
    ``spend_at_risk`` is annual spend times incident probability: a way to
    rank exposure, not a forecast of loss. ``main_drivers`` names the factors
    raising the model's prediction for suppliers on watch.
    """
    config, rules = config or ScoreConfig(), rules or TierRules()
    month = data.month(as_of)
    panel = score_panel(data, config)
    now = panel[panel["month"] == month]
    info = data.suppliers[["supplier_id", "supplier_name", "category", "country", "state", "region", "single_site",
                           "annual_spend_usd", "sole_sourced_spend_usd", "sole_sourced_items"]]
    out = info.merge(now[["supplier_id", "month", "overall"] + DIMENSIONS + ["trend", "change_1m", "deteriorating"]], on="supplier_id")
    if model is not None:
        rows = build_features(data)
        rows = rows[rows["month"] == month].set_index("supplier_id")
        X = rows.loc[out["supplier_id"], model.features_]
        out["incident_probability"] = model.proba_from_features(X)
    else:
        out["incident_probability"] = np.nan
    p = out["incident_probability"].fillna(0.0)
    out["tier"] = np.select(
        [(out["overall"] >= rules.critical_score) | (p >= rules.critical_probability),
         (out["overall"] >= rules.elevated_score) | (p >= rules.elevated_probability) | out["deteriorating"]],
        ["Critical", "Elevated"], "Normal")
    out["reason"] = [_reason(row, rules) for _, row in out.iterrows()]
    out["spend_at_risk"] = out["annual_spend_usd"] * out["incident_probability"]
    out["main_drivers"] = ""
    watched = (out["tier"] != "Normal").to_numpy()
    if model is not None and explain and watched.any():
        effects = explain_features(model, X[watched])
        out.loc[watched, "main_drivers"] = top_reasons(effects).to_numpy()
    order = out["tier"].map({t: i for i, t in enumerate(TIERS)})
    out = out.assign(_tier=order, _p=p).sort_values(["_tier", "_p", "overall"], ascending=[True, False, False])
    columns = ["supplier_id", "supplier_name", "tier", "reason", "overall"] + DIMENSIONS + [
        "trend", "deteriorating", "incident_probability", "main_drivers", "annual_spend_usd", "spend_at_risk",
        "sole_sourced_spend_usd", "sole_sourced_items", "category", "country", "state", "region", "single_site", "change_1m", "month"]
    return out[columns].reset_index(drop=True)


def watchlist_headline(watch: pd.DataFrame) -> str:
    """One paragraph on the watchlist."""
    if watch.empty:
        return "No suppliers are active in this month."
    month = pd.Timestamp(watch["month"].iloc[0])
    critical, elevated = watch[watch["tier"] == "Critical"], watch[watch["tier"] == "Elevated"]
    total = watch["annual_spend_usd"].sum()
    on_watch = critical["annual_spend_usd"].sum() + elevated["annual_spend_usd"].sum()
    text = (f"{month:%B %Y}: {len(watch)} active suppliers with ${total:,.0f} of annual spend. {len(critical)} "
            f"{'is' if len(critical) == 1 else 'are'} Critical and {len(elevated)} Elevated, together "
            f"${on_watch:,.0f} of spend ({on_watch / total if total else 0:.0%}).")
    flagged = int(watch["deteriorating"].sum())
    if flagged:
        text += f" {flagged} {'has' if flagged == 1 else 'have'} a fast-rising score."
    if len(critical):
        top = critical.sort_values("annual_spend_usd", ascending=False).iloc[0]
        text += f" The largest Critical supplier is {top['supplier_name']} (${top['annual_spend_usd']:,.0f} a year)."
    return text


def what_changed(data: SupplierData, as_of=None, model: IncidentModel | None = None, config: ScoreConfig | None = None,
                 rules: TierRules | None = None) -> pd.DataFrame:
    """What is different in the month ``as_of`` compared with the month before, one row per supplier.

    ``move`` is ``moved up`` (to a more serious tier), ``moved down``,
    ``unchanged``, ``new`` (no record last month) or ``left`` (active last
    month, gone now). ``new_flags`` lists events first seen this month: a
    deterioration flag, a compliance flag, an ownership change, or a
    recorded incident. Sorted with the moves up first.
    """
    month = data.month(as_of)
    months = list(data.months)
    if months.index(month) == 0:
        raise ValueError("The first month of history has no earlier month to compare with")
    previous = months[months.index(month) - 1]
    now = watchlist(data, month, model, config, rules, explain=False)
    before = watchlist(data, previous, model, config, rules, explain=False)
    keep = ["supplier_id", "tier", "overall", "incident_probability", "deteriorating"]
    merged = now[["supplier_id", "supplier_name", "tier", "overall", "incident_probability", "deteriorating", "annual_spend_usd"]].merge(
        before[keep].rename(columns={c: f"previous_{c}" for c in keep[1:]}), on="supplier_id", how="outer", indicator=True)
    names = data.suppliers.set_index("supplier_id")
    merged["supplier_name"] = merged["supplier_id"].map(names["supplier_name"])
    merged["annual_spend_usd"] = merged["supplier_id"].map(names["annual_spend_usd"])
    rank = {t: i for i, t in enumerate(TIERS)}
    step = merged["previous_tier"].map(rank) - merged["tier"].map(rank)  # positive: more serious now
    merged["move"] = np.select([merged["_merge"] == "left_only", merged["_merge"] == "right_only", step > 0, step < 0],
                               ["new", "left", "moved up", "moved down"], "unchanged")
    merged["tier"] = merged["tier"].where(merged["_merge"] != "right_only", "Left the base")
    merged["previous_tier"] = merged["previous_tier"].where(merged["_merge"] != "left_only", "Not active")
    merged["score_change"] = merged["overall"] - merged["previous_overall"]
    merged["probability_change"] = merged["incident_probability"] - merged["previous_incident_probability"]

    hist = data.history.set_index(["supplier_id", "month"])
    incidents = data.incidents[data.incidents["month"] == month].groupby("supplier_id")["incident_type"].agg(list).to_dict()
    yes = lambda v: bool(v) if not pd.isna(v) else False  # noqa: E731
    flags = []
    for _, row in merged.iterrows():
        sid, found = row["supplier_id"], []
        for kind in incidents.get(sid, []):
            found.append(f"incident recorded: {kind.replace('_', ' ')}")
        if row["_merge"] == "right_only":
            found.append("no longer in the supplier base")
        if yes(row["deteriorating"]) and not yes(row["previous_deteriorating"]):
            found.append("deterioration flag raised")
        if (sid, month) in hist.index:
            this = hist.loc[(sid, month)]
            last = hist.loc[(sid, previous)] if (sid, previous) in hist.index else None
            if this["compliance_flag"] and (last is None or not last["compliance_flag"]):
                found.append("compliance flag opened")
            if this["ownership_change"]:
                found.append("ownership changed")
        flags.append("; ".join(found))
    merged["new_flags"] = flags
    order = merged["move"].map({"moved up": 0, "left": 1, "new": 2, "moved down": 3, "unchanged": 4})
    merged = merged.assign(_o=order, _f=(merged["new_flags"] == "")).sort_values(["_o", "_f", "score_change"], ascending=[True, True, False])
    merged["month"] = month
    return merged[["supplier_id", "supplier_name", "previous_tier", "tier", "move", "overall", "score_change", "incident_probability",
                   "probability_change", "new_flags", "annual_spend_usd", "month"]].reset_index(drop=True)


def changes_headline(changes: pd.DataFrame) -> str:
    """One paragraph on what changed since the month before."""
    if changes.empty:
        return "Nothing to compare."
    month = pd.Timestamp(changes["month"].iloc[0])
    count = changes["move"].value_counts()
    up, down, left = (int(count.get(k, 0)) for k in ("moved up", "moved down", "left"))
    flagged = int((changes["new_flags"] != "").sum())
    text = (f"Since the month before {month:%B %Y}: {up} supplier{'' if up == 1 else 's'} moved to a more serious tier, "
            f"{down} moved to a less serious one, and {flagged} {'has' if flagged == 1 else 'have'} a new flag or incident.")
    if left:
        text += f" {left} left the supplier base."
    risen = changes[changes["move"] == "moved up"]
    if len(risen):
        top = risen.sort_values("annual_spend_usd", ascending=False).iloc[0]
        text += f" The largest supplier to move up is {top['supplier_name']} ({top['previous_tier']} to {top['tier']})."
    return text
