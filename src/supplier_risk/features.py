"""Model inputs and the outcome to predict, built so that nothing from the future leaks in.

Every feature for a supplier in month *t* is computed from that supplier's
records for month *t* and earlier. The outcome (an incident in the following
quarter) is only treated as known once that whole quarter is in the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SupplierData

_LEVELS = {"otif": "otif_rate", "log_ppm": "log_ppm", "lead_time_cv": "lead_time_cv", "current_ratio": "current_ratio",
           "debt_to_equity": "debt_to_equity", "operating_margin": "operating_margin", "days_payable": "days_payable"}
FEATURES = ([f"{k}_3m" for k in _LEVELS] + [f"{k}_vs_usual" for k in _LEVELS]
            + ["audit_findings_3m", "compliance_flag_6m", "ownership_change_12m", "incidents_12m", "months_since_incident",
               "single_site", "hazard_index", "cross_border"])

# Groups used to explain a prediction in plain terms.
FACTOR_GROUPS = {
    "delivery performance": ["otif_3m", "otif_vs_usual", "lead_time_cv_3m", "lead_time_cv_vs_usual"],
    "quality": ["log_ppm_3m", "log_ppm_vs_usual"],
    "financial health": ["current_ratio_3m", "current_ratio_vs_usual", "debt_to_equity_3m", "debt_to_equity_vs_usual",
                         "operating_margin_3m", "operating_margin_vs_usual", "days_payable_3m", "days_payable_vs_usual"],
    "compliance and ownership": ["audit_findings_3m", "compliance_flag_6m", "ownership_change_12m"],
    "incident history": ["incidents_12m", "months_since_incident"],
    "location": ["single_site", "hazard_index", "cross_border"],
}
NO_INCIDENT_MONTHS = 36.0  # value of months_since_incident when no incident is on record

# The direction each input is allowed to move the prediction: +1 means a higher value can only raise the
# predicted risk, -1 can only lower it, 0 is unconstrained. The data decides how much each input matters;
# these only stop a small data set from teaching the model something backwards.
_RISK_UP = {"otif": -1, "log_ppm": 1, "lead_time_cv": 1, "current_ratio": -1, "debt_to_equity": 1, "operating_margin": -1, "days_payable": 1}
DIRECTION = {**{f"{k}_3m": v for k, v in _RISK_UP.items()}, **{f"{k}_vs_usual": v for k, v in _RISK_UP.items()},
             "audit_findings_3m": 1, "compliance_flag_6m": 1, "ownership_change_12m": 1, "incidents_12m": 1,
             "months_since_incident": -1, "single_site": 1, "hazard_index": 1, "cross_border": 0}


def build_features(data: SupplierData, recent: int = 3, usual_months: int = 24) -> pd.DataFrame:
    """One row per supplier and month: ``supplier_id``, ``month``, the model inputs, and ``history_months``.

    ``<measure>_3m``          average over the latest ``recent`` months
    ``<measure>_vs_usual``    that average minus the supplier's own average over the (up to)
                              ``usual_months`` months before them: how far the supplier is from its
                              own normal. Missing until there are three earlier months.
    ``audit_findings_3m``     open audit findings, averaged over the latest ``recent`` months
    ``compliance_flag_6m``    1 if a compliance flag was open in the last 6 months
    ``ownership_change_12m``  1 if ownership changed in the last 12 months
    ``incidents_12m``         incidents recorded in the last 12 months, this month included
    ``months_since_incident`` months since the latest recorded incident (36 if none)
    ``single_site``, ``hazard_index``, ``cross_border``   where the supplier is
    ``history_months``        months of history up to this one (not a model input)

    Dependency measures (share of spend, sole-sourcing) are left out on
    purpose: they describe how much an incident would hurt, not how likely
    one is.
    """
    cached = ("features", recent, usual_months)
    if cached in data._cache:
        return data._cache[cached].copy()
    h = data.history.sort_values(["supplier_id", "month"]).reset_index(drop=True)
    h["log_ppm"] = np.log1p(h["defect_ppm"])
    counts = data.incidents.groupby(["supplier_id", "month"]).size().rename("n_incidents").reset_index()
    h = h.merge(counts, on=["supplier_id", "month"], how="left")
    h["n_incidents"] = h["n_incidents"].fillna(0.0)
    g = h.groupby("supplier_id", sort=False)

    def rolling(column: str, months: int, how: str) -> pd.Series:
        return getattr(g[column].rolling(months, min_periods=1), how)().reset_index(level=0, drop=True)

    x = h[["supplier_id", "month"]].copy()
    for key, column in _LEVELS.items():
        x[f"{key}_3m"] = rolling(column, recent, "mean")
    for key, column in _LEVELS.items():
        usual = g[column].transform(lambda s: s.shift(recent).rolling(usual_months, min_periods=3).mean())
        x[f"{key}_vs_usual"] = x[f"{key}_3m"] - usual
    x["audit_findings_3m"] = rolling("audit_findings", recent, "mean")
    x["compliance_flag_6m"] = rolling("compliance_flag", 6, "max")
    x["ownership_change_12m"] = rolling("ownership_change", 12, "max")
    x["incidents_12m"] = rolling("n_incidents", 12, "sum")
    position = g.cumcount()
    last_seen = position.where(h["n_incidents"] > 0).groupby(h["supplier_id"]).ffill()
    x["months_since_incident"] = (position - last_seen).fillna(NO_INCIDENT_MONTHS).clip(upper=NO_INCIDENT_MONTHS)
    s = data.suppliers.set_index("supplier_id")
    for column in ("single_site", "hazard_index", "cross_border"):
        x[column] = x["supplier_id"].map(s[column]).astype(float)
    x["history_months"] = position + 1
    data._cache[cached] = x[["supplier_id", "month"] + FEATURES + ["history_months"]]
    return data._cache[cached].copy()


def incident_labels(data: SupplierData, rows: pd.DataFrame, horizon: int = 3) -> pd.Series:
    """For each supplier and month in ``rows``: did the supplier have an incident in the next ``horizon`` months?

    1 or 0 where the answer is known, missing where it is not: the outcome
    for month *t* is known only when the data runs to month *t + horizon*.
    An insolvency counts even though the supplier's history stops before it.
    """
    hits = set()
    for sid, month in zip(data.incidents["supplier_id"], data.incidents["month"]):
        for k in range(1, horizon + 1):
            hits.add((sid, month - pd.DateOffset(months=k)))
    label = pd.Series([float((sid, month) in hits) for sid, month in zip(rows["supplier_id"], rows["month"])], index=rows.index)
    known = rows["month"] + pd.DateOffset(months=horizon) <= data.last_month
    return label.where(known)
