"""The transparent risk score: 0 to 100 per dimension and overall, with a trend.

Higher means riskier. Every number is built the same way:

1. A **measure** is read from the records (for example on-time-in-full
   averaged over the last three months).
2. It becomes a **sub-score** on a straight line between a ``good`` value
   (scores 0) and a ``bad`` value (scores 100), capped at both ends.
3. A **dimension** score is the weighted average of its sub-scores.
4. The **overall** score is the weighted average of the five dimensions.

The good and bad values, and all the weights, are illustrative defaults: set
your own in ``ScoreConfig``. The score for a month uses only that month and
earlier months. It is a monitoring aid, not a credit rating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import SupplierData

DIMENSIONS = ["operational", "financial", "geographic", "dependency", "compliance"]
DEFAULT_WEIGHTS = {"operational": 0.30, "financial": 0.25, "geographic": 0.10, "dependency": 0.15, "compliance": 0.20}


@dataclass(frozen=True)
class Measure:
    """One input to the score: where it comes from and which values count as good and bad."""

    key: str
    dimension: str
    weight: float       # weight within its dimension; the weights of a dimension sum to 1
    good: float         # this value or better scores 0
    bad: float          # this value or worse scores 100
    kind: str           # how the value is written: percent, ppm, ratio, days, count, index, flag
    text: str           # start of the one-sentence explanation


MEASURES = [
    Measure("otif", "operational", 0.45, 0.98, 0.85, "percent", "On-time-in-full averaged {value} over the last {window} months"),
    Measure("defects", "operational", 0.30, 500, 5000, "ppm", "Defects averaged {value} over the last {window} months"),
    Measure("lead_time", "operational", 0.25, 0.10, 0.40, "ratio", "Lead-time variability (standard deviation over mean) averaged {value} over the last {window} months"),
    Measure("current_ratio", "financial", 0.30, 2.0, 0.8, "ratio", "The current ratio averaged {value} over the last {window} months"),
    Measure("debt_to_equity", "financial", 0.25, 0.5, 3.0, "ratio", "Debt-to-equity averaged {value} over the last {window} months"),
    Measure("operating_margin", "financial", 0.25, 0.10, -0.02, "percent", "Operating margin averaged {value} over the last {window} months"),
    Measure("days_payable", "financial", 0.20, 45, 120, "days", "The supplier took {value} on average to pay its own bills over the last {window} months"),
    Measure("hazard", "geographic", 0.50, 0.0, 1.0, "index", "The natural-hazard exposure index for its location is {value}"),
    Measure("single_site", "geographic", 0.30, 0.0, 1.0, "flag", "The supplier makes everything at a single site|The supplier has more than one site"),
    Measure("cross_border", "geographic", 0.20, 0.0, 1.0, "flag", "The supplier is outside the US, so goods cross a border|The supplier is in the US"),
    Measure("spend_share", "dependency", 0.40, 0.0, 0.05, "percent", "This supplier takes {value} of total spend"),
    Measure("sole_source_share", "dependency", 0.60, 0.0, 1.0, "percent", "{value} of the spend with this supplier is on items it alone supplies"),
    Measure("audit_findings", "compliance", 0.40, 0.0, 5.0, "count", "Open audit findings averaged {value} over the last {window} months"),
    Measure("compliance_flag", "compliance", 0.40, 0.0, 1.0, "flag", "A compliance flag was open in the last {flag_months} months|No compliance flag was open in the last {flag_months} months"),
    Measure("ownership_change", "compliance", 0.20, 0.0, 1.0, "flag", "Ownership changed in the last {ownership_months} months|Ownership did not change in the last {ownership_months} months"),
]
MEASURE_KEYS = [m.key for m in MEASURES]


@dataclass(frozen=True)
class ScoreConfig:
    """Settings for the score. All defaults are illustrative.

    ``weights``              weight of each dimension in the overall score (rescaled to sum to 1)
    ``thresholds``           ``{measure key: (good, bad)}`` to replace the default good and bad values
    ``window``               months averaged for delivery, quality, financial and audit measures
    ``flag_months``          how long a compliance flag keeps counting
    ``ownership_months``     how long an ownership change keeps counting
    ``trend_months``         months the trend is measured over
    ``deterioration_slope``  the deterioration flag is raised when the overall score has been
                             rising by at least this many points per month
    """

    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    thresholds: dict = field(default_factory=dict)
    window: int = 3
    flag_months: int = 6
    ownership_months: int = 12
    trend_months: int = 6
    deterioration_slope: float = 2.0

    def normalised_weights(self) -> dict:
        unknown = set(self.weights) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"Unknown dimension(s) in weights: {', '.join(sorted(unknown))}")
        w = {d: float(self.weights.get(d, 0.0)) for d in DIMENSIONS}
        if min(w.values()) < 0 or sum(w.values()) <= 0:
            raise ValueError("Weights must be zero or greater and not all zero")
        return {d: v / sum(w.values()) for d, v in w.items()}

    def limits(self, measure: Measure) -> tuple[float, float]:
        good, bad = self.thresholds.get(measure.key, (measure.good, measure.bad))
        if good == bad:
            raise ValueError(f"The good and bad values for '{measure.key}' must differ")
        return float(good), float(bad)


def scale(value, good: float, bad: float):
    """Sub-score from 0 to 100: 0 at ``good`` or better, 100 at ``bad`` or worse, a straight line between."""
    return 100.0 * np.clip((np.asarray(value, dtype=float) - good) / (bad - good), 0.0, 1.0)


def slope(values) -> float:
    """Least-squares slope of evenly spaced values, per step. Missing with fewer than three values."""
    y = np.asarray(values, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return float("nan")
    x = np.arange(len(y)) - (len(y) - 1) / 2
    return float((x * y).sum() / (x * x).sum())


def measure_panel(data: SupplierData, config: ScoreConfig | None = None) -> pd.DataFrame:
    """The value of every measure for every supplier and month, from that month and earlier months only."""
    config = config or ScoreConfig()
    h = data.history.sort_values(["supplier_id", "month"]).reset_index(drop=True)
    g = h.groupby("supplier_id", sort=False)

    def rolling(column: str, months: int, how: str) -> pd.Series:
        return getattr(g[column].rolling(months, min_periods=1), how)().reset_index(level=0, drop=True)

    m = h[["supplier_id", "month"]].copy()
    w = config.window
    for key, column in (("otif", "otif_rate"), ("defects", "defect_ppm"), ("lead_time", "lead_time_cv"),
                        ("current_ratio", "current_ratio"), ("debt_to_equity", "debt_to_equity"),
                        ("operating_margin", "operating_margin"), ("days_payable", "days_payable"),
                        ("audit_findings", "audit_findings")):
        m[key] = rolling(column, w, "mean")
    m["compliance_flag"] = rolling("compliance_flag", config.flag_months, "max")
    m["ownership_change"] = rolling("ownership_change", config.ownership_months, "max")
    s = data.suppliers.set_index("supplier_id")
    for key, column in (("hazard", "hazard_index"), ("single_site", "single_site"), ("cross_border", "cross_border")):
        m[key] = m["supplier_id"].map(s[column]).astype(float)
    spend = m["supplier_id"].map(s["annual_spend_usd"]).astype(float)
    sole = m["supplier_id"].map(s["sole_sourced_spend_usd"]).astype(float)
    # Share of the spend with suppliers that were active in that month.
    m["spend_share"] = spend / spend.groupby(m["month"]).transform("sum").clip(lower=1e-9)
    m["sole_source_share"] = (sole / spend.clip(lower=1e-9)).clip(0, 1)
    return m


def score_panel(data: SupplierData, config: ScoreConfig | None = None) -> pd.DataFrame:
    """Scores for every supplier and month.

    Columns: the measures (``m_<key>``), their sub-scores (``s_<key>``), the
    five dimension scores, ``overall``, and for the overall score and each
    dimension a ``trend`` (``trend_<dimension>``): the least-squares slope
    over the last ``trend_months`` months, in points per month. Positive
    means getting riskier. ``change_1m`` is the overall score minus last
    month's. ``deteriorating`` is true when the overall trend is at least
    ``deterioration_slope``, whatever the level of the score.
    """
    config = config or ScoreConfig()
    if repr(config) in data._cache:
        return data._cache[repr(config)].copy()
    weights = config.normalised_weights()
    m = measure_panel(data, config)
    out = m[["supplier_id", "month"]].copy()
    for measure in MEASURES:
        out[f"m_{measure.key}"] = m[measure.key]
    for measure in MEASURES:
        out[f"s_{measure.key}"] = scale(m[measure.key], *config.limits(measure))
    for d in DIMENSIONS:
        parts = [x for x in MEASURES if x.dimension == d]
        out[d] = sum(x.weight * out[f"s_{x.key}"] for x in parts) / sum(x.weight for x in parts)
    out["overall"] = sum(weights[d] * out[d] for d in DIMENSIONS)
    g = out.groupby("supplier_id", sort=False)
    for column in ["overall"] + DIMENSIONS:
        name = "trend" if column == "overall" else f"trend_{column}"
        out[name] = g[column].rolling(config.trend_months, min_periods=3).apply(slope, raw=True).reset_index(level=0, drop=True)
    out["change_1m"] = g["overall"].diff()
    out["deteriorating"] = out["trend"] >= config.deterioration_slope
    data._cache[repr(config)] = out
    return out.copy()


def score_as_of(data: SupplierData, as_of=None, config: ScoreConfig | None = None) -> pd.DataFrame:
    """Scores for every supplier active in the month ``as_of`` (default: the latest), riskiest first."""
    month = data.month(as_of)
    panel = score_panel(data, config)
    now = panel[panel["month"] == month]
    info = data.suppliers[["supplier_id", "supplier_name", "category", "country", "state", "region", "annual_spend_usd"]]
    cols = ["supplier_id", "month", "overall"] + DIMENSIONS + ["trend", "change_1m", "deteriorating"]
    return info.merge(now[cols], on="supplier_id").sort_values("overall", ascending=False).reset_index(drop=True)


def _write(value: float, kind: str) -> str:
    if kind == "percent":
        return f"{value:.1%}"
    if kind == "ppm":
        return f"{value:,.0f} PPM"
    if kind == "days":
        return f"{value:.0f} days"
    if kind == "count":
        return f"{value:.1f}"
    if kind == "flag":
        return "yes" if value > 0 else "no"
    return f"{value:.2f}"


def explain_row(row: pd.Series, config: ScoreConfig | None = None) -> pd.DataFrame:
    """One sentence per sub-score for one row of ``score_panel``.

    Columns: ``dimension``, ``measure``, ``value`` (as text), ``score``,
    ``weight`` (within the dimension), ``points`` (what the sub-score adds
    to the overall score) and ``sentence``.
    """
    config = config or ScoreConfig()
    weights = config.normalised_weights()
    periods = {"window": config.window, "flag_months": config.flag_months, "ownership_months": config.ownership_months}
    rows = []
    for x in MEASURES:
        value, score = float(row[f"m_{x.key}"]), float(row[f"s_{x.key}"])
        good, bad = config.limits(x)
        if x.kind == "flag":
            yes, no = x.text.split("|")
            sentence = f"{(yes if value > 0 else no).format(**periods)}, which scores {score:.0f}."
        else:
            start = x.text.format(value=_write(value, x.kind), **periods)
            better, worse = ("or less", "or more") if bad > good else ("or more", "or less")
            sentence = (f"{start}; {_write(good, x.kind)} {better} scores 0 and {_write(bad, x.kind)} {worse} scores 100, "
                        f"so this scores {score:.0f}.")
        share = x.weight / sum(y.weight for y in MEASURES if y.dimension == x.dimension)
        rows.append({"dimension": x.dimension, "measure": x.key, "value": _write(value, x.kind), "score": score, "weight": share,
                     "points": score * share * weights[x.dimension], "sentence": sentence})
    return pd.DataFrame(rows)


def explain_score(data: SupplierData, supplier_id: str, as_of=None, config: ScoreConfig | None = None) -> pd.DataFrame:
    """Why a supplier has the score it has in a month: one sentence per sub-score (see ``explain_row``)."""
    month = data.month(as_of)
    panel = score_panel(data, config)
    row = panel[(panel["supplier_id"] == supplier_id) & (panel["month"] == month)]
    if row.empty:
        raise ValueError(f"Supplier '{supplier_id}' has no history for {month:%B %Y}")
    return explain_row(row.iloc[0], config)
