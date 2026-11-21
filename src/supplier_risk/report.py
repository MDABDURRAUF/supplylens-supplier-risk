"""Printable outputs: the supplier risk profile and the monthly portfolio risk report, each a single HTML file."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from . import evaluate
from .data import SupplierData
from .explain import explain_features
from .features import build_features
from .model import IncidentModel
from .monitor import TierRules, changes_headline, watchlist, watchlist_headline, what_changed
from .portfolio import concentration, dual_source_what_if, hhi, probability_impact, spend_at_risk
from .score import DIMENSIONS, ScoreConfig, explain_row, score_panel


def money(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def pct(x, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def _num(x, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def _signed(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.{digits}f}"


def _month(x) -> str:
    return "" if x is None or pd.isna(x) else f"{pd.Timestamp(x):%b %Y}"


_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 8px;font-size:19px} p,li{line-height:1.5}
p.note{color:#56627a;font-size:14px;margin:0 0 12px}
.lead{font-size:16px;background:#fff;border:1px solid #dfe3ea;border-left:5px solid #3a6ea5;border-radius:8px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:21px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #edf0f4;vertical-align:top}
th{background:#eef1f6;font-weight:600;white-space:nowrap} td.num{text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;color:#fff;white-space:nowrap}
.caveat{background:#fff8e6;border:1px solid #f0d58a;border-radius:8px;padding:12px 14px;font-size:14px}
.chart{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:10px}
@media print{body{background:#fff}}
"""
_TIER = {"Critical": "#d1495b", "Elevated": "#e9a23b", "Normal": "#9aa5b1", "Left the base": "#7a1f2b", "Not active": "#9aa5b1"}
_NOT_A_RATING = ("The risk score is a monitoring aid built from the records supplied and a set of illustrative thresholds and "
                 "weights. It is not a credit rating and says nothing certain about any supplier's solvency. The incident "
                 "probability is an estimate from past patterns: among suppliers given 30%, about three in ten are expected to "
                 "have an incident in the following quarter.")


def _table(df: pd.DataFrame, formats: dict | None = None, empty: str = "Nothing to show.") -> str:
    if df is None or df.empty:
        return f'<p class="note">{escape(empty)}</p>'
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if c in ("tier", "previous_tier"):
                cells.append(f'<td><span class="tag" style="background:{_TIER.get(v, "#9aa5b1")}">{escape(str(v))}</span></td>')
            elif c in formats:
                cells.append(f'<td class="num">{escape(formats[c](v))}</td>')
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kpis(items) -> str:
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for k, v in items) + "</div>"


def _page(title: str, subtitle: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main><h1>{escape(title)}</h1>"
            f'<p class="note">{escape(subtitle)}</p>{body}</main></body></html>')


def _line_chart(months, values, marks=(), width: int = 980, height: int = 200) -> str:
    """A small inline SVG line chart of a 0-100 score by month; ``marks`` are months to flag (incidents)."""
    months, values = list(months), [float(v) for v in values]
    if len(values) < 2:
        return ""
    left, right, top, bottom = 34, 12, 10, 24
    x = lambda i: left + (width - left - right) * i / (len(values) - 1)  # noqa: E731
    y = lambda v: top + (height - top - bottom) * (1 - v / 100)  # noqa: E731
    grid = "".join(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#edf0f4"/>'
                   f'<text x="{left - 6}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#56627a">{v}</text>' for v in (0, 25, 50, 75, 100))
    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    step = max(1, len(months) // 8)
    labels = "".join(f'<text x="{x(i):.1f}" y="{height - 6}" font-size="11" text-anchor="middle" fill="#56627a">{_month(m)}</text>'
                     for i, m in enumerate(months) if i % step == 0)
    flagged = set(pd.Timestamp(m) for m in marks)
    dots = "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="5" fill="#d1495b"/>' for i, (m, v) in enumerate(zip(months, values))
                   if pd.Timestamp(m) in flagged)
    return (f'<div class="chart"><svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Overall risk score by month">'
            f'{grid}<polyline points="{points}" fill="none" stroke="#3a6ea5" stroke-width="2"/>{dots}{labels}</svg></div>')


def what_if_text(table: pd.DataFrame) -> pd.DataFrame:
    """The dual-sourcing what-if with each figure written in its own unit (dollars, percent, points)."""
    def write(figure: str, v) -> str:
        if pd.isna(v):
            return "n/a"
        if figure.startswith("share"):
            return f"{v:.1%}"
        if "spend" in figure:
            return money(v)
        return f"{v:,.1f}"

    out = table.copy()
    for c in ("before", "after", "change"):
        out[c] = [write(f, v) for f, v in zip(table["figure"], table[c])]
    return out


def build_supplier_profile(data: SupplierData, supplier_id: str, as_of=None, model: IncidentModel | None = None,
                           config: ScoreConfig | None = None, rules: TierRules | None = None, note: str = "") -> str:
    """The risk profile of one supplier as an HTML page.

    Shows the tier and the rule behind it, each dimension score, one sentence
    per sub-score, what drives the model's prediction, the score over the
    last 24 months with recorded incidents marked, the sole-sourced items and
    what a second source would change. Uses only records up to ``as_of``.
    """
    config = config or ScoreConfig()
    month = data.month(as_of)
    watch = watchlist(data, month, model, config, rules, explain=False)
    row = watch[watch["supplier_id"] == supplier_id]
    if row.empty:
        raise ValueError(f"Supplier '{supplier_id}' is not active in {month:%B %Y}")
    row = row.iloc[0]
    panel = score_panel(data, config)
    mine = panel[(panel["supplier_id"] == supplier_id) & (panel["month"] <= month)]
    now = mine.iloc[-1]
    where = f"{row['state']}, US ({row['region']})" if row["country"] == "US" else row["country"]
    lead = f"{row['supplier_name']} is {row['tier']} for {month:%B %Y}. {row['reason']}"
    if model is not None:
        lead += (f" The model puts the chance of an incident in the next {model.horizon} months at "
                 f"{row['incident_probability']:.0%}, against {model.prior_:.0%} for a typical supplier in its training period.")
    body = f'<p class="lead">{escape(lead)}</p>'
    body += _kpis([("overall risk score (0-100)", _num(row["overall"])), ("trend, points a month", _signed(row["trend"])),
                   ("incident probability, next quarter", pct(row["incident_probability"])), ("annual spend", money(row["annual_spend_usd"])),
                   ("sole-sourced spend", money(row["sole_sourced_spend_usd"])), ("location", where)])

    weights = config.normalised_weights()
    dims = pd.DataFrame({"dimension": DIMENSIONS, "score": [now[d] for d in DIMENSIONS], "weight": [weights[d] for d in DIMENSIONS],
                         "trend_points_a_month": [now[f"trend_{d}"] for d in DIMENSIONS]})
    body += "<h2>Score by dimension</h2><p class=\"note\">Higher is riskier. The overall score is the weighted average of the five.</p>"
    body += _table(dims, {"score": _num, "weight": pct, "trend_points_a_month": _signed})
    why = explain_row(now, config)[["dimension", "score", "points", "sentence"]]
    body += ("<h2>Why the score is what it is</h2><p class=\"note\">One sentence per measure. Points is what the measure adds to the "
             "overall score. All thresholds are illustrative settings.</p>")
    body += _table(why, {"score": _num, "points": lambda v: _num(v, 1)})

    if model is not None:
        feats = build_features(data)
        x = feats[(feats["supplier_id"] == supplier_id) & (feats["month"] == month)][model.features_]
        effects = explain_features(model, x).iloc[0].sort_values(ascending=False)
        drivers = pd.DataFrame({"factor": effects.index, "effect_on_incident_probability": effects.to_numpy()})
        body += ("<h2>What drives the model's prediction</h2><p class=\"note\">Change in the incident probability against a typical "
                 "supplier, one factor at a time. Positive raises the risk. The effects do not add up exactly to the prediction.</p>")
        body += _table(drivers, {"effect_on_incident_probability": lambda v: f"{v * 100:+.1f} pts"})

    recent = mine.tail(24)
    incidents = data.incidents[(data.incidents["supplier_id"] == supplier_id) & (data.incidents["month"] <= month)]
    body += "<h2>Overall score, last 24 months</h2><p class=\"note\">Red dots mark months with a recorded incident.</p>"
    body += _line_chart(recent["month"], recent["overall"], incidents["month"])
    hist = data.history[(data.history["supplier_id"] == supplier_id) & (data.history["month"] <= month)].tail(12)
    table = hist.merge(mine[["month", "overall"]], on="month")[["month", "overall", "otif_rate", "defect_ppm", "lead_time_cv", "current_ratio",
                                                                  "debt_to_equity", "operating_margin", "days_payable", "audit_findings", "compliance_flag"]]
    body += "<h2>Records, last 12 months</h2>" + _table(table, {
        "month": _month, "overall": _num, "otif_rate": lambda v: pct(v, 1), "defect_ppm": lambda v: f"{v:,.0f}", "lead_time_cv": lambda v: _num(v, 2),
        "current_ratio": lambda v: _num(v, 2), "debt_to_equity": lambda v: _num(v, 2), "operating_margin": lambda v: pct(v, 1),
        "days_payable": _num, "audit_findings": _num, "compliance_flag": lambda v: "yes" if v else ""})
    body += "<h2>Incidents on record</h2>" + _table(
        incidents.assign(incident_type=incidents["incident_type"].str.replace("_", " "))[["month", "incident_type"]],
        {"month": _month}, empty="No incidents on record up to this month.")

    items = data.items[data.items["supplier_id"] == supplier_id]
    body += "<h2>Items bought from this supplier</h2>" + _table(
        items.assign(sole_sourced=items["sole_sourced"].map({1: "yes", 0: ""}))[["item_id", "item_name", "annual_spend_usd", "sole_sourced"]],
        {"annual_spend_usd": money}, empty="No items are recorded for this supplier.")
    if row["sole_sourced_spend_usd"] > 0:
        what_if = what_if_text(dual_source_what_if(data, supplier_id, 0.5, month, model, config, rules))
        body += ("<h2>If a second source took half of the sole-sourced volume</h2><p class=\"note\">The incident probability is left "
                 "unchanged; the second source's own risk and the cost of qualifying it are not modelled.</p>")
        body += _table(what_if, {c: str for c in ("before", "after", "change")})
    body += f'<h2>How to read this</h2><p class="caveat">{escape(_NOT_A_RATING)}</p>'
    subtitle = f"SupplyLens Supplier Risk. {supplier_id}, {row['category']}, {where}. As of {month:%B %Y}." + (f" {note}" if note else "")
    return _page(f"Supplier risk profile: {row['supplier_name']}", subtitle, body)


def build_portfolio_report(data: SupplierData, as_of=None, model: IncidentModel | None = None, config: ScoreConfig | None = None,
                           rules: TierRules | None = None, note: str = "") -> str:
    """The monthly portfolio risk report as an HTML page.

    Headline, the suppliers on watch with the reason for each, what changed
    since last month, spend at risk, concentration, the suppliers that are
    both likely to have an incident and large, and (with a model) how the
    model did on months it had not seen.
    """
    month = data.month(as_of)
    watch = watchlist(data, month, model, config, rules)
    s = spend_at_risk(watch)
    body = f'<p class="lead">{escape(watchlist_headline(watch))}</p>'
    counts = watch["tier"].value_counts()
    body += _kpis([("active suppliers", f"{len(watch)}"), ("Critical", f"{int(counts.get('Critical', 0))}"),
                   ("Elevated", f"{int(counts.get('Elevated', 0))}"), ("spend on watch", money(s["watch_spend"])),
                   ("share of spend on watch", pct(s["watch_share"])), ("probability-weighted spend", money(s["probability_weighted_spend"])),
                   ("sole-sourced spend on watch", money(s["sole_sourced_watch_spend"]))])
    on_watch = watch[watch["tier"] != "Normal"]
    cols = ["supplier_id", "supplier_name", "tier", "overall", "trend", "incident_probability", "annual_spend_usd", "reason", "main_drivers"]
    body += ("<h2>Suppliers on watch</h2><p class=\"note\">Overall is the risk score from 0 to 100; trend is its change in points a "
             "month; main drivers are the factors raising the model's prediction.</p>")
    body += _table(on_watch[cols], {"overall": _num, "trend": _signed, "incident_probability": pct, "annual_spend_usd": money},
                   empty="No supplier is Critical or Elevated this month.")

    if list(data.months).index(month) > 0:
        changes = what_changed(data, month, model, config, rules)
        moved = changes[(changes["move"] != "unchanged") | (changes["new_flags"] != "")]
        body += f"<h2>What changed since last month</h2><p class=\"note\">{escape(changes_headline(changes))}</p>"
        body += _table(moved[["supplier_id", "supplier_name", "previous_tier", "tier", "score_change", "probability_change", "new_flags"]],
                       {"score_change": _signed, "probability_change": lambda v: "n/a" if pd.isna(v) else f"{v * 100:+.0f} pts"},
                       empty="No tier moves and no new flags.")

    by_supplier, by_region = concentration(watch, "supplier"), concentration(watch, "region")
    body += ("<h2>Concentration</h2><p class=\"note\">"
             f"The ten largest suppliers take {pct(by_supplier['share'].head(10).sum())} of spend. The concentration index (HHI, 0 to 10,000) "
             f"is {hhi(by_supplier['spend']):,.0f} by supplier and {hhi(by_region['spend']):,.0f} by region; 10,000 would mean everything "
             "in one place.</p>")
    body += _table(by_region, {"spend": money, "share": pct, "cumulative_share": pct, "average_score": _num})
    body += "<h2>Largest suppliers</h2>" + _table(by_supplier.head(10), {"spend": money, "share": pct, "cumulative_share": pct, "average_score": _num})

    if model is not None:
        pi = probability_impact(watch)
        body += ("<h2>Likely and large</h2><p class=\"note\">Suppliers with an incident probability of at least 20% whose spend is in the "
                 "top quarter. Spend at risk is annual spend times incident probability: a ranking aid, not a forecast of loss.</p>")
        body += _table(pi[pi["quadrant"] == "act now"].drop(columns=["quadrant", "overall"]),
                       {"incident_probability": pct, "annual_spend_usd": money, "sole_sourced_spend_usd": money, "spend_at_risk": money},
                       empty="No supplier is both likely to have an incident and in the top quarter by spend.")
        if model.holdout_ is not None:
            q, split = evaluate.summary(model), model.split_
            lo, hi = evaluate.auc_interval(model)
            lift = evaluate.lift_table(model)
            body += "<h2>How far to trust the model</h2>"
            body += (f"<p>Trained on supplier-months up to {_month(split['train_until'])}, calibrated on {_month(split['calibration_from'])} to "
                     f"{_month(split['calibration_until'])}, and tested on {q['rows']:,} supplier-months from {_month(split['test_from'])} on. "
                     f"On the test period the ROC AUC is {q['roc_auc']:.3f} (rough 95% range {lo:.2f} to {hi:.2f}), and the riskiest fifth "
                     f"of supplier-months holds {lift['cumulative_capture'].iloc[0]:.0%} of those followed by an incident.</p>")
            num = lambda v: "" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
            body += _table(evaluate.compare_models(model), {"roc_auc": num, "average_precision": num,
                                                            "brier_score": lambda v: f"{v:.4f}", "log_loss": lambda v: f"{v:.4f}"})
    body += (f'<h2>Limits</h2><p class="caveat">{escape(_NOT_A_RATING)} Thresholds, weights and the regional hazard index are '
             "illustrative settings, not measurements. With a supplier base of this size the model's accuracy figures rest on a small "
             "number of incidents and will move when it is refitted.</p>")
    subtitle = "SupplyLens Supplier Risk." + (f" {note}" if note else "")
    return _page(f"Portfolio risk report: {month:%B %Y}", subtitle, body)


def write_html(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
