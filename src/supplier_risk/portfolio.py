"""The portfolio view: spend at risk, concentration, probability against impact, and a dual-sourcing what-if."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SupplierData
from .model import IncidentModel
from .monitor import TierRules, watchlist
from .score import MEASURES, ScoreConfig, scale


def spend_at_risk(watch: pd.DataFrame) -> dict:
    """Spend totals for a watchlist, in dollars of annual spend.

    ``total_spend``                 all active suppliers
    ``critical_spend``, ``elevated_spend``   spend with suppliers in each tier
    ``watch_spend``, ``watch_share``         Critical plus Elevated, and its share of the total
    ``probability_weighted_spend``  sum of annual spend times incident probability: the spend you
                                    would expect to sit with a supplier that has an incident next
                                    quarter. It ranks exposure; it is not a forecast of loss.
    ``sole_sourced_spend``          spend on items with a single source
    ``sole_sourced_watch_spend``    the part of that with Critical or Elevated suppliers
    """
    on_watch = watch["tier"].isin(["Critical", "Elevated"])
    total = float(watch["annual_spend_usd"].sum())
    out = {"total_spend": total,
           "critical_spend": float(watch.loc[watch["tier"] == "Critical", "annual_spend_usd"].sum()),
           "elevated_spend": float(watch.loc[watch["tier"] == "Elevated", "annual_spend_usd"].sum()),
           "watch_spend": float(watch.loc[on_watch, "annual_spend_usd"].sum()),
           "probability_weighted_spend": float(watch["spend_at_risk"].sum()) if watch["spend_at_risk"].notna().any() else float("nan"),
           "sole_sourced_spend": float(watch["sole_sourced_spend_usd"].sum()),
           "sole_sourced_watch_spend": float(watch.loc[on_watch, "sole_sourced_spend_usd"].sum())}
    out["watch_share"] = out["watch_spend"] / total if total else 0.0
    return out


def hhi(spend) -> float:
    """Herfindahl-Hirschman index of a set of spend amounts, on the 0 to 10,000 scale.

    The sum of squared percentage shares: 10,000 when everything is with one
    party, and 10,000 / n when n parties have equal shares.
    """
    spend = np.asarray(spend, dtype=float)
    total = spend.sum()
    return float(((spend / total * 100) ** 2).sum()) if total > 0 else 0.0


def concentration(watch: pd.DataFrame, by: str = "supplier") -> pd.DataFrame:
    """Share of spend per supplier, region, state, country or category, largest first.

    ``by`` is ``supplier`` or a column of the watchlist. ``share`` is the
    share of total annual spend and ``cumulative_share`` the running total.
    ``average_score`` is the spend-weighted overall risk score. Pass the
    ``spend`` column to ``hhi`` for the concentration index.
    """
    key = "supplier_name" if by == "supplier" else by
    if key not in watch.columns:
        raise ValueError(f"Cannot group by '{by}'")
    frame = watch.assign(_key=watch[key].where(watch[key].astype(str) != "", "(outside the US)"),
                         _weighted=watch["overall"] * watch["annual_spend_usd"])
    g = frame.groupby("_key")
    out = pd.DataFrame({"suppliers": g.size(), "spend": g["annual_spend_usd"].sum(), "on_watch": g["tier"].apply(lambda t: int((t != "Normal").sum())),
                        "_weighted": g["_weighted"].sum()})
    out["average_score"] = out["_weighted"] / out["spend"].clip(lower=1e-9)
    out = out.drop(columns="_weighted").sort_values("spend", ascending=False)
    out["share"] = out["spend"] / max(out["spend"].sum(), 1e-9)
    out["cumulative_share"] = out["share"].cumsum()
    return out.rename_axis(by).reset_index()


def probability_impact(watch: pd.DataFrame, probability: float = 0.20, impact_quantile: float = 0.75) -> pd.DataFrame:
    """Each supplier placed by how likely an incident is and how much it would matter.

    Impact is annual spend. ``quadrant`` is ``act now`` (likely and large),
    ``watch closely`` (likely, small), ``protect`` (unlikely, large: worth a
    contingency plan) or ``routine``. ``probability`` is the line between
    likely and unlikely; a supplier is large when its spend is in the top
    quarter (``impact_quantile``). Needs a watchlist built with a model.
    """
    if watch["incident_probability"].isna().all():
        raise ValueError("The watchlist has no incident probabilities; build it with a model")
    cut = float(watch["annual_spend_usd"].quantile(impact_quantile))
    likely, large = watch["incident_probability"] >= probability, watch["annual_spend_usd"] >= cut
    out = watch[["supplier_id", "supplier_name", "tier", "incident_probability", "annual_spend_usd", "sole_sourced_spend_usd",
                 "spend_at_risk", "overall"]].copy()
    out["quadrant"] = np.select([likely & large, likely, large], ["act now", "watch closely", "protect"], "routine")
    return out.sort_values("spend_at_risk", ascending=False).reset_index(drop=True)


def dependency_score(spend: float, sole_sourced_spend: float, total_spend: float, config: ScoreConfig | None = None) -> float:
    """The dependency dimension score for a supplier with the given spend figures."""
    config = config or ScoreConfig()
    parts = {m.key: m for m in MEASURES if m.dimension == "dependency"}
    share = spend / total_spend if total_spend > 0 else 0.0
    sole = min(1.0, sole_sourced_spend / spend) if spend > 0 else 0.0
    values = {"spend_share": share, "sole_source_share": sole}
    weight = sum(m.weight for m in parts.values())
    return float(sum(m.weight * scale(values[k], *config.limits(m)) for k, m in parts.items()) / weight)


def dual_source_what_if(data: SupplierData, supplier_id: str, share_moved: float = 0.5, as_of=None,
                        model: IncidentModel | None = None, config: ScoreConfig | None = None,
                        rules: TierRules | None = None) -> pd.DataFrame:
    """What qualifying a second source for a supplier's sole-sourced items would change.

    ``share_moved`` of the volume of every sole-sourced item goes to a new
    second source; those items then no longer count as sole-sourced. Returns
    one row per figure with ``before``, ``after`` and ``change``:

    ``spend with supplier``         annual spend left with the supplier
    ``sole-sourced spend``          spend on items only this supplier can provide
    ``share of total spend``        the supplier's share of all spend
    ``dependency score``            the dependency dimension (0 to 100)
    ``overall risk score``          the overall score with the new dependency score
    ``probability-weighted spend``  spend with the supplier times its incident probability
    ``portfolio HHI``               supplier concentration, with the second source counted as one
                                    new supplier

    The supplier's incident probability is left unchanged: a second source
    reduces what an incident would cost, not how likely it is. The second
    source's own risk, and the cost of qualifying it, are not modelled.
    """
    if not 0 <= share_moved <= 1:
        raise ValueError("share_moved must be between 0 and 1")
    config = config or ScoreConfig()
    watch = watchlist(data, as_of, model, config, rules, explain=False)
    row = watch[watch["supplier_id"] == supplier_id]
    if row.empty:
        raise ValueError(f"Supplier '{supplier_id}' is not active in {data.month(as_of):%B %Y}")
    row = row.iloc[0]
    total = float(watch["annual_spend_usd"].sum())
    spend, sole = float(row["annual_spend_usd"]), float(row["sole_sourced_spend_usd"])
    moved = share_moved * sole
    new_spend, new_sole = spend - moved, 0.0 if share_moved > 0 else sole
    dep_before = dependency_score(spend, sole, total, config)
    dep_after = dependency_score(new_spend, new_sole, total, config)
    weight = config.normalised_weights()["dependency"]
    p = row["incident_probability"]
    others = watch.loc[watch["supplier_id"] != supplier_id, "annual_spend_usd"].to_numpy()
    rows = [
        ("spend with supplier", spend, new_spend),
        ("sole-sourced spend", sole, new_sole),
        ("share of total spend", spend / total, new_spend / total),
        ("dependency score", dep_before, dep_after),
        ("overall risk score", float(row["overall"]), float(row["overall"]) + weight * (dep_after - dep_before)),
        ("probability-weighted spend", spend * p, new_spend * p),
        ("portfolio HHI", hhi(np.append(others, spend)), hhi(np.append(others, [new_spend, moved]))),
    ]
    out = pd.DataFrame(rows, columns=["figure", "before", "after"])
    out["change"] = out["after"] - out["before"]
    return out
