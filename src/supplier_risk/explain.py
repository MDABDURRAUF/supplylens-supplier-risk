"""Why the model says what it says: which factors matter overall, and for one supplier.

Gradient-boosted trees have no simple formula, so explanations here are
measured by experiment: change a factor and see how the prediction moves.
Factors are groups of related inputs (see ``features.FACTOR_GROUPS``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import SupplierData
from .features import FACTOR_GROUPS, build_features
from .model import IncidentModel


def _groups(model: IncidentModel) -> dict[str, list[str]]:
    groups = {name: [c for c in cols if c in model.features_] for name, cols in FACTOR_GROUPS.items()}
    return {name: cols for name, cols in groups.items() if cols}


def factor_importance(model: IncidentModel, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """How much each factor contributes to telling suppliers with an incident ahead from the rest.

    On the held-out period, each factor's columns are shuffled across rows
    (breaking their link to the outcome) and the drop in ROC AUC is
    recorded, averaged over ``repeats`` shuffles. A larger drop means the
    model leans on that factor more. Factors that overlap in what they know
    can share credit, so read this as a ranking, not an exact split.
    """
    if model.holdout_ is None:
        raise RuntimeError("The model has no held-out period")
    X, y = model.holdout_features_, model.holdout_["incident"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.proba_from_features(X))
    rows = []
    for name, cols in _groups(model).items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - roc_auc_score(y, model.proba_from_features(shuffled)))
        rows.append({"factor": name, "auc_drop": float(np.mean(drops)), "inputs": ", ".join(cols)})
    out = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    out["share"] = out["auc_drop"].clip(lower=0) / max(out["auc_drop"].clip(lower=0).sum(), 1e-12)
    return out


def explain_features(model: IncidentModel, X: pd.DataFrame) -> pd.DataFrame:
    """Factor effects for already-built feature rows: one row per supplier, one column per factor.

    For each factor, the supplier's values are replaced by those of typical
    supplier-months (a sample of the training data) and the incident
    probability is recomputed. The effect is the actual prediction minus the
    average of those: positive means this factor makes the supplier riskier
    than a typical one, negative safer. Effects are measured one factor at a
    time and do not add up exactly to the prediction.
    """
    background = model.background_
    groups = _groups(model)
    actual = model.proba_from_features(X)
    out = pd.DataFrame(index=X.index, columns=list(groups), dtype=float)
    b = len(background)
    for name, cols in groups.items():
        # Every supplier paired with every background row, taking only this factor from the background.
        tiled = X.loc[X.index.repeat(b), model.features_].reset_index(drop=True)
        tiled[cols] = np.tile(background[cols].to_numpy(), (len(X), 1))
        typical = model.proba_from_features(tiled).reshape(len(X), b).mean(axis=1)
        out[name] = actual - typical
    return out


def explain(model: IncidentModel, data: SupplierData, as_of=None) -> pd.DataFrame:
    """Factor effects for every supplier active in ``as_of``, indexed by ``supplier_id`` (see ``explain_features``)."""
    month = data.month(as_of)
    rows = build_features(data)
    rows = rows[rows["month"] == month].set_index("supplier_id")
    return explain_features(model, rows[model.features_])


def top_reasons(effects: pd.DataFrame, n: int = 2, minimum: float = 0.02) -> pd.Series:
    """The ``n`` factors raising each supplier's risk the most, as short text.

    Factors adding less than ``minimum`` to the incident probability are
    left out; a supplier with none gets "no single factor stands out".
    """
    def describe(row: pd.Series) -> str:
        top = row[row >= minimum].sort_values(ascending=False).head(n)
        return ", ".join(f"{name} (+{value * 100:.0f} pts)" for name, value in top.items()) or "no single factor stands out"

    return effects.apply(describe, axis=1)
