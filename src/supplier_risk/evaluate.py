"""How far the predictions can be trusted, measured on months the model never saw.

Everything here reads ``model.holdout_``: one row per supplier and month in
the most recent period, after the model's training and calibration data
ended. The outcome is whether the supplier had an incident in the three
months that followed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .data import SupplierData, generate_sample_data
from .features import incident_labels
from .model import IncidentModel
from .score import ScoreConfig, score_panel

MODEL_COLUMNS = {
    "p_constant": "Overall incident rate (no model)",
    "p_score": "Transparent risk score",
    "p_logistic": "Logistic regression",
    "p_boosted_raw": "Gradient boosting",
    "p": "Gradient boosting, calibrated",
}


def _holdout(model: IncidentModel) -> pd.DataFrame:
    if model.holdout_ is None:
        raise RuntimeError("The model has no held-out period; fit it with train_fraction + calib_fraction below 1")
    return model.holdout_


def _scores(y, p) -> dict:
    y, p = np.asarray(y, dtype=int), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    both = len(np.unique(y)) == 2
    return {"roc_auc": roc_auc_score(y, p) if both and len(np.unique(p)) > 1 else np.nan,
            "average_precision": average_precision_score(y, p) if both else np.nan,
            "brier_score": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1])}


def summary(model: IncidentModel) -> dict:
    """Headline accuracy of the calibrated model on the held-out period.

    ``rows`` counts supplier-months; ``incident_rate`` is the share followed
    by an incident within the horizon; ``skill`` is the share of the
    no-model baseline's Brier score that the model removes.
    """
    h = _holdout(model)
    out = {"rows": len(h), "suppliers": int(h["supplier_id"].nunique()), "months": int(h["month"].nunique()),
           "incident_rate": float(h["incident"].mean()), **_scores(h["incident"], h["p"])}
    out["brier_baseline"] = float(brier_score_loss(h["incident"], h["p_constant"]))
    out["skill"] = 1 - out["brier_score"] / out["brier_baseline"]
    return out


def compare_models(model: IncidentModel) -> pd.DataFrame:
    """The model against simpler alternatives on the same held-out supplier-months.

    Higher is better for ``roc_auc`` and ``average_precision``; lower is
    better for ``brier_score`` and ``log_loss``. The no-model row quotes the
    training period's incident rate for everyone. The transparent-score row
    ranks suppliers by the overall risk score and converts it to a
    probability with one curve fitted on the training period.
    """
    h = _holdout(model)
    return pd.DataFrame([{"model": label, **_scores(h["incident"], h[col])} for col, label in MODEL_COLUMNS.items()])


def calibration_table(model: IncidentModel, bins: int = 5, column: str = "p") -> pd.DataFrame:
    """Predicted against observed incident rate, in equal-sized bands of predicted risk."""
    h = _holdout(model)
    band = pd.qcut(h[column].rank(method="first"), q=bins, labels=False)
    g = h.groupby(band)
    out = pd.DataFrame({"rows": g.size(), "predicted": g[column].mean(), "observed": g["incident"].mean()}).reset_index(drop=True)
    out["gap"] = out["observed"] - out["predicted"]
    return out


def lift_table(model: IncidentModel, bins: int = 5, column: str = "p") -> pd.DataFrame:
    """How concentrated incidents are among the highest-risk predictions.

    Supplier-months are ranked by ``column`` (``p`` for the model, ``score``
    for the transparent score) and cut into ``bins`` equal groups, riskiest
    first. ``cumulative_capture`` is the share of all incident-quarters
    found by working down to that group; ``lift`` is the group's incident
    rate over the average.
    """
    h = _holdout(model).sort_values(column, ascending=False, kind="stable").reset_index(drop=True)
    group = (np.arange(len(h)) * bins // len(h)) + 1
    g = h.groupby(group)
    out = pd.DataFrame({"group": g.size().index, "rows": g.size().to_numpy(), "incidents": g["incident"].sum().to_numpy(),
                        "incident_rate": g["incident"].mean().to_numpy()})
    out["lift"] = out["incident_rate"] / h["incident"].mean()
    out["cumulative_capture"] = out["incidents"].cumsum() / h["incident"].sum()
    out["cumulative_share_reviewed"] = out["rows"].cumsum() / len(h)
    return out


def auc_interval(model: IncidentModel, column: str = "p", repeats: int = 400, seed: int = 0) -> tuple[float, float]:
    """A rough 95% range for the ROC AUC on the held-out period.

    Whole suppliers are resampled with replacement, because one supplier's
    consecutive months are not independent observations. A wide range means
    the held-out period is too small to pin the figure down.
    """
    h = _holdout(model)
    rng = np.random.default_rng(seed)
    groups = [g[["incident", column]].to_numpy() for _, g in h.groupby("supplier_id")]
    values = []
    for _ in range(repeats):
        sample = np.vstack([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if 0 < sample[:, 0].sum() < len(sample):
            values.append(roc_auc_score(sample[:, 0].astype(int), sample[:, 1]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def by_segment(model: IncidentModel, column: str = "region") -> pd.DataFrame:
    """Predicted against observed incident rate for each region or category in the held-out period."""
    h = _holdout(model)
    g = h.groupby(column)
    out = pd.DataFrame({"rows": g.size(), "observed": g["incident"].mean(), "predicted": g["p"].mean()})
    out["gap"] = out["observed"] - out["predicted"]
    return out.sort_values("rows", ascending=False).reset_index()


def trend_flag_check(data: SupplierData, config: ScoreConfig | None = None, horizon: int = 3, below: float = 48.0) -> dict:
    """Does the deterioration flag catch suppliers whose score still looks acceptable?

    Takes every supplier-month with a known outcome whose overall score is
    under ``below`` (the default Elevated threshold) and compares how often
    an incident followed within ``horizon`` months, with and without the
    deterioration flag. The score involves no fitting, so every month counts.
    """
    panel = score_panel(data, config)
    panel["incident"] = incident_labels(data, panel, horizon)
    ok = panel[panel["incident"].notna() & (panel["overall"] < below) & panel["trend"].notna()]
    flagged, rest = ok[ok["deteriorating"]], ok[~ok["deteriorating"]]
    return {"flagged_rows": len(flagged), "flagged_incident_rate": float(flagged["incident"].mean()) if len(flagged) else float("nan"),
            "other_rows": len(rest), "other_incident_rate": float(rest["incident"].mean()) if len(rest) else float("nan")}


def seed_check(seeds=range(1, 11), **sample_options) -> pd.DataFrame:
    """Refit on freshly simulated supplier bases to see how much the headline figures move.

    One row per seed: the held-out ROC AUC and Brier score of the calibrated
    model, the transparent score and the no-model baseline. With about 80
    suppliers the held-out period holds only a few dozen incidents, so the
    figures vary from one simulated base to the next; this shows by how much.
    Applies to the simulated data only.
    """
    rows = []
    for seed in seeds:
        table = compare_models(IncidentModel().fit(generate_sample_data(seed=seed, **sample_options))).set_index("model")
        model, score, none = (table.loc[MODEL_COLUMNS[c]] for c in ("p", "p_score", "p_constant"))
        rows.append({"seed": seed, "auc_model": model["roc_auc"], "auc_score": score["roc_auc"], "brier_model": model["brier_score"],
                     "brier_score_only": score["brier_score"], "brier_no_model": none["brier_score"]})
    return pd.DataFrame(rows)
