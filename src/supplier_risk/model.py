"""The incident model: the chance a supplier has an incident in the next quarter.

Gradient-boosted trees give a raw score for each supplier and month, which
is then calibrated so that a predicted 30% means about three in ten such
suppliers have an incident within the following three months. Each input is
only allowed to move the prediction in its common-sense direction (see
``features.DIRECTION``), which keeps a model trained on a few dozen incidents
from learning something backwards.

All splits are by time. The outcome for a month is known only three months
later, so a month is used for training only if its whole outcome quarter
ended before the calibration period starts, and for calibration only if its
outcome quarter ended before the test period starts. The months in between
are left out rather than used early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SupplierData
from .features import DIRECTION, FEATURES, build_features, incident_labels
from .score import ScoreConfig, score_panel


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class IncidentModel:
    """Fit with ``fit(data)``, then ``predict(data)``.

    ``horizon``          months ahead the incident is predicted over (3 = the next quarter)
    ``train_fraction``   share of the months with a known outcome that the training period spans
    ``calib_fraction``   share that the calibration period spans
    ``min_history``      a supplier needs this many months of history before its months are used

    The remaining months are the held-out test period; its predictions are
    kept in ``holdout_`` for every check in ``evaluate``.
    """

    horizon: int = 3
    train_fraction: float = 0.52
    calib_fraction: float = 0.26
    min_history: int = 3
    random_state: int = 0
    features_: list = field(default_factory=list, repr=False)
    prior_: float = 0.1
    holdout_: pd.DataFrame | None = field(default=None, repr=False)
    holdout_features_: pd.DataFrame | None = field(default=None, repr=False)
    background_: pd.DataFrame | None = field(default=None, repr=False)
    split_: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(self, data: SupplierData) -> "IncidentModel":
        if not 0 < self.train_fraction < 1 or not 0 < self.calib_fraction < 1 or self.train_fraction + self.calib_fraction > 1:
            raise ValueError("train_fraction and calib_fraction must be positive and sum to at most 1")
        if data.incidents.empty:
            raise ValueError("The incidents table is empty; the model needs past incidents to learn from")
        rows = build_features(data)
        rows["incident"] = incident_labels(data, rows, self.horizon)
        score = score_panel(data, ScoreConfig())[["supplier_id", "month", "overall"]]
        rows = rows.merge(score, on=["supplier_id", "month"], how="left")
        rows = rows[(rows["history_months"] >= self.min_history) & rows["incident"].notna()].reset_index(drop=True)
        months = np.sort(rows["month"].unique())
        if len(months) < 12:
            raise ValueError("At least 12 months with a known outcome are needed to fit the model "
                             f"(that is {12 + self.horizon + self.min_history - 1} months of history)")
        has_test = self.train_fraction + self.calib_fraction < 1
        calib_start = pd.Timestamp(months[min(len(months) - 1, int(round(self.train_fraction * len(months))))])
        test_start = pd.Timestamp(months[min(len(months) - 1, int(round((self.train_fraction + self.calib_fraction) * len(months))))])
        known_by = rows["month"] + pd.DateOffset(months=self.horizon)  # the last month of the outcome quarter
        train = known_by < calib_start
        if has_test:
            calib = (rows["month"] >= calib_start) & (known_by < test_start)
            test = rows["month"] >= test_start
        else:
            calib, test = rows["month"] >= calib_start, pd.Series(False, index=rows.index)
        y = rows["incident"].astype(int)
        if y[train].nunique() < 2 or y[calib].nunique() < 2:
            raise ValueError("The training and calibration periods must each contain suppliers with and without an incident")

        self.features_ = list(FEATURES)
        X = rows[self.features_]
        self.prior_ = float(y[train].mean())
        self.classifier_ = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=25, l2_regularization=2.0,
            monotonic_cst=[DIRECTION[f] for f in self.features_], random_state=self.random_state).fit(X[train], y[train])
        # Two-number (Platt) calibration: with a few hundred calibration rows it is steadier than a step function.
        self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(self.classifier_.predict_proba(X[calib])[:, 1]), y[calib])
        self.logistic_ = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                   ("model", LogisticRegression(C=0.3, max_iter=2000))]).fit(X[train], y[train])
        # The transparent score turned into a probability: one logistic curve fitted on the training period.
        self.score_curve_ = LogisticRegression(C=1e6, max_iter=1000).fit(rows.loc[train, ["overall"]].to_numpy(), y[train])

        self.background_ = X[train].sample(min(80, int(train.sum())), random_state=self.random_state)
        self.split_ = {"train_rows": int(train.sum()), "calibration_rows": int(calib.sum()), "test_rows": int(test.sum()),
                       "first_month": pd.Timestamp(months[0]), "train_until": pd.Timestamp(rows.loc[train, "month"].max()),
                       "calibration_from": calib_start, "calibration_until": pd.Timestamp(rows.loc[calib, "month"].max()),
                       "test_from": test_start if has_test else None, "last_month": pd.Timestamp(months[-1]),
                       "train_incident_rate": self.prior_, "horizon": self.horizon}
        if has_test and test.sum():
            h = rows.loc[test, ["supplier_id", "month", "incident", "overall"]].rename(columns={"overall": "score"})
            h = h.merge(data.suppliers[["supplier_id", "supplier_name", "region", "category"]], on="supplier_id", how="left")
            Xt = X[test]
            h["p_constant"] = self.prior_
            h["p_score"] = self.score_curve_.predict_proba(h[["score"]].to_numpy())[:, 1]
            h["p_logistic"] = self.logistic_.predict_proba(Xt)[:, 1]
            h["p_boosted_raw"] = self.classifier_.predict_proba(Xt)[:, 1]
            h["p"] = self.proba_from_features(Xt)
            self.holdout_, self.holdout_features_ = h.reset_index(drop=True), Xt.reset_index(drop=True)
        else:
            self.holdout_, self.holdout_features_ = None, None
        return self

    # -------------------------------------------------------------- predict
    def _check(self) -> None:
        if not self.features_:
            raise RuntimeError("Call fit() first")

    def proba_from_features(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated incident probability for an already-built feature table."""
        self._check()
        return self.calibrator_.predict_proba(_logit(self.classifier_.predict_proba(X[self.features_])[:, 1]))[:, 1]

    def predict_panel(self, data: SupplierData) -> pd.DataFrame:
        """``incident_probability`` for every supplier and month in the data.

        Months inside the model's own training period are scored too, which
        is useful for replaying history but flatters the model there: only
        the months from ``split_['test_from']`` on are a fair test.
        """
        rows = build_features(data)
        out = rows[["supplier_id", "month"]].copy()
        out["incident_probability"] = self.proba_from_features(rows)
        return out

    def predict(self, data: SupplierData, as_of=None) -> pd.DataFrame:
        """Probability that each active supplier has an incident in the ``horizon`` months after ``as_of``."""
        month = data.month(as_of)
        panel = self.predict_panel(data)
        return panel[panel["month"] == month].reset_index(drop=True)

    # ----------------------------------------------------------------- files
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "IncidentModel":
        model = joblib.load(path)
        if not isinstance(model, IncidentModel):
            raise ValueError("The file does not contain an IncidentModel")
        return model
