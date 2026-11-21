"""Supplier records: the expected tables, validation, and a simulated supplier base with known incidents.

Four tables describe a supplier base:

``suppliers``  one row per supplier: where it is, how much is spent with it
``history``    one row per supplier and month: delivery, quality, financial and compliance measures
``items``      one row per purchased item: its annual spend and whether this supplier is the only source
``incidents``  one row per incident: a month of failed deliveries, a quality escape, or an insolvency
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .us import CENSUS_REGIONS, STATE_REGION

SUPPLIER_REQUIRED = ["supplier_id", "supplier_name", "country", "annual_spend_usd"]
SUPPLIER_OPTIONAL = ["state", "category", "single_site", "hazard_index"]
HISTORY_MEASURES = ["otif_rate", "defect_ppm", "lead_time_cv", "current_ratio", "debt_to_equity", "operating_margin", "days_payable"]
HISTORY_REQUIRED = ["supplier_id", "month"] + HISTORY_MEASURES
HISTORY_FLAGS = ["ownership_change", "audit_findings", "compliance_flag"]
ITEM_COLUMNS = ["item_id", "supplier_id", "item_name", "annual_spend_usd", "sole_sourced"]
INCIDENT_COLUMNS = ["supplier_id", "month", "incident_type"]
INCIDENT_TYPES = ["delivery_failure", "quality_escape", "insolvency"]
INTERNATIONAL = "International"

# Natural-hazard exposure by region, from 0 (none) to 1 (high). These are ILLUSTRATIVE placeholders, not
# measurements of any real hazard: replace them with your own assessment, or supply a ``hazard_index``
# column per supplier.
REGION_HAZARD = {"Northeast": 0.25, "Midwest": 0.35, "South": 0.60, "West": 0.55, INTERNATIONAL: 0.45}


class SchemaError(ValueError):
    """Raised when the supplier tables do not match the expected layout."""


@dataclass
class SupplierData:
    """The four validated tables. Build one with ``validate_data`` or ``load_data``."""

    suppliers: pd.DataFrame
    history: pd.DataFrame
    items: pd.DataFrame
    incidents: pd.DataFrame
    # Derived tables kept for reuse; the four tables above must not be changed in place.
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def months(self) -> pd.DatetimeIndex:
        """Every month with history, in order (each is the first day of the month)."""
        return pd.DatetimeIndex(np.sort(self.history["month"].unique()))

    @property
    def last_month(self) -> pd.Timestamp:
        return pd.Timestamp(self.history["month"].max())

    def month(self, as_of=None) -> pd.Timestamp:
        """The month ``as_of`` falls in (default: the latest), checked to be in the history."""
        if as_of is None:
            return self.last_month
        m = pd.Timestamp(as_of).to_period("M").to_timestamp()
        if m not in set(self.months):
            raise ValueError(f"No history for {m:%B %Y}; the data covers {self.months[0]:%B %Y} to {self.last_month:%B %Y}")
        return m

    def key(self) -> tuple:
        """A cheap fingerprint of the data, for caching."""
        h = self.history
        return (len(self.suppliers), len(h), str(h["month"].min()), str(h["month"].max()), float(h["otif_rate"].sum()),
                float(self.suppliers["annual_spend_usd"].sum()), len(self.items), len(self.incidents))


def _flag(s: pd.Series) -> pd.Series:
    """0/1 from numbers, booleans or yes/no text. Blank means 0; anything unreadable becomes missing."""
    text = s.astype(str).str.strip().str.lower()
    return text.map({"1": 1.0, "1.0": 1.0, "true": 1.0, "yes": 1.0, "y": 1.0, "0": 0.0, "0.0": 0.0, "false": 0.0, "no": 0.0,
                     "n": 0.0, "": 0.0, "nan": 0.0, "none": 0.0, "<na>": 0.0})


def _to_month(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():  # unreadable dates are reported by the caller, with a row count
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(s, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def validate_data(suppliers: pd.DataFrame, history: pd.DataFrame, items: pd.DataFrame | None = None,
                  incidents: pd.DataFrame | None = None) -> SupplierData:
    """Check the tables and return cleaned copies.

    Problems raise a ``SchemaError`` saying how many rows are affected;
    nothing is dropped silently. ``items`` and ``incidents`` may be omitted:
    without items no spend counts as sole-sourced, and without incidents the
    risk score works but the predictive model cannot be fitted.

    Adds to ``suppliers``:

    ``region``                  Census region for US suppliers, ``International`` otherwise
    ``cross_border``            1 when the supplier is outside the US
    ``hazard_index``            filled from the illustrative ``REGION_HAZARD`` table when absent
    ``items``, ``sole_sourced_items``, ``sole_sourced_spend_usd``   totals from the items table
    """
    problems = []
    for name, table, required in (("suppliers", suppliers, SUPPLIER_REQUIRED), ("history", history, HISTORY_REQUIRED)):
        missing = [c for c in required if c not in table.columns]
        if missing:
            raise SchemaError(f"Missing required column(s) in {name}: {', '.join(missing)}")

    # ---------------------------------------------------------------- suppliers
    sup = suppliers.copy()
    for c in SUPPLIER_OPTIONAL:
        if c not in sup.columns:
            sup[c] = {"state": "", "category": "uncategorised", "single_site": 0, "hazard_index": np.nan}[c]
    for c in ("supplier_id", "supplier_name", "country", "state", "category"):
        sup[c] = sup[c].fillna("").astype(str).str.strip()
    us = sup["country"].str.upper().isin(["US", "USA", "U.S.", "UNITED STATES"])
    sup["country"] = sup["country"].where(~us, "US")
    sup["state"] = sup["state"].str.upper().where(us, "")
    sup["category"] = sup["category"].where(sup["category"] != "", "uncategorised")
    sup["annual_spend_usd"] = pd.to_numeric(sup["annual_spend_usd"], errors="coerce")
    sup["hazard_index"] = pd.to_numeric(sup["hazard_index"], errors="coerce")
    single = _flag(sup["single_site"])
    if sup["supplier_id"].duplicated().any() or (sup["supplier_id"] == "").any():
        problems.append(f"{int((sup['supplier_id'].duplicated() | (sup['supplier_id'] == '')).sum())} blank or duplicate supplier_id value(s)")
    if (sup["country"] == "").any():
        problems.append(f"{int((sup['country'] == '').sum())} supplier(s) with no country")
    bad_state = us & ~sup["state"].isin(STATE_REGION)
    if bad_state.any():
        problems.append(f"{int(bad_state.sum())} US supplier(s) with a state code that is not a US state")
    if sup["annual_spend_usd"].isna().any() or (sup["annual_spend_usd"] < 0).any():
        problems.append("annual_spend_usd must be a number, zero or greater, for every supplier")
    if single.isna().any():
        problems.append(f"{int(single.isna().sum())} supplier(s) with an unreadable single_site value (use 1 or 0)")
    if ((sup["hazard_index"] < 0) | (sup["hazard_index"] > 1)).any():
        problems.append("hazard_index must be between 0 and 1")
    sup["single_site"] = single.fillna(0).astype(int)
    sup["region"] = sup["state"].map(STATE_REGION).where(us, INTERNATIONAL)
    sup["cross_border"] = (~us).astype(int)
    sup["hazard_index"] = sup["hazard_index"].fillna(sup["region"].map(REGION_HAZARD))

    # ------------------------------------------------------------------ history
    hist = history.copy()
    hist["supplier_id"] = hist["supplier_id"].astype(str).str.strip()
    hist["month"] = _to_month(hist["month"])
    for c in HISTORY_MEASURES:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    for c in HISTORY_FLAGS:
        hist[c] = pd.to_numeric(hist[c], errors="coerce").fillna(0) if c in hist.columns else 0
    if hist["month"].isna().any():
        problems.append(f"{int(hist['month'].isna().sum())} history row(s) with a missing or unreadable 'month'")
    for c in HISTORY_MEASURES:
        if hist[c].isna().any():
            problems.append(f"{int(hist[c].isna().sum())} history row(s) with missing or unreadable '{c}'")
    if hist.duplicated(["supplier_id", "month"]).any():
        problems.append(f"{int(hist.duplicated(['supplier_id', 'month']).sum())} duplicate supplier and month row(s) in history")
    unknown = ~hist["supplier_id"].isin(sup["supplier_id"])
    if unknown.any():
        problems.append(f"{int(unknown.sum())} history row(s) for a supplier_id that is not in suppliers")
    if ((hist["otif_rate"] < 0) | (hist["otif_rate"] > 1)).any():
        problems.append("otif_rate must be a share between 0 and 1")
    for c in ("defect_ppm", "lead_time_cv", "days_payable", "audit_findings"):
        if (hist[c] < 0).any():
            problems.append(f"{c} cannot be negative")
    if not hist["month"].isna().any() and not hist.duplicated(["supplier_id", "month"]).any():
        number = hist["month"].dt.year * 12 + hist["month"].dt.month
        span = number.groupby(hist["supplier_id"]).agg(["min", "max", "size"])
        gaps = int((span["max"] - span["min"] + 1 != span["size"]).sum())
        if gaps:
            problems.append(f"{gaps} supplier(s) with a gap in their monthly history; every month from first to last is needed")
    for c in ("ownership_change", "compliance_flag"):
        hist[c] = (hist[c] > 0).astype(int)

    # -------------------------------------------------------------------- items
    if items is None or len(items) == 0:
        it = pd.DataFrame({c: pd.Series(dtype=float if c == "annual_spend_usd" else object) for c in ITEM_COLUMNS})
        it["sole_sourced"] = it["sole_sourced"].astype(int)
    else:
        missing = [c for c in ("item_id", "supplier_id", "annual_spend_usd", "sole_sourced") if c not in items.columns]
        if missing:
            raise SchemaError(f"Missing required column(s) in items: {', '.join(missing)}")
        it = items.copy()
        if "item_name" not in it.columns:
            it["item_name"] = it["item_id"]
        for c in ("item_id", "supplier_id", "item_name"):
            it[c] = it[c].astype(str).str.strip()
        it["annual_spend_usd"] = pd.to_numeric(it["annual_spend_usd"], errors="coerce")
        sole = _flag(it["sole_sourced"])
        if it["annual_spend_usd"].isna().any() or (it["annual_spend_usd"] < 0).any():
            problems.append("items: annual_spend_usd must be a number, zero or greater")
        if sole.isna().any():
            problems.append(f"{int(sole.isna().sum())} item(s) with an unreadable sole_sourced value (use 1 or 0)")
        if (~it["supplier_id"].isin(sup["supplier_id"])).any():
            problems.append(f"{int((~it['supplier_id'].isin(sup['supplier_id'])).sum())} item(s) for a supplier_id that is not in suppliers")
        it["sole_sourced"] = sole.fillna(0).astype(int)
        it = it[ITEM_COLUMNS]

    # ---------------------------------------------------------------- incidents
    if incidents is None or len(incidents) == 0:
        inc = pd.DataFrame({"supplier_id": pd.Series(dtype=object), "month": pd.Series(dtype="datetime64[ns]"),
                            "incident_type": pd.Series(dtype=object)})
    else:
        missing = [c for c in INCIDENT_COLUMNS if c not in incidents.columns]
        if missing:
            raise SchemaError(f"Missing required column(s) in incidents: {', '.join(missing)}")
        inc = incidents[INCIDENT_COLUMNS].copy()
        inc["supplier_id"] = inc["supplier_id"].astype(str).str.strip()
        inc["incident_type"] = inc["incident_type"].astype(str).str.strip().str.lower().str.replace(" ", "_")
        inc["month"] = _to_month(inc["month"])
        if inc["month"].isna().any():
            problems.append(f"{int(inc['month'].isna().sum())} incident(s) with a missing or unreadable 'month'")
        if (~inc["supplier_id"].isin(sup["supplier_id"])).any():
            problems.append(f"{int((~inc['supplier_id'].isin(sup['supplier_id'])).sum())} incident(s) for a supplier_id that is not in suppliers")

    if problems:
        raise SchemaError("Invalid supplier data: " + "; ".join(problems))

    totals = it.groupby("supplier_id").agg(items=("item_id", "size"), sole_sourced_items=("sole_sourced", "sum"))
    totals["sole_sourced_spend_usd"] = (it["annual_spend_usd"] * it["sole_sourced"]).groupby(it["supplier_id"]).sum()
    for c in ("items", "sole_sourced_items", "sole_sourced_spend_usd"):
        sup[c] = sup["supplier_id"].map(totals[c]).fillna(0)
        if c != "sole_sourced_spend_usd":
            sup[c] = sup[c].astype(int)
    # Sole-sourced spend cannot exceed what is spent with the supplier.
    sup["sole_sourced_spend_usd"] = np.minimum(sup["sole_sourced_spend_usd"], sup["annual_spend_usd"])
    return SupplierData(
        suppliers=sup.sort_values("supplier_id").reset_index(drop=True),
        history=hist[HISTORY_REQUIRED + HISTORY_FLAGS].sort_values(["supplier_id", "month"]).reset_index(drop=True),
        items=it.sort_values(["supplier_id", "item_id"]).reset_index(drop=True),
        incidents=inc.sort_values(["month", "supplier_id"]).reset_index(drop=True))


_RAW_SUPPLIER_COLUMNS = ["supplier_id", "supplier_name", "category", "country", "state", "single_site", "hazard_index", "annual_spend_usd"]


def save_data(data: SupplierData, folder: str | Path) -> Path:
    """Write the four tables as CSV files (``suppliers.csv``, ``history.csv``, ``items.csv``, ``incidents.csv``)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data.suppliers[_RAW_SUPPLIER_COLUMNS].to_csv(folder / "suppliers.csv", index=False)
    for name in ("history", "incidents"):
        table = getattr(data, name)
        table.assign(month=table["month"].dt.strftime("%Y-%m")).to_csv(folder / f"{name}.csv", index=False)
    data.items.to_csv(folder / "items.csv", index=False)
    return folder


def load_data(folder: str | Path) -> SupplierData:
    """Read the tables from a folder and validate them. ``items.csv`` and ``incidents.csv`` are optional."""
    folder = Path(folder)
    for name in ("suppliers", "history"):
        if not (folder / f"{name}.csv").exists():
            raise FileNotFoundError(f"{folder / (name + '.csv')} not found")
    read = lambda name: pd.read_csv(folder / f"{name}.csv") if (folder / f"{name}.csv").exists() else None  # noqa: E731
    return validate_data(read("suppliers"), read("history"), read("items"), read("incidents"))


# --------------------------------------------------------------- sample data
_NAME_STEMS = [
    "Wobblewick", "Fizzlethorpe", "Quibbleton", "Snorkelby", "Bumblecrest", "Dapplemoor", "Gigglestone", "Noodleford",
    "Pumpernook", "Zanzibop", "Flumpton", "Crumplehorn", "Twizzlebrook", "Mumblewood", "Sprocketfen", "Dinglehop",
    "Fuddlemere", "Jibberwick", "Kerfuffle", "Lollygate", "Muddlecombe", "Nibbleshaw", "Piffleton", "Quackenbury",
    "Rumpleside", "Skedaddle", "Thimblerig", "Umberwhistle", "Vexlebury", "Whifflegate", "Yonderquill", "Zigglemont",
    "Bafflewick", "Clatterby", "Doodlemarsh", "Flibberton", "Gobbleford", "Higgledown", "Wafflehurst", "Tiddlywink",
]
_CATEGORIES = {"castings": "Castings", "fasteners": "Fasteners", "electronics": "Electronics", "plastics": "Polymers",
               "packaging": "Packaging", "machining": "Machining", "chemicals": "Chemicals", "wiring": "Harnesses"}
_STATES = {"Midwest": ["OH", "MI", "IL", "IN", "WI", "MN", "MO"], "South": ["TX", "GA", "NC", "TN", "SC", "AL", "FL", "KY"],
           "West": ["CA", "WA", "AZ", "CO", "OR", "UT"], "Northeast": ["PA", "NY", "NJ", "MA", "CT"]}
_REGION_MIX = {"Midwest": 0.32, "South": 0.30, "West": 0.20, "Northeast": 0.18}
_COUNTRIES = ["Canada", "Mexico", "Germany", "Vietnam", "India", "Japan", "Poland", "Brazil"]
# What is normal differs by category: (typical on-time-in-full rate, typical defects in PPM). Made-up values.
_CATEGORY_NORMS = {"castings": (0.905, 2600), "fasteners": (0.965, 500), "electronics": (0.975, 250), "plastics": (0.945, 1100),
                   "packaging": (0.970, 800), "machining": (0.925, 1800), "chemicals": (0.955, 400), "wiring": (0.935, 1400)}
assert all(s in CENSUS_REGIONS[r] for r, states in _STATES.items() for s in states)


def generate_sample_data(n_suppliers: int = 80, months: int = 36, start: str = "2023-07-01", seed: int = 15,
                         international_share: float = 0.2, burn_in: int = 18) -> SupplierData:
    """A simulated supplier base with monthly history and a ground-truth incident table.

    Everything here is made up, including the supplier names. Hidden rules
    decide what happens: each supplier has an unobserved fragility and an
    unobserved financial weakness; from time to time a supplier enters a
    period of operational or financial stress that builds over several
    months, pulling its delivery, quality and financial measures down
    *before* anything fails; and the chance of an incident in a month rises
    steeply with that stress. Some stress periods pass without an incident,
    and some incidents (those driven by the regional hazard index, which
    hits single-site suppliers harder) come with no warning. What counts as
    a normal delivery or defect rate differs by category and by supplier, so
    the same reading can be routine for one supplier and a warning for
    another. The score and the model never see these rules, only the
    records.

    ``burn_in`` months are simulated before the first recorded month so that
    the base starts with a usual mix of calm and stressed suppliers.

    Incidents are ``delivery_failure`` (deliveries fail for a month),
    ``quality_escape`` and ``insolvency``. An insolvent supplier's history
    stops the month before the insolvency.
    """
    rng = np.random.default_rng(seed)
    month_index = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), periods=months, freq="MS")
    categories = list(_CATEGORIES)
    suppliers, history, items, incidents, used = [], [], [], [], set()
    first_category = []
    for i in range(n_suppliers):
        sid = f"S-{i + 1:03d}"
        # A stem reused on a later pass gets a different category, so names stay unique.
        if i < len(_NAME_STEMS):
            cat = int(rng.integers(len(categories)))
            first_category.append(cat)
        else:
            cat = (first_category[i % len(_NAME_STEMS)] + int(rng.integers(1, len(categories)))) % len(categories)
        name = f"{_NAME_STEMS[i % len(_NAME_STEMS)]} {_CATEGORIES[categories[cat]]}"
        while name in used:
            name += " II"
        used.add(name)
        if rng.random() < international_share:
            country, state, region = str(rng.choice(_COUNTRIES)), "", INTERNATIONAL
        else:
            region = str(rng.choice(list(_REGION_MIX), p=list(_REGION_MIX.values())))
            country, state = "US", str(rng.choice(_STATES[region]))
        hazard = REGION_HAZARD[region]
        single = int(rng.random() < 0.45)
        spend = float(np.round(np.exp(rng.normal(13.6, 1.0)), -3))

        n_items = 1 + int(rng.poisson(4))
        split = rng.dirichlet(np.full(n_items, 1.2))
        p_sole = rng.beta(1.2, 2.5)
        item_spend = np.round(spend * split, 0)
        item_spend[0] += spend - item_spend.sum()
        for k in range(n_items):
            items.append({"item_id": f"{sid}-{k + 1:02d}", "supplier_id": sid, "item_name": f"{categories[cat]} item {k + 1}",
                          "annual_spend_usd": float(item_spend[k]), "sole_sourced": int(rng.random() < p_sole)})
        suppliers.append({"supplier_id": sid, "supplier_name": name, "category": categories[cat], "country": country, "state": state,
                          "single_site": single, "hazard_index": hazard, "annual_spend_usd": spend})

        # Hidden traits: fragility (operations) and weakness (finances). Neither is written to the tables.
        fragile, weak = rng.normal(), rng.normal()
        base_otif = float(np.clip(_CATEGORY_NORMS[categories[cat]][0] - 0.006 * fragile + rng.normal(0, 0.02), 0.84, 0.995))
        base_ppm = float(np.exp(rng.normal(np.log(_CATEGORY_NORMS[categories[cat]][1]), 0.5) + 0.2 * fragile))
        base_cv = float(np.clip(0.16 + 0.02 * fragile + rng.normal(0, 0.05), 0.06, 0.35))
        base_cr = float(np.clip(1.7 - 0.25 * weak + rng.normal(0, 0.35), 0.9, 3.0))
        base_de = float(np.clip(1.1 + 0.4 * weak + rng.normal(0, 0.45), 0.2, 3.0))
        base_om = float(np.clip(0.075 - 0.015 * weak + rng.normal(0, 0.03), -0.01, 0.18))
        base_dp = float(np.clip(55 + 6 * weak + rng.normal(0, 14), 30, 95))

        op = fin = op_rate = fin_rate = 0.0       # stress levels and how fast each is building
        op_left = fin_left = own_left = flag_left = 0
        drift = np.zeros(4)                        # slow random drift in the four financial measures
        for t in range(-burn_in, months):  # the months before 0 only let stress reach its usual mix; they are not recorded
            month = month_index[max(t, 0)]
            # Chance of each incident this month, from the stress carried into it.
            h_delivery = 0.002 * np.exp(3.2 * op + 1.0 * fin + 0.3 * fragile) + 0.002 * hazard * (1.6 if single else 0.5)
            h_quality = 0.0015 * np.exp(3.0 * op + 1.0 * (flag_left > 0) + 0.3 * fragile)
            h_insolvency = 0.0002 * np.exp(3.2 * fin)
            if rng.random() < min(h_insolvency, 0.5):
                if t >= 3:
                    incidents.append({"supplier_id": sid, "month": month, "incident_type": "insolvency"})
                    break
                fin, fin_left = 0.0, 0             # before the records start, a failed supplier is simply replaced
            delivery_failure = rng.random() < min(h_delivery, 0.6)
            quality_escape = rng.random() < min(h_quality, 0.6)

            # Stress periods start, build, and fade.
            if op_left == 0 and rng.random() < 0.040 * np.exp(0.5 * fragile) * (3.0 if own_left > 0 else 1.0):
                op_left, op_rate = int(rng.integers(6, 15)), float(rng.uniform(0.08, 0.20))
            if fin_left == 0 and rng.random() < 0.018 * np.exp(0.6 * weak):
                fin_left, fin_rate = int(rng.integers(6, 15)), float(rng.uniform(0.07, 0.18))
            if op_left > 0:
                op, op_left = op + op_rate, op_left - 1
            else:
                op *= 0.75
            if fin_left > 0:
                fin, fin_left = fin + fin_rate, fin_left - 1
            else:
                fin *= 0.85
            ownership_change = int(rng.random() < 0.006)
            own_left = 9 if ownership_change else max(0, own_left - 1)
            if flag_left == 0 and rng.random() < 0.008 + 0.05 * op:
                flag_left = int(rng.integers(3, 9))

            drift = 0.9 * drift + rng.normal(0, [0.05, 0.05, 0.006, 2.5])
            otif = base_otif - 0.12 * op - 0.03 * fin + rng.normal(0, 0.008)
            ppm = base_ppm * np.exp(0.9 * op) * rng.lognormal(0, 0.15)
            if delivery_failure:
                if t >= 0:
                    incidents.append({"supplier_id": sid, "month": month, "incident_type": "delivery_failure"})
                otif *= rng.uniform(0.3, 0.6)
                op, op_left = op * 0.5, 0          # the cause is tackled, but not all at once
            if quality_escape:
                if t >= 0:
                    incidents.append({"supplier_id": sid, "month": month, "incident_type": "quality_escape"})
                ppm *= rng.uniform(4, 10)
                op, flag_left = op * 0.7, max(flag_left, int(rng.integers(3, 7)))
            if t < 0:
                flag_left = max(0, flag_left - 1)
                continue
            history.append({
                "supplier_id": sid, "month": month,
                "otif_rate": round(float(np.clip(otif, 0.05, 1.0)), 4),
                "defect_ppm": round(float(ppm), 0),
                "lead_time_cv": round(float(max(0.03, base_cv * (1 + 0.7 * op + 0.3 * fin) + rng.normal(0, 0.010))), 3),
                "current_ratio": round(float(max(0.3, base_cr - 0.7 * fin + drift[0])), 2),
                "debt_to_equity": round(float(max(0.05, base_de + 1.2 * fin + drift[1])), 2),
                "operating_margin": round(float(base_om - 0.10 * fin - 0.02 * op + drift[2]), 4),
                "days_payable": round(float(max(15, base_dp + 35 * fin + drift[3])), 0),
                "ownership_change": ownership_change,
                "audit_findings": int(rng.poisson(0.25 + 1.8 * op + 0.2 * max(fragile, 0))),
                "compliance_flag": int(flag_left > 0),
            })
            flag_left = max(0, flag_left - 1)
    return validate_data(pd.DataFrame(suppliers), pd.DataFrame(history), pd.DataFrame(items),
                         pd.DataFrame(incidents, columns=INCIDENT_COLUMNS))
