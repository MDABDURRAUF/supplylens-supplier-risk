# SupplyLens Supplier Risk

**Supplier risk monitoring: which suppliers to worry about, why, whether they are getting worse, and how much
spend is exposed.**

For every supplier and every month the system keeps two things:

1. **A transparent risk score**, 0 to 100 overall and for five dimensions (operational, financial, geographic,
   dependency, compliance), where every sub-score can be explained in one sentence, plus a trend that catches a
   supplier whose score is still acceptable but rising fast.
2. **An incident probability**: the calibrated chance of a delivery failure, a quality escape or an insolvency
   in the next quarter, learned from past incidents.

Around those it produces a watchlist with the reason for each supplier, what changed since last month, spend at
risk and concentration, a probability-against-impact view, and a what-if for dual-sourcing.

The included data is simulated and the supplier names are invented. Results on it show that the method works;
they are not findings about real suppliers. A risk score is not a credit rating.

## What it does

| Question | Where | Function |
|---|---|---|
| Which suppliers need attention this month, and why? | **Watchlist** | `watchlist`, `TierRules` |
| Why does this supplier score what it scores? | **Supplier profile** | `score_as_of`, `explain_score` |
| Is it getting worse, even if the score still looks fine? | **Watchlist**, **Supplier profile** | `score_panel` (`trend`, `deteriorating`) |
| How likely is an incident next quarter, and what drives that? | **Supplier profile** | `IncidentModel.predict`, `explain` |
| What changed since last month? | **What changed** | `what_changed` |
| How much spend is exposed, and how concentrated is it? | **Portfolio** | `spend_at_risk`, `concentration`, `hhi`, `probability_impact` |
| What would a second source change? | **Portfolio** | `dual_source_what_if` |
| How accurate is the model, and against what? | **Model quality** | `evaluate.compare_models`, `calibration_table`, `lift_table`, `auc_interval` |

Outputs: the watchlist as CSV, a single-file HTML risk profile per supplier, a monthly portfolio risk report
(HTML), a saved model file, and a command line for all of it.

## Results on the simulated data

80 suppliers (66 in 25 US states, 14 abroad) with 36 months of history from July 2023 to June 2026: 2,739
monthly records, 376 purchased items of which 116 are sole-sourced, and 108 incidents (58 delivery failures,
42 quality escapes, 8 insolvencies).

The model was trained on 1,013 supplier-months up to September 2024, calibrated on 383 from January to May
2025, and tested on 511 supplier-months from September 2025 to March 2026, of which 10.4% were followed by an
incident within three months.

| Predictor | ROC AUC | Brier score |
|---|---|---|
| Overall incident rate (no model) | | 0.0930 |
| Transparent risk score | 0.785 | 0.0826 |
| Logistic regression | 0.835 | 0.0708 |
| **Gradient boosting, calibrated** | **0.842** | **0.0784** |

- Ranked by the model, the riskiest fifth of supplier-months contains 72% of those followed by an incident;
  ranked by the transparent score, 57%.
- The test period is small (53 supplier-months followed by an incident), so these figures are rough. Resampling
  whole suppliers gives a 95% range for the model's ROC AUC of about 0.73 to 0.93.
- A plain logistic regression on the same inputs does about as well as gradient boosting here, and has the
  lower Brier score. With 80 suppliers there is not enough data for the more flexible model to pull ahead.
- Calibration is imperfect on this test period: in the riskiest fifth the model predicted 23% and 37% had an
  incident. The lower four fifths are within 5 points.
- The trend flag earns its place. Among supplier-months whose score was still under the Elevated threshold,
  24% of the 182 with a fast-rising score had an incident within a quarter, against 5% of the other 1,710.
- In June 2026, 72 suppliers are active: 7 Critical and 17 Elevated, together $24.1 million of $88.9 million
  annual spend (27%), including $5.6 million on items with no second source. Four of the Elevated suppliers are
  there on the trend alone. The ten largest suppliers take 48% of spend (HHI 350).

How much do these figures depend on the simulated base? Refitting on 20 simulated bases, seeds 1 to 20
(`evaluate.seed_check(range(1, 21))`): the model's ROC AUC averaged 0.827 (lowest 0.675, highest 0.906) and
the score's 0.775 (0.665 to 0.848). The model had the higher ROC AUC in 16 of the 20, a lower Brier score than
the transparent score in 14, and a lower Brier score than no model in 18. The included base (seed 15) was
chosen because it sits close to those averages.

These numbers come from simulated data with built-in patterns, so they are easier to learn than a real
supplier base. Expect lower accuracy on real data, and measure it on your own most recent period.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-supplier-risk.git
cd supplylens-supplier-risk
pip install -e ".[app,dev]"
streamlit run app.py
```

Command line:

```bash
supplier-risk sample --out data
supplier-risk score --data data --as-of 2026-06 --out scores.csv
supplier-risk train --data data --model model.joblib
supplier-risk watchlist --data data --model model.joblib --out watchlist.csv
supplier-risk profile --data data --model model.joblib --supplier S-022 --out profile.html
supplier-risk report --data data --model model.joblib --out report.html
supplier-risk whatif --data data --model model.joblib --supplier S-022 --share 0.5
```

`--as-of YYYY-MM` replays any earlier month, `--weights operational=0.4,financial=0.3` changes the dimension
weights, and `--model` can be left out to work from the transparent score alone. Sample outputs are in
[`examples/`](examples).

Python:

```python
import supplier_risk as sr
from supplier_risk import evaluate

data = sr.load_data("data")                           # or sr.generate_sample_data()
scores = sr.score_as_of(data)                         # the transparent score, latest month
print(sr.explain_score(data, "S-022")["sentence"])    # one sentence per sub-score

model = sr.IncidentModel().fit(data)
print(evaluate.summary(model))                        # accuracy on the most recent, unseen months

watch = sr.watchlist(data, model=model)               # tier, reason, score, trend, probability, spend at risk
print(sr.watchlist_headline(watch))
print(sr.what_changed(data, model=model).head())
print(sr.spend_at_risk(watch))
print(sr.dual_source_what_if(data, "S-022", share_moved=0.5, model=model))

custom = sr.ScoreConfig(weights={"operational": 0.5, "financial": 0.5}, thresholds={"otif": (0.99, 0.90)})
print(sr.watchlist(data, as_of="2025-12", model=model, config=custom).head())
```

Only load model files you created yourself: a saved model is a Python pickle.

## Input data

A folder of CSV files. `suppliers.csv` and `history.csv` are required.

| File | Column | Required | Meaning |
|---|---|---|---|
| suppliers | `supplier_id`, `supplier_name` | yes | Identifier and display name |
| suppliers | `country`, `state` | yes | `US` with a two-letter state code, or the country name for suppliers abroad |
| suppliers | `annual_spend_usd` | yes | Annual spend with the supplier |
| suppliers | `category`, `single_site` | no | What it supplies; 1 if everything comes from one site |
| suppliers | `hazard_index` | no | Natural-hazard exposure, 0 to 1; filled from an illustrative table by region when absent |
| history | `supplier_id`, `month` | yes | One row per supplier and month, with no gaps |
| history | `otif_rate`, `defect_ppm`, `lead_time_cv` | yes | On-time-in-full share (0 to 1), defects in parts per million, lead-time standard deviation over mean |
| history | `current_ratio`, `debt_to_equity`, `operating_margin`, `days_payable` | yes | Financial indicators as last reported |
| history | `ownership_change`, `audit_findings`, `compliance_flag` | no | 1 in the month ownership changed; open audit findings; 1 while a compliance issue is open |
| items | `item_id`, `supplier_id`, `annual_spend_usd`, `sole_sourced` | yes | One row per purchased item; 1 if this supplier is the only source |
| incidents | `supplier_id`, `month`, `incident_type` | yes | `delivery_failure`, `quality_escape` or `insolvency` |

`items.csv` and `incidents.csv` are optional: without items nothing counts as sole-sourced, and without
incidents the score, trend and watchlist still work but the model cannot be fitted. Files are validated on
load; problems are reported with row counts, and nothing is dropped silently. The model needs at least 17
months of history.

## How it works

**The score.** Each measure (for example on-time-in-full averaged over the last three months) becomes a
sub-score on a straight line between a "good" value that scores 0 and a "bad" value that scores 100. A
dimension is the weighted average of its sub-scores, and the overall score is the weighted average of the five
dimensions (operational 30%, financial 25%, compliance 20%, dependency 15%, geographic 10% by default).
`explain_score` writes each sub-score as one sentence, such as "On-time-in-full averaged 91.5% over the last 3
months; 98.0% or more scores 0 and 85.0% or less scores 100, so this scores 50." The good and bad values, the
weights and the regional natural-hazard index are illustrative defaults, not measurements or industry
standards: set your own in `ScoreConfig`, or supply a `hazard_index` per supplier.

**Trend.** The least-squares slope of the overall score over the last six months, in points per month. A slope
of two points a month or more raises the deterioration flag, whatever the level of the score.

**Watchlist tiers.** Critical: score of at least 60, or incident probability of at least 35%. Elevated: score
of at least 48, probability of at least 20%, or the deterioration flag. Otherwise Normal. Every row carries a
sentence saying which rule applied. The thresholds are settings (`TierRules`).

**No leakage.** A score, a trend or a model input for a month uses only that month and earlier months. The
outcome, an incident in the next three months, is only known three months later, so the data is split by time
into training, calibration and test periods and the months just before each boundary are left out rather than
used early. A test checks that changing later records and erasing later incidents leaves every earlier score,
model input and prediction untouched.

**Incident probability.** Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) on each
supplier's recent levels, how far it is from its own usual level, compliance and ownership events, incident
history and location. Each input may only move the prediction in its common-sense direction; the data decides
by how much. The raw output is then calibrated on the calibration period with a two-parameter logistic curve,
which is steadier than a step function on a few hundred rows. Dependency measures are not given to the model:
they describe how much an incident would hurt, not how likely it is.

**The model against the score.** The score judges every supplier against the same fixed thresholds. The model
also sees each supplier against its own history, which matters when what is normal differs between suppliers,
as it does in the simulated base. The overall score also blends likelihood with exposure (dependency), so it
is not purely a predictor.

**Explanations.** Factors are groups of related inputs (delivery performance, quality, financial health and so
on). For one supplier, each factor is swapped for typical values from the training data and the change in
incident probability is recorded. Overall importance is the loss of accuracy when a factor is shuffled. Both
are measured one factor at a time and are approximate.

**Portfolio.** Spend on watch is the annual spend with Critical and Elevated suppliers. Probability-weighted
spend is annual spend times incident probability: a way to rank exposure, not a forecast of loss.
Concentration is the share of spend per supplier, region, state or category, with the Herfindahl-Hirschman
index on a 0 to 10,000 scale (10,000 means everything in one place). The probability-against-impact view
separates suppliers that are likely to have an incident from those that would hurt most.

**Dual-sourcing what-if.** A chosen share of the volume of a supplier's sole-sourced items moves to a second
source. The table shows the spend left with the supplier, the sole-sourced spend that would be stranded by a
failure, and the effect on the dependency and overall scores. The incident probability is left unchanged, and
the second source's own risk and qualification cost are not modelled.

## Limits

- The sample data is simulated, with patterns that are cleaner than a real supplier base, and the supplier
  names are invented.
- A risk score is not a credit rating. The financial thresholds are illustrative, and four ratios cannot
  establish whether a company is solvent.
- The natural-hazard index is an illustrative number per region, not a measurement of any real hazard.
- With about 80 suppliers the model's accuracy rests on a few dozen incidents. The figures are rough, the
  calibration in the highest-risk band was off on the test period, a simpler model did about as well, and all
  of it will move when the model is refitted.
- Incident types are pooled: the model predicts that some incident will happen, not which kind.
- The model learns from history and cannot foresee an event unlike anything in it. Incidents with no warning
  in the records stay unpredicted.
- Replaying an early month uses the model fitted on the whole history, so probabilities shown for months
  inside its training period look better than they would have at the time. The score and trend do not have
  this problem.
- Explanations are one-factor-at-a-time approximations and do not add up exactly to the prediction.
- Spend at risk is annual spend times probability. It ignores how long a disruption lasts, inventory and
  substitutes, so it ranks exposure and does not estimate loss.

## Tests

```bash
pytest
```

Twenty-five tests: the score, the trend and the model inputs checked against hand-worked examples, one sentence
per sub-score, no leakage from later records, time-based splitting, accuracy against the transparent score and
the no-model baseline, calibration and lift, explanations, watchlist rules, what changed, spend at risk and
concentration by hand, the dual-sourcing what-if, reports and the command line.

## Related projects

- [SupplyLens](https://github.com/MDABDURRAUF/supplylens): supply chain disruption analytics on purchase orders
- [SupplyLens Control Tower](https://github.com/MDABDURRAUF/supplylens-control-tower): track decisions and outcomes
- [SupplyLens SME Sourcing Planner](https://github.com/MDABDURRAUF/supplylens-sme-sourcing): landed-cost sourcing for small businesses
- [SupplyLens Network Simulator](https://github.com/MDABDURRAUF/supplylens-network-simulator): disruption scenarios for service and goods networks

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
