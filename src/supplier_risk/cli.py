"""Command line.

    supplier-risk sample --out data
    supplier-risk score --data data --as-of 2026-06 --out scores.csv
    supplier-risk train --data data --model model.joblib
    supplier-risk watchlist --data data --model model.joblib --out watchlist.csv
    supplier-risk profile --data data --model model.joblib --supplier S-017 --out profile.html
    supplier-risk report --data data --model model.joblib --out report.html
    supplier-risk whatif --data data --model model.joblib --supplier S-017 --share 0.5
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__, evaluate
from .data import SchemaError, generate_sample_data, load_data, save_data
from .model import IncidentModel
from .monitor import watchlist, watchlist_headline
from .portfolio import dual_source_what_if
from .report import build_portfolio_report, build_supplier_profile, what_if_text, write_html
from .score import DIMENSIONS, ScoreConfig, score_as_of


def _show(df: pd.DataFrame, formatters: dict | None = None) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 120):
        print(df.to_string(index=False, formatters=formatters or {}))


def _weights(text: str | None) -> ScoreConfig:
    """``operational=0.4,financial=0.3`` -> a ScoreConfig with those dimension weights (the rest keep their defaults)."""
    config = ScoreConfig()
    if not text:
        return config
    weights = dict(config.weights)
    for part in text.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in DIMENSIONS:
            raise ValueError(f"'{name.strip()}' is not a dimension; choose from {', '.join(DIMENSIONS)}")
        weights[name.strip()] = float(value)
    config = ScoreConfig(weights=weights)
    config.normalised_weights()
    return config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="supplier-risk", description="Supplier risk monitoring.")
    p.add_argument("--version", action="version", version=f"supplier-risk {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="write a simulated supplier base to a folder of CSV files")
    s.add_argument("--out", default="data", help="folder for suppliers.csv, history.csv, items.csv and incidents.csv")
    s.add_argument("--seed", type=int, default=15)

    def common(parser, model: bool = True):
        parser.add_argument("--data", required=True, help="folder with suppliers.csv and history.csv (items.csv, incidents.csv optional)")
        parser.add_argument("--as-of", help="month to report on, YYYY-MM (default: the latest month in the data)")
        parser.add_argument("--weights", help="dimension weights, for example operational=0.4,financial=0.3")
        if model:
            parser.add_argument("--model", help="a fitted model file; without it only the transparent score is used")

    sc = sub.add_parser("score", help="the transparent risk score for every supplier")
    common(sc, model=False)
    sc.add_argument("--out", help="write the scores to this CSV file")

    t = sub.add_parser("train", help="fit the incident model and report how it did on the most recent months")
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="model.joblib", help="where to save the fitted model")

    w = sub.add_parser("watchlist", help="suppliers by tier, with the reason for each")
    common(w)
    w.add_argument("--out", help="write the watchlist to this CSV file")

    pr = sub.add_parser("profile", help="write one supplier's risk profile as an HTML page")
    common(pr)
    pr.add_argument("--supplier", required=True, help="supplier_id")
    pr.add_argument("--out", default="profile.html")
    pr.add_argument("--note", default="", help="a line to print under the title, for example 'Simulated data'")

    r = sub.add_parser("report", help="write the monthly portfolio risk report as an HTML page")
    common(r)
    r.add_argument("--out", default="report.html")
    r.add_argument("--note", default="")

    wi = sub.add_parser("whatif", help="what a second source for a supplier's sole-sourced items would change")
    common(wi)
    wi.add_argument("--supplier", required=True)
    wi.add_argument("--share", type=float, default=0.5, help="share of the sole-sourced volume moved to the second source")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pc, one = "{:.1%}".format, "{:.1f}".format
    try:
        if args.command == "sample":
            data = generate_sample_data(seed=args.seed)
            save_data(data, args.out)
            print(f"Wrote {len(data.suppliers)} simulated suppliers, {len(data.history):,} monthly records, {len(data.items)} items and "
                  f"{len(data.incidents)} incidents to {args.out}. Not real data.")
            return 0
        data = load_data(args.data)
        if args.command == "train":
            model = IncidentModel().fit(data)
            s, split = evaluate.summary(model), model.split_
            lo, hi = evaluate.auc_interval(model)
            print(f"Tested on {s['rows']:,} supplier-months from {split['test_from']:%B %Y} on, after training ended "
                  f"({s['incident_rate']:.1%} were followed by an incident within {model.horizon} months).")
            _show(evaluate.compare_models(model), {"roc_auc": "{:.3f}".format, "average_precision": "{:.3f}".format,
                                                   "brier_score": "{:.4f}".format, "log_loss": "{:.4f}".format})
            lift = evaluate.lift_table(model)
            print(f"\nROC AUC {s['roc_auc']:.3f}, rough 95% range {lo:.2f} to {hi:.2f}. The riskiest fifth of supplier-months holds "
                  f"{lift['cumulative_capture'].iloc[0]:.0%} of those followed by an incident.")
            print(f"Saved model to {model.save(args.model)}")
            return 0
        config = _weights(args.weights)
        month = data.month(args.as_of)
        if args.command == "score":
            scores = score_as_of(data, month, config)
            print(f"Risk scores for {month:%B %Y}: {len(scores)} active suppliers, {int(scores['deteriorating'].sum())} with a fast-rising score.\n")
            _show(scores.head(15)[["supplier_id", "supplier_name", "overall"] + DIMENSIONS + ["trend", "deteriorating"]],
                  {c: one for c in ["overall", "trend"] + DIMENSIONS})
            if args.out:
                scores.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
            return 0
        model = IncidentModel.load(args.model) if args.model else None
        if args.command == "watchlist":
            watch = watchlist(data, month, model, config)
            print(watchlist_headline(watch) + "\n")
            _show(watch[watch["tier"] != "Normal"][["supplier_id", "supplier_name", "tier", "overall", "trend", "incident_probability", "reason"]],
                  {"overall": one, "trend": "{:+.1f}".format, "incident_probability": pc})
            if args.out:
                watch.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
        elif args.command == "profile":
            print(f"Wrote {write_html(build_supplier_profile(data, args.supplier, month, model, config, note=args.note), args.out)}")
        elif args.command == "report":
            print(f"Wrote {write_html(build_portfolio_report(data, month, model, config, note=args.note), args.out)}")
        elif args.command == "whatif":
            table = dual_source_what_if(data, args.supplier, args.share, month, model, config)
            name = data.suppliers.set_index("supplier_id").loc[args.supplier, "supplier_name"]
            print(f"{name}: a second source takes {args.share:.0%} of the sole-sourced volume ({month:%B %Y}).\n")
            _show(what_if_text(table))
    except (SchemaError, ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
