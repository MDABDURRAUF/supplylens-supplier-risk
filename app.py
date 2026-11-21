"""SupplyLens Supplier Risk dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import zipfile
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import supplier_risk as sr
from supplier_risk import evaluate as ev
from supplier_risk.report import money, pct

st.set_page_config(page_title="Supplier Risk", layout="wide")

RED, DARK, AMBER, GREEN, BLUE, GREY = "#d1495b", "#7a1f2b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1"
TIER_COLORS = {"Critical": RED, "Elevated": AMBER, "Normal": GREY}
QUADRANT_COLORS = {"act now": RED, "watch closely": AMBER, "protect": BLUE, "routine": GREY}
PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")
ONE = st.column_config.NumberColumn(format="%.1f")
SIGNED = st.column_config.NumberColumn(format="%+.1f")
MONTH = st.column_config.DateColumn(format="MMM YYYY")
SCORE = st.column_config.ProgressColumn("Risk score", format="%.0f", min_value=0, max_value=100)
PROB = st.column_config.ProgressColumn("Incident probability", format="percent", min_value=0, max_value=1)
SIMULATED = "Simulated data: the suppliers, their names and every figure are made up."


def show(fig: go.Figure, height: int = 340, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


@st.cache_resource(show_spinner="Generating a simulated supplier base...")
def sample_data() -> sr.SupplierData:
    return sr.generate_sample_data()


@st.cache_resource(max_entries=3, show_spinner="Fitting the incident model on past months...")
def fit(key: tuple, _data: sr.SupplierData):
    try:
        return sr.IncidentModel().fit(_data), ""
    except ValueError as exc:
        return None, str(exc)


@st.cache_data(max_entries=3, show_spinner=False)
def importance(key: tuple, _model: sr.IncidentModel) -> pd.DataFrame:
    return sr.factor_importance(_model)


@st.cache_data(max_entries=3, show_spinner=False)
def auc_range(key: tuple, _model: sr.IncidentModel, column: str) -> tuple:
    return ev.auc_interval(_model, column)


@st.cache_data(max_entries=3, show_spinner=False)
def probability_history(key: tuple, _model: sr.IncidentModel, _data: sr.SupplierData) -> pd.DataFrame:
    return _model.predict_panel(_data)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Supplier Risk")
st.sidebar.caption("Supplier risk monitoring")
page = st.sidebar.radio("View", ["Watchlist", "Supplier profile", "Portfolio", "What changed", "Model quality", "Data", "How it works"])

if "data" not in st.session_state:
    st.session_state.update(data=sample_data(), source="sample")
data: sr.SupplierData = st.session_state["data"]
key = data.key()
is_sample = st.session_state["source"] == "sample"
if is_sample:
    st.sidebar.info("Using a simulated supplier base. Load your own on the **Data** view.")
model, model_problem = fit(key, data)
if model is None:
    st.sidebar.warning(f"No incident model: {model_problem}. Only the transparent score is shown.")

months = list(data.months)
month_names = [f"{m:%b %Y}" for m in months]
as_of = months[month_names.index(st.sidebar.select_slider(
    "As-of month", options=month_names, value=month_names[-1],
    help="Replay history: every view shows what was known at the end of this month."))]
with st.sidebar.expander("Score weights"):
    st.caption("How much each dimension counts in the overall score. They are rescaled to sum to 100%.")
    weights = {d: st.slider(d.capitalize(), 0.0, 1.0, float(sr.DEFAULT_WEIGHTS[d]), 0.05, key=f"w_{d}") for d in sr.DIMENSIONS}
if sum(weights.values()) <= 0:
    st.sidebar.warning("All weights are zero; using the defaults.")
    weights = dict(sr.DEFAULT_WEIGHTS)
config = sr.ScoreConfig(weights=weights)
note = SIMULATED if is_sample else ""
st.sidebar.caption(f"{len(data.suppliers)} suppliers, {len(months)} months of history, {len(data.incidents)} recorded incidents. "
                   f"Latest month: {months[-1]:%b %Y}.")

watch = sr.watchlist(data, as_of, model, config)
in_sample = model is not None and model.split_.get("test_from") is not None and as_of < model.split_["test_from"]
IN_SAMPLE_NOTE = ("The incident model was fitted on months before "
                  f"{model.split_['test_from']:%B %Y}. Its probabilities for the month shown come from the period it learned or "
                  "was calibrated on, so they look better here than they would have at the time.") if in_sample else ""

# ================================================================= WATCHLIST
if page == "Watchlist":
    st.title(f"Supplier watchlist: {as_of:%B %Y}")
    if watch.empty:
        st.info("No suppliers are active in this month.")
        st.stop()
    st.info(sr.watchlist_headline(watch) + (f" {SIMULATED}" if is_sample else ""))
    s = sr.spend_at_risk(watch)
    counts = watch["tier"].value_counts()
    k = st.columns(5)
    k[0].metric("Active suppliers", len(watch))
    k[1].metric("Critical", int(counts.get("Critical", 0)))
    k[2].metric("Elevated", int(counts.get("Elevated", 0)))
    k[3].metric("Spend on watch", money(s["watch_spend"]), f"{s['watch_share']:.0%} of spend", delta_color="off")
    k[4].metric("Fast-rising scores", int(watch["deteriorating"].sum()),
                help=f"Suppliers whose overall score has risen at least {config.deterioration_slope:.1f} points a month over the last {config.trend_months} months")
    if in_sample:
        st.caption(IN_SAMPLE_NOTE)

    left, right = st.columns(2)
    with left:
        by = watch.groupby(["region", "tier"]).size().reset_index(name="suppliers")
        fig = px.bar(by, x="suppliers", y="region", color="tier", orientation="h", color_discrete_map=TIER_COLORS,
                     category_orders={"tier": sr.TIERS}, labels={"suppliers": "Suppliers", "region": ""})
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 340)
    with right:
        if model is not None:
            fig = px.scatter(watch, x="overall", y="incident_probability", size=watch["annual_spend_usd"].clip(lower=1), color="tier",
                             hover_name="supplier_name", hover_data={"annual_spend_usd": ":,.0f", "trend": ":+.1f"},
                             color_discrete_map=TIER_COLORS, category_orders={"tier": sr.TIERS},
                             labels={"overall": "Risk score (0-100)", "incident_probability": "Incident probability, next quarter"})
            fig.update_yaxes(tickformat=".0%")
        else:
            fig = px.scatter(watch, x="overall", y="trend", size=watch["annual_spend_usd"].clip(lower=1), color="tier", hover_name="supplier_name",
                             color_discrete_map=TIER_COLORS, category_orders={"tier": sr.TIERS},
                             labels={"overall": "Risk score (0-100)", "trend": "Trend, points a month"})
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 340)
    st.caption("Right: each bubble is a supplier, sized by annual spend. The score and the model usually agree; where they do not, "
               "the reason column says which one put the supplier on watch.")

    st.subheader("Watchlist")
    tiers = st.multiselect("Show", sr.TIERS, default=["Critical", "Elevated"])
    view = watch[watch["tier"].isin(tiers)] if tiers else watch
    st.caption("Click a row for the reason and the score breakdown. Scores run from 0 to 100; higher is riskier.")
    cols = ["supplier_id", "supplier_name", "tier", "overall", "trend", "incident_probability", "annual_spend_usd",
            "sole_sourced_spend_usd", "region", "reason"]
    event = st.dataframe(view[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="watch_table", column_config={
        "overall": SCORE, "trend": st.column_config.NumberColumn("Trend (pts/month)", format="%+.1f"), "incident_probability": PROB,
        "annual_spend_usd": st.column_config.NumberColumn("Annual spend", format="dollar"),
        "sole_sourced_spend_usd": st.column_config.NumberColumn("Sole-sourced spend", format="dollar")})
    if len(view):
        chosen = view.iloc[event.selection.rows[0] if event.selection.rows else 0]
        st.subheader(f"{chosen['supplier_name']} ({chosen['supplier_id']}): {chosen['tier']}")
        st.markdown(f"**Why:** {chosen['reason']}" + (f" **Model drivers:** {chosen['main_drivers']}." if chosen["main_drivers"] else ""))
        why = sr.explain_score(data, chosen["supplier_id"], as_of, config)
        top = why.sort_values("points", ascending=False).head(5)
        st.dataframe(top[["dimension", "score", "points", "sentence"]], hide_index=True, column_config={
            "score": st.column_config.NumberColumn("Sub-score", format="%.0f"),
            "points": st.column_config.NumberColumn("Adds to overall", format="%.1f"), "sentence": st.column_config.TextColumn("In one sentence", width="large")})
        st.caption("The five measures adding most to this supplier's overall score. The full breakdown is on the Supplier profile view.")
    st.subheader("Take it with you")
    d = st.columns(2)
    d[0].download_button("Portfolio risk report (HTML)", sr.build_portfolio_report(data, as_of, model, config, note=note),
                         f"portfolio_risk_report_{as_of:%Y-%m}.html", "text/html", width="stretch", type="primary")
    d[1].download_button("Watchlist (CSV)", watch.to_csv(index=False), f"watchlist_{as_of:%Y-%m}.csv", "text/csv", width="stretch")

# ========================================================== SUPPLIER PROFILE
elif page == "Supplier profile":
    st.title("Supplier profile")
    if watch.empty:
        st.info("No suppliers are active in this month.")
        st.stop()
    labels = {r["supplier_id"]: f"{r['supplier_name']} ({r['supplier_id']}, {r['tier']})" for _, r in watch.iterrows()}
    sid = st.selectbox("Supplier", list(labels), format_func=labels.get, help="Listed most worrying first")
    row = watch[watch["supplier_id"] == sid].iloc[0]
    where = f"{row['state']}, {row['region']}" if row["country"] == "US" else row["country"]
    st.info(f"**{row['supplier_name']}** ({row['category']}, {where}) is **{row['tier']}** for {as_of:%B %Y}. {row['reason']}"
            + (f" {SIMULATED}" if is_sample else ""))
    k = st.columns(5)
    k[0].metric("Risk score (0-100)", f"{row['overall']:.0f}", None if pd.isna(row["change_1m"]) else f"{row['change_1m']:+.1f} vs last month", delta_color="inverse")
    k[1].metric("Trend", "n/a" if pd.isna(row["trend"]) else f"{row['trend']:+.1f} pts/month", "fast rise" if row["deteriorating"] else None, delta_color="inverse")
    k[2].metric("Incident probability", pct(row["incident_probability"]), help="Chance of a delivery failure, quality escape or insolvency in the next quarter")
    k[3].metric("Annual spend", money(row["annual_spend_usd"]))
    k[4].metric("Sole-sourced spend", money(row["sole_sourced_spend_usd"]), f"{int(row['sole_sourced_items'])} item(s)", delta_color="off")
    if in_sample:
        st.caption(IN_SAMPLE_NOTE)

    left, right = st.columns(2)
    with left:
        st.subheader("Score by dimension")
        dims = pd.DataFrame({"dimension": sr.DIMENSIONS, "score": [row[d] for d in sr.DIMENSIONS]})
        fig = go.Figure(go.Bar(x=dims["score"], y=dims["dimension"], orientation="h",
                               marker_color=[RED if v >= 60 else AMBER if v >= 40 else BLUE for v in dims["score"]]))
        fig.update_layout(xaxis_title="Score (0-100, higher is riskier)", xaxis_range=[0, 100])
        fig.update_yaxes(autorange="reversed")
        show(fig, 300)
    with right:
        st.subheader("What drives the model's prediction")
        if model is None:
            st.info("No incident model is available for this data.")
        else:
            effects = sr.explain(model, data, as_of).loc[sid].sort_values()
            fig = go.Figure(go.Bar(x=effects.values, y=effects.index, orientation="h",
                                   marker_color=[RED if v > 0 else GREEN for v in effects.values]))
            fig.update_layout(xaxis_title="Change in incident probability against a typical supplier")
            fig.update_xaxes(tickformat="+.0%")
            show(fig, 300)
    st.caption("Right: red factors raise this supplier's risk above a typical one; green lower it. Each factor is measured on its own, "
               "so the bars do not add up exactly to the prediction.")

    st.subheader("History")
    panel = sr.score_panel(data, config)
    mine = panel[(panel["supplier_id"] == sid) & (panel["month"] <= as_of)]
    incidents = data.incidents[(data.incidents["supplier_id"] == sid) & (data.incidents["month"] <= as_of)]
    left, right = st.columns(2)
    with left:
        long = mine.melt(id_vars="month", value_vars=["overall", "operational", "financial", "compliance"], var_name="score", value_name="value")
        fig = px.line(long, x="month", y="value", color="score", labels={"month": "", "value": "Score (0-100)"},
                      color_discrete_map={"overall": DARK, "operational": BLUE, "financial": GREEN, "compliance": AMBER})
        fig.update_traces(line_width=1.5)
        fig.update_traces(selector=dict(name="overall"), line_width=3.5)
        for m in incidents["month"].unique():
            fig.add_vline(x=m, line_dash="dot", line_color=RED)
        fig.update_layout(legend=dict(orientation="h", y=-0.2), yaxis_range=[0, 100])
        show(fig, 320)
        st.caption("Dotted red lines mark months with a recorded incident. Geographic and dependency scores barely move, so they are left out.")
    with right:
        measure = st.selectbox("Record", ["otif_rate", "defect_ppm", "lead_time_cv", "current_ratio", "debt_to_equity", "operating_margin",
                                          "days_payable", "audit_findings"] + (["incident probability"] if model is not None else []))
        if measure == "incident probability":
            hist = probability_history(key, model, data)
            hist = hist[(hist["supplier_id"] == sid) & (hist["month"] <= as_of)].rename(columns={"incident_probability": measure})
        else:
            hist = data.history[(data.history["supplier_id"] == sid) & (data.history["month"] <= as_of)]
        fig = px.line(hist, x="month", y=measure, markers=True, labels={"month": "", measure: measure.replace("_", " ")})
        fig.update_traces(line_color=BLUE)
        if measure in ("otif_rate", "operating_margin", "incident probability"):
            fig.update_yaxes(tickformat=".0%")
        for m in incidents["month"].unique():
            fig.add_vline(x=m, line_dash="dot", line_color=RED)
        show(fig, 250)

    st.subheader("Why the score is what it is")
    why = sr.explain_score(data, sid, as_of, config)
    st.dataframe(why[["dimension", "value", "score", "weight", "points", "sentence"]], hide_index=True, column_config={
        "score": st.column_config.NumberColumn("Sub-score", format="%.0f"), "weight": st.column_config.NumberColumn("Weight in dimension", format="percent"),
        "points": st.column_config.NumberColumn("Adds to overall", format="%.1f"), "sentence": st.column_config.TextColumn("In one sentence", width="large")})
    st.caption("The good and bad values in each sentence are illustrative settings (see ScoreConfig), and the hazard index is an "
               "illustrative figure by region. The score is a monitoring aid, not a credit rating.")
    left, right = st.columns(2)
    with left:
        st.subheader("Items bought")
        st.dataframe(data.items[data.items["supplier_id"] == sid][["item_id", "item_name", "annual_spend_usd", "sole_sourced"]], hide_index=True,
                     column_config={"annual_spend_usd": USD, "sole_sourced": st.column_config.CheckboxColumn("Sole-sourced")})
    with right:
        st.subheader("Incidents on record")
        if incidents.empty:
            st.write("None up to this month.")
        else:
            st.dataframe(incidents.assign(incident_type=incidents["incident_type"].str.replace("_", " "))[["month", "incident_type"]],
                         hide_index=True, column_config={"month": MONTH})
    st.download_button("Supplier risk profile (HTML)", sr.build_supplier_profile(data, sid, as_of, model, config, note=note),
                       f"supplier_profile_{sid}_{as_of:%Y-%m}.html", "text/html", type="primary")

# ================================================================= PORTFOLIO
elif page == "Portfolio":
    st.title(f"Portfolio: spend at risk and concentration, {as_of:%B %Y}")
    if watch.empty:
        st.info("No suppliers are active in this month.")
        st.stop()
    s = sr.spend_at_risk(watch)
    by_supplier = sr.concentration(watch, "supplier")
    index = sr.hhi(by_supplier["spend"])
    st.info(f"{money(s['watch_spend'])} of {money(s['total_spend'])} annual spend ({s['watch_share']:.0%}) is with Critical or Elevated "
            f"suppliers, of which {money(s['sole_sourced_watch_spend'])} is on items with no second source. The ten largest suppliers "
            f"take {by_supplier['share'].head(10).sum():.0%} of spend." + (f" {SIMULATED}" if is_sample else ""))
    k = st.columns(5)
    k[0].metric("Annual spend", money(s["total_spend"]))
    k[1].metric("Spend on watch", money(s["watch_spend"]), f"{s['watch_share']:.0%} of spend", delta_color="off")
    k[2].metric("Probability-weighted spend", money(s["probability_weighted_spend"]),
                help="Sum of annual spend times incident probability. It ranks exposure; it is not a forecast of loss.")
    k[3].metric("Sole-sourced spend on watch", money(s["sole_sourced_watch_spend"]))
    k[4].metric("Supplier concentration (HHI)", f"{index:,.0f}", help="0 to 10,000. 10,000 means all spend is with one supplier; "
                f"{10000 / len(watch):,.0f} would mean it is spread evenly over these {len(watch)}.")
    if in_sample:
        st.caption(IN_SAMPLE_NOTE)

    left, right = st.columns(2)
    with left:
        st.subheader("Probability against impact")
        if model is None:
            st.info("This view needs the incident model, which could not be fitted on this data.")
        else:
            pi = sr.probability_impact(watch)
            fig = px.scatter(pi, x="annual_spend_usd", y="incident_probability", color="quadrant", size=pi["sole_sourced_spend_usd"].clip(lower=1) + pi["annual_spend_usd"] * 0.05,
                             hover_name="supplier_name", hover_data={"sole_sourced_spend_usd": ":,.0f", "spend_at_risk": ":,.0f", "tier": True},
                             color_discrete_map=QUADRANT_COLORS, category_orders={"quadrant": list(QUADRANT_COLORS)}, log_x=True,
                             labels={"annual_spend_usd": "Annual spend (log scale)", "incident_probability": "Incident probability, next quarter"})
            fig.add_hline(y=0.20, line_dash="dash", line_color=GREY)
            fig.add_vline(x=float(watch["annual_spend_usd"].quantile(0.75)), line_dash="dash", line_color=GREY)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(legend=dict(orientation="h", y=-0.25))
            show(fig, 380)
            st.caption("Bubbles grow with sole-sourced spend. Act now: likely and large. Protect: unlikely but large, worth a contingency "
                       "plan. The lines are at 20% probability and the top quarter of spend.")
    with right:
        st.subheader("US suppliers by state")
        us = watch[watch["country"] == "US"]
        metric = st.radio("Colour by", ["Annual spend", "Suppliers", "Average risk score", "Suppliers on watch"], horizontal=True)
        g = us.assign(weighted=us["overall"] * us["annual_spend_usd"]).groupby(["state", "region"])
        by_state = pd.DataFrame({"Suppliers": g.size(), "Annual spend": g["annual_spend_usd"].sum(),
                                 "Suppliers on watch": g["tier"].apply(lambda t: int((t != "Normal").sum())),
                                 "weighted": g["weighted"].sum()}).reset_index()
        by_state["Average risk score"] = by_state["weighted"] / by_state["Annual spend"].clip(lower=1e-9)
        if by_state.empty:
            st.info("No US suppliers are active in this month.")
        else:
            fig = px.choropleth(by_state, locations="state", locationmode="USA-states", scope="usa", color=metric,
                                color_continuous_scale="Reds" if metric != "Annual spend" else "Blues", hover_name="state",
                                hover_data={"region": True, "Suppliers": True, "Annual spend": ":,.0f", "Suppliers on watch": True,
                                            "Average risk score": ":.0f", "state": False})
            fig.update_layout(coloraxis_colorbar_title_text="")
            show(fig, 340)
        abroad = watch[watch["country"] != "US"]
        st.caption(f"Average risk score is weighted by spend. Not on the map: {len(abroad)} suppliers outside the US with "
                   f"{money(abroad['annual_spend_usd'].sum())} of spend ({int((abroad['tier'] != 'Normal').sum())} on watch).")

    st.subheader("Concentration")
    left, right = st.columns(2)
    with left:
        top = by_supplier.head(15).merge(watch[["supplier_name", "tier"]], left_on="supplier", right_on="supplier_name")
        fig = px.bar(top, x="share", y="supplier", color="tier", orientation="h", color_discrete_map=TIER_COLORS,
                     category_orders={"tier": sr.TIERS, "supplier": list(top["supplier"])}, labels={"share": "Share of annual spend", "supplier": ""})
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 420)
    with right:
        group = st.radio("Share of spend by", ["region", "category", "country", "state"], horizontal=True)
        by_group = sr.concentration(watch, group)
        fig = px.bar(by_group.head(15), x="share", y=group, orientation="h", color="average_score", color_continuous_scale="Reds", range_color=[0, 70],
                     labels={"share": "Share of annual spend", group: "", "average_score": "Average score"})
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        show(fig, 300)
        st.caption(f"Concentration index (HHI) by {group}: {sr.hhi(by_group['spend']):,.0f} on a 0 to 10,000 scale.")
    st.dataframe(by_group, hide_index=True, column_config={"spend": USD, "share": PCT, "cumulative_share": PCT, "average_score": ONE})

    st.subheader("What if a sole-sourced supplier had a second source?")
    sole = watch[watch["sole_sourced_spend_usd"] > 0].sort_values(["tier", "sole_sourced_spend_usd"], ascending=[True, False])
    if sole.empty:
        st.info("No active supplier has sole-sourced items in this data.")
    else:
        labels = {r["supplier_id"]: f"{r['supplier_name']} ({r['tier']}, {money(r['sole_sourced_spend_usd'])} sole-sourced)" for _, r in sole.iterrows()}
        c = st.columns([2, 1])
        sid = c[0].selectbox("Supplier", list(labels), format_func=labels.get)
        share = c[1].slider("Share of sole-sourced volume moved to the second source", 0.1, 1.0, 0.5, 0.1)
        table = sr.dual_source_what_if(data, sid, share, as_of, model, config)
        figure = table.set_index("figure")
        st.info(f"Spend that would be stranded if this supplier failed (its sole-sourced spend) falls from "
                f"**{money(figure.loc['sole-sourced spend', 'before'])}** to **{money(figure.loc['sole-sourced spend', 'after'])}**, and its "
                f"risk score from {figure.loc['overall risk score', 'before']:.0f} to {figure.loc['overall risk score', 'after']:.0f}.")
        st.dataframe(sr.what_if_text(table), hide_index=True)
        st.caption("The supplier's incident probability is left unchanged: a second source reduces what an incident would cost, not how "
                   "likely it is. The second source's own risk and the cost of qualifying it are not modelled.")
    st.download_button("Spend by supplier (CSV)", by_supplier.to_csv(index=False), f"spend_concentration_{as_of:%Y-%m}.csv", "text/csv")

# ============================================================== WHAT CHANGED
elif page == "What changed":
    st.title(f"What changed in {as_of:%B %Y}")
    if months.index(as_of) == 0:
        st.info("This is the first month of history, so there is no earlier month to compare with. Move the as-of month forward.")
        st.stop()
    changes = sr.what_changed(data, as_of, model, config)
    st.info(sr.changes_headline(changes) + (f" {SIMULATED}" if is_sample else ""))
    count = changes["move"].value_counts()
    k = st.columns(4)
    k[0].metric("Moved to a more serious tier", int(count.get("moved up", 0)))
    k[1].metric("Moved to a less serious tier", int(count.get("moved down", 0)))
    k[2].metric("New flags or incidents", int((changes["new_flags"] != "").sum()))
    k[3].metric("Left the supplier base", int(count.get("left", 0)))
    if in_sample:
        st.caption(IN_SAMPLE_NOTE)

    moved = changes[(changes["move"] != "unchanged") | (changes["new_flags"] != "")]
    left, right = st.columns(2)
    with left:
        st.subheader("Largest score moves")
        movers = changes.dropna(subset=["score_change"])
        movers = movers.reindex(movers["score_change"].abs().sort_values(ascending=False).index).head(14).sort_values("score_change")
        fig = go.Figure(go.Bar(x=movers["score_change"], y=movers["supplier_name"], orientation="h",
                               marker_color=[RED if v > 0 else GREEN for v in movers["score_change"]]))
        fig.update_layout(xaxis_title="Change in overall risk score since last month (points)")
        show(fig, 400)
    with right:
        st.subheader("Tier moves")
        flow = changes.groupby(["previous_tier", "tier"]).size().reset_index(name="suppliers")
        order = ["Critical", "Elevated", "Normal", "Not active", "Left the base"]
        grid = flow.pivot(index="previous_tier", columns="tier", values="suppliers").reindex(index=[t for t in order if t in set(flow["previous_tier"])],
                                                                                                columns=[t for t in order if t in set(flow["tier"])]).fillna(0)
        show(px.imshow(grid, text_auto=".0f", color_continuous_scale="Blues", aspect="auto", labels={"x": "This month", "y": "Last month", "color": "Suppliers"}), 400)
    st.subheader("Suppliers with a tier move or a new flag")
    if moved.empty:
        st.success("No tier moves and no new flags this month.")
    else:
        st.dataframe(moved[["supplier_id", "supplier_name", "previous_tier", "tier", "move", "overall", "score_change", "incident_probability",
                            "probability_change", "new_flags", "annual_spend_usd"]], hide_index=True, column_config={
            "overall": SCORE, "score_change": SIGNED, "incident_probability": PCT,
            "probability_change": st.column_config.NumberColumn("Probability change", format="%+.2f"),
            "annual_spend_usd": st.column_config.NumberColumn("Annual spend", format="dollar"),
            "new_flags": st.column_config.TextColumn("New this month", width="large")})
    st.download_button("What changed (CSV)", changes.to_csv(index=False), f"what_changed_{as_of:%Y-%m}.csv", "text/csv")

# ============================================================= MODEL QUALITY
elif page == "Model quality":
    st.title("How far the predictions can be trusted")
    if model is None or model.holdout_ is None:
        st.info(f"No incident model could be fitted on this data ({model_problem}). The transparent score still works on every other view.")
        st.stop()
    q, split = ev.summary(model), model.split_
    comp = ev.compare_models(model)
    lo, hi = auc_range(key, model, "p")
    lift, lift_score = ev.lift_table(model), ev.lift_table(model, column="score")
    score_auc = float(comp.set_index("model").loc[ev.MODEL_COLUMNS["p_score"], "roc_auc"])
    st.info(f"On {q['rows']:,} supplier-months the model never saw ({split['test_from']:%B %Y} to {split['last_month']:%B %Y}), it ranks "
            f"suppliers with an incident ahead above the rest with a ROC AUC of **{q['roc_auc']:.3f}**, against {score_auc:.3f} for the "
            f"transparent score alone. The test period holds only {int(q['rows'] * q['incident_rate'])} supplier-months followed by an "
            f"incident, so the figure is rough: a 95% range of about {lo:.2f} to {hi:.2f}." + (f" {SIMULATED}" if is_sample else ""))
    st.caption(f"Trained on {split['train_rows']:,} supplier-months up to {split['train_until']:%b %Y}; calibrated on {split['calibration_rows']:,} "
               f"from {split['calibration_from']:%b %Y} to {split['calibration_until']:%b %Y}; tested from {split['test_from']:%b %Y}. The outcome "
               f"is an incident in the following {model.horizon} months, so the months just before each boundary are left out rather than used early.")
    k = st.columns(4)
    k[0].metric("ROC AUC", f"{q['roc_auc']:.3f}", f"{q['roc_auc'] - score_auc:+.3f} vs the score", help="0.5 is guessing, 1.0 is perfect ranking")
    k[1].metric("Error removed", pct(q["skill"]), help="Share of the Brier score of quoting the average incident rate for everyone that the model removes")
    k[2].metric("Incidents found in the riskiest fifth", pct(lift["cumulative_capture"].iloc[0]),
                f"{(lift['cumulative_capture'].iloc[0] - lift_score['cumulative_capture'].iloc[0]) * 100:+.0f} pts vs the score")
    k[3].metric("Incident rate, test period", pct(q["incident_rate"], 1), help="Share of supplier-months followed by an incident within the horizon")

    left, right = st.columns(2)
    with left:
        st.subheader("Against simpler alternatives")
        fig = px.bar(comp, x="brier_score", y="model", orientation="h", text_auto=".4f", color_discrete_sequence=[BLUE],
                     labels={"brier_score": "Brier score (lower is better)", "model": ""})
        fig.update_yaxes(categoryorder="total descending")
        show(fig, 300)
        st.dataframe(comp, hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in comp.columns[1:]})
    with right:
        st.subheader("Calibration")
        cal = ev.calibration_table(model)
        top = float(max(cal["predicted"].max(), cal["observed"].max()) * 1.1)
        fig = px.scatter(cal, x="predicted", y="observed", color_discrete_sequence=[BLUE],
                         labels={"predicted": "Predicted incident rate", "observed": "Observed incident rate"})
        fig.update_traces(marker_size=12)
        fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(dash="dash", color=GREY), showlegend=False)
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        show(fig, 300)
        worst = cal.loc[cal["gap"].abs().idxmax()]
        st.caption(f"Each point is a fifth of the test supplier-months. On the dashed line, a predicted 30% really is 30%. The largest gap "
                   f"is {worst['gap'] * 100:+.0f} points (predicted {worst['predicted']:.0%}, observed {worst['observed']:.0%}).")

    left, right = st.columns(2)
    with left:
        st.subheader("Where the incidents are")
        both = pd.concat([lift.assign(ranked_by="Model"), lift_score.assign(ranked_by="Transparent score")])
        fig = px.bar(both, x="group", y="incident_rate", color="ranked_by", barmode="group", color_discrete_map={"Model": BLUE, "Transparent score": GREY},
                     labels={"group": "Fifth of supplier-months, riskiest first", "incident_rate": "Followed by an incident"})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(legend=dict(orientation="h", y=-0.3))
        show(fig, 300)
    with right:
        st.subheader("What the model relies on")
        imp = importance(key, model)
        fig = px.bar(imp, x="share", y="factor", orientation="h", color_discrete_sequence=[BLUE], labels={"share": "Share of predictive power", "factor": ""})
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_xaxes(tickformat=".0%")
        show(fig, 300)
    st.subheader("Predicted against observed, by region")
    st.dataframe(ev.by_segment(model, "region"), hide_index=True, column_config={"observed": PCT, "predicted": PCT,
                                                                                 "gap": st.column_config.NumberColumn(format="%+.3f")})
    st.caption("With a supplier base of this size every figure on this page rests on a few dozen incidents. Treat differences of a few "
               "points between predictors, or between regions, as noise, and refit as more months arrive.")
    st.download_button("Held-out predictions (CSV)", model.holdout_.to_csv(index=False), "holdout_predictions.csv", "text/csv")

# ====================================================================== DATA
elif page == "Data":
    st.title("Data")
    st.caption("Four CSV files. Suppliers and history are required; items adds sole-sourcing; incidents is needed for the incident model.")
    c = st.columns(4)
    uploads = {name: c[i].file_uploader(f"{name}.csv", type="csv", key=f"up_{name}") for i, name in enumerate(["suppliers", "history", "items", "incidents"])}
    c = st.columns(3)
    if c[0].button("Use these files", type="primary", disabled=uploads["suppliers"] is None or uploads["history"] is None):
        try:
            tables = {name: pd.read_csv(BytesIO(f.getvalue())) if f is not None else None for name, f in uploads.items()}
            loaded = sr.validate_data(tables["suppliers"], tables["history"], tables["items"], tables["incidents"])
        except (sr.SchemaError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not use these files: {exc}")
        else:
            st.session_state.update(data=loaded, source="yours")
            st.rerun()
    if c[1].button("Reset to simulated data"):
        st.session_state.update(data=sample_data(), source="sample")
        st.rerun()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("suppliers.csv", data.suppliers[["supplier_id", "supplier_name", "category", "country", "state", "single_site", "hazard_index",
                                                    "annual_spend_usd"]].to_csv(index=False))
        z.writestr("history.csv", data.history.assign(month=data.history["month"].dt.strftime("%Y-%m")).to_csv(index=False))
        z.writestr("items.csv", data.items.to_csv(index=False))
        z.writestr("incidents.csv", data.incidents.assign(month=data.incidents["month"].dt.strftime("%Y-%m")).to_csv(index=False))
    c[2].download_button("Download current data (ZIP of CSV files)", buffer.getvalue(), "supplier_data.zip", "application/zip")

    st.subheader("Columns")
    st.dataframe(pd.DataFrame([
        ("suppliers", "supplier_id", "yes", "Unique identifier"),
        ("suppliers", "supplier_name", "yes", "Name, for display"),
        ("suppliers", "country", "yes", "US, or the country name for suppliers abroad"),
        ("suppliers", "state", "for US suppliers", "Two-letter US state code; gives the Census region"),
        ("suppliers", "annual_spend_usd", "yes", "Annual spend with the supplier"),
        ("suppliers", "category", "no", "What the supplier provides"),
        ("suppliers", "single_site", "no", "1 if everything comes from one site, else 0"),
        ("suppliers", "hazard_index", "no", "Natural-hazard exposure, 0 to 1; filled from an illustrative table by region when absent"),
        ("history", "supplier_id, month", "yes", "One row per supplier and month (YYYY-MM), with no gaps"),
        ("history", "otif_rate", "yes", "Share of orders delivered on time and in full, 0 to 1"),
        ("history", "defect_ppm", "yes", "Defective parts per million received"),
        ("history", "lead_time_cv", "yes", "Lead-time variability: standard deviation of lead time over its mean"),
        ("history", "current_ratio, debt_to_equity, operating_margin, days_payable", "yes", "Financial indicators as last reported"),
        ("history", "ownership_change", "no", "1 in the month ownership changed"),
        ("history", "audit_findings", "no", "Open audit findings in the month"),
        ("history", "compliance_flag", "no", "1 while a compliance issue is open"),
        ("items", "item_id, supplier_id, annual_spend_usd", "yes", "One row per purchased item"),
        ("items", "sole_sourced", "yes", "1 if this supplier is the only source for the item"),
        ("items", "item_name", "no", "For display"),
        ("incidents", "supplier_id, month, incident_type", "yes", "One row per incident: delivery_failure, quality_escape or insolvency"),
    ], columns=["file", "column", "required", "meaning"]), hide_index=True)

    st.subheader("Current data")
    if is_sample:
        st.caption(SIMULATED)
    k = st.columns(5)
    k[0].metric("Suppliers", len(data.suppliers))
    k[1].metric("Outside the US", int(data.suppliers["cross_border"].sum()))
    k[2].metric("Months of history", len(months))
    k[3].metric("Recorded incidents", len(data.incidents))
    k[4].metric("Period", f"{months[0]:%b %Y} to {months[-1]:%b %Y}")
    if len(data.incidents):
        monthly = data.incidents.assign(incident_type=data.incidents["incident_type"].str.replace("_", " ")).groupby(
            ["month", "incident_type"]).size().reset_index(name="incidents")
        fig = px.bar(monthly, x="month", y="incidents", color="incident_type", labels={"month": "", "incidents": "Incidents recorded"},
                     color_discrete_sequence=[BLUE, AMBER, DARK])
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        show(fig, 280)
    tabs = st.tabs(["suppliers", "history", "items", "incidents"])
    tabs[0].dataframe(data.suppliers, hide_index=True, height=300, column_config={"annual_spend_usd": USD, "sole_sourced_spend_usd": USD})
    tabs[1].dataframe(data.history.tail(500), hide_index=True, height=300, column_config={"month": MONTH})
    tabs[2].dataframe(data.items, hide_index=True, height=300, column_config={"annual_spend_usd": USD})
    tabs[3].dataframe(data.incidents, hide_index=True, height=300, column_config={"month": MONTH})

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    st.markdown(
        """
**Two views of every supplier**

1. *A transparent risk score*, 0 to 100, higher is riskier. Five dimensions (operational, financial, geographic, dependency,
   compliance), each a weighted average of a few sub-scores. Every sub-score is a straight line between a "good" value that scores 0
   and a "bad" value that scores 100, so each can be explained in one sentence. You set the weights in the sidebar.
2. *An incident probability*: the chance of a delivery failure, a quality escape or an insolvency in the next quarter, learned from
   past incidents by a gradient-boosted model and calibrated so that 30% means about three in ten.

**Trend.** The score's slope over the last six months is measured in points per month. A supplier whose score is rising by two
points a month or more gets a deterioration flag and goes on the watchlist even if its score is still acceptable.

**Watchlist tiers.** Critical: score of 60 or more, or incident probability of 35% or more. Elevated: score of 48 or more,
probability of 20% or more, or a deterioration flag. Otherwise Normal. The reason column says which rule applied.

**No peeking.** A score or a prediction for a month uses only that month and earlier months. The model is trained on older months,
calibrated on newer ones and tested on the newest. Because "an incident in the next quarter" is only known three months later, the
months just before each boundary are left out instead of being used early.

**The model against the score.** The score judges every supplier against the same fixed thresholds. The model also sees how far a
supplier is from *its own* usual level, which matters when what is normal differs between suppliers. Each input may only push the
prediction in the direction common sense says (worse delivery can only raise risk); the data decides by how much.

**Explanations** of the model are measured by experiment: swap one factor for typical values and see how the prediction moves.

**Portfolio.** Spend on watch is the annual spend with Critical and Elevated suppliers. Probability-weighted spend is spend times
incident probability: a way to rank exposure, not a forecast of loss. Concentration is the share of spend per supplier or region
and the Herfindahl-Hirschman index (10,000 means everything in one place). The dual-sourcing what-if moves part of a supplier's
sole-sourced volume to a second source and shows how much less spend would be stranded.

**Limits.** The sample data is simulated and the supplier names are invented. The thresholds, the weights and the regional
natural-hazard index are illustrative settings, not measurements: replace them with your own. A risk score is not a credit rating.
With around 80 suppliers the model's accuracy rests on a few dozen incidents, so its figures are rough and will move when it is
refitted; the Model quality view shows the range. The model cannot foresee an event unlike anything in its history, and
dependency measures how much an incident would hurt, not how likely it is.
"""
    )
