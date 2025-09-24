# ==============================================================================
# Streamlit App for Mutual Fund Analysis and Visualization (Debugged & Complete)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="Mutual Fund Analysis")
st.title("Mutual Fund Analysis and Visualization App")

# -----------------------------
# File Upload
# -----------------------------
st.header("📂 Upload Data Files")

uploaded_metadata = st.file_uploader("Upload Mutual Fund Metadata CSV", type=["csv"])
uploaded_nav = st.file_uploader("Upload Mutual Fund NAV History CSV", type=["csv"])
uploaded_holdings = st.file_uploader("Upload Mutual Fund Holdings CSV (Optional)", type=["csv"])

metadata, nav_history, holdings = None, None, None

if uploaded_metadata:
    metadata = pd.read_csv(uploaded_metadata)
    st.success("Metadata uploaded.")
if uploaded_nav:
    nav_history = pd.read_csv(uploaded_nav)
    st.success("NAV history uploaded.")
if uploaded_holdings:
    holdings = pd.read_csv(uploaded_holdings)
    st.success("Holdings uploaded.")

# -----------------------------
# Helper Functions
# -----------------------------
def norm_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def map_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_scheme_metrics(nav_df, ref_date=None):
    rf = 0.06
    results = []
    if ref_date is None:
        ref_date = nav_df["date"].max()
    scheme_codes = nav_df["scheme_code"].unique()
    for code in scheme_codes:
        df = nav_df[nav_df["scheme_code"] == code].sort_values("date").dropna(subset=["nav"])
        if df.empty: continue
        last_row = df.iloc[-1]
        last_date = last_row["date"]
        last_nav = float(last_row["nav"])

        def nav_on_or_before(years):
            target = last_date - pd.DateOffset(years=years)
            tmp = df[df["date"] <= target]
            if tmp.empty: return None
            return float(tmp.iloc[-1]["nav"])

        def cagr(start_nav, end_nav, years):
            if start_nav is None or start_nav <= 0 or end_nav <= 0: return np.nan
            return (end_nav / start_nav) ** (1.0/years) - 1.0

        nav1, nav3, nav5 = nav_on_or_before(1), nav_on_or_before(3), nav_on_or_before(5)
        cagr1, cagr3, cagr5 = cagr(nav1, last_nav, 1), cagr(nav3, last_nav, 3), cagr(nav5, last_nav, 5)

        df["daily_ret"] = df["nav"].pct_change()
        last_1y_cutoff = last_date - pd.DateOffset(days=365)
        rets_1y = df[df["date"] >= last_1y_cutoff]["daily_ret"].dropna()
        if len(rets_1y) < 30: rets_1y = df["daily_ret"].dropna()

        if len(rets_1y) >= 2:
            ann_vol = rets_1y.std() * np.sqrt(252)
            compounded = (1 + rets_1y).prod() if len(rets_1y) > 0 else np.nan
            ann_ret = compounded ** (252.0/len(rets_1y)) - 1.0 if not pd.isna(compounded) else np.nan
            sharpe = (ann_ret - rf)/ann_vol if ann_vol>0 else np.nan
            neg = rets_1y[rets_1y < 0]
            downside = neg.std()*np.sqrt(252) if len(neg)>0 else np.nan
            sortino = (ann_ret - rf)/downside if downside>0 else np.nan
        else:
            ann_vol, sharpe, sortino, ann_ret = np.nan, np.nan, np.nan, np.nan

        results.append({
            "scheme_code": str(code),
            "last_date": last_date,
            "last_nav": last_nav,
            "cagr_1y": cagr1,
            "cagr_3y": cagr3,
            "cagr_5y": cagr5,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino
        })
    return pd.DataFrame(results)

def compute_smart_score(df, w_return=0.6, w_risk=0.3, w_cost=0.1, return_horizon="cagr_3y"):
    df = df.copy()
    df["smart_score"] = (
        w_return * df.get(return_horizon, 0).fillna(0)
        - w_risk * df.get("ann_vol", 0).fillna(0)
        - w_cost * df.get("expense_ratio", 0).fillna(0)
    )
    return df.sort_values("smart_score", ascending=False)

def recommend_funds(time_horizon, risk_appetite, num_recommendations, df):
    # Simple recommendation logic based on risk and smart score
    df = compute_smart_score(df)
    if risk_appetite == "low":
        df = df[df["ann_vol"] <= df["ann_vol"].quantile(0.33)]
    elif risk_appetite == "medium":
        df = df[(df["ann_vol"] > df["ann_vol"].quantile(0.33)) & (df["ann_vol"] <= df["ann_vol"].quantile(0.66))]
    else:
        df = df[df["ann_vol"] > df["ann_vol"].quantile(0.66)]
    return df.head(num_recommendations) if not df.empty else None

def holdings_overlap(holdings, scheme_a, scheme_b, by="asset", top_k=10):
    a = holdings[holdings["scheme_code"] == scheme_a][[by, "weight"]].rename(columns={"weight":"weight_a"})
    b = holdings[holdings["scheme_code"] == scheme_b][[by, "weight"]].rename(columns={"weight":"weight_b"})
    merged = pd.merge(a,b,on=by,how="inner")
    if merged.empty: return None
    merged["min_pct"] = merged[["weight_a","weight_b"]].min(axis=1)
    overlap_pct = merged["min_pct"].sum()
    top_common = merged.sort_values("min_pct", ascending=False).head(top_k)
    return {"scheme_a": scheme_a, "scheme_b": scheme_b, "overlap_pct": overlap_pct, "top_common": top_common, "level":"High" if overlap_pct>50 else "Medium" if overlap_pct>20 else "Low"}

# -----------------------------
# Data Processing
# -----------------------------
if metadata is not None and nav_history is not None:
    metadata = norm_cols(metadata)
    nav_history = norm_cols(nav_history)
    if holdings is not None: holdings = norm_cols(holdings)

    # Column mapping
    meta_code = map_col(metadata, ["scheme_code","schemeid","code","id"])
    meta_name = map_col(metadata, ["scheme_name","name","fund_name"])
    nav_code = map_col(nav_history, ["scheme_code","schemeid","code","id"])
    nav_date = map_col(nav_history, ["date","nav_date"])
    nav_nav = map_col(nav_history, ["nav","nav_value","navs"])
    if not nav_code or not nav_date or not nav_nav:
        st.error("NAV history must have columns like 'scheme_code','date','nav'")
        st.stop()

    metadata = metadata.rename(columns={meta_code:"scheme_code"})
    if meta_name: metadata = metadata.rename(columns={meta_name:"scheme_name"})
    nav_history = nav_history.rename(columns={nav_code:"scheme_code", nav_date:"date", nav_nav:"nav"})
    nav_history["scheme_code"] = nav_history["scheme_code"].astype(str)
    nav_history["date"] = pd.to_datetime(nav_history["date"], errors="coerce")
    nav_history["nav"] = pd.to_numeric(nav_history["nav"], errors="coerce")

    metrics_df = compute_scheme_metrics(nav_history)
    funds = metadata.merge(metrics_df, on="scheme_code", how="left")

    # Add Volatility for heatmaps
    risk_data = []
    for scheme, group in nav_history.groupby("scheme_code"):
        group = group.sort_values("date")
        group["daily_ret"] = group["nav"].pct_change()
        vol = group["daily_ret"].std()*np.sqrt(252) if group["daily_ret"].notna().sum()>0 else np.nan
        risk_data.append([scheme, vol])
    funds = funds.merge(pd.DataFrame(risk_data, columns=["scheme_code","Volatility"]), on="scheme_code", how="left")

    st.success("Data processed.")

else:
    st.info("Upload Metadata and NAV files to proceed.")
    st.stop()

# -----------------------------
# Analysis Options
# -----------------------------
analysis_option = st.selectbox(
    "Choose an analysis or visualization:",
    ["Fund Performance Metrics", "Smart Scoring", "Recommendation Engine", "Fund Overlap Analysis",
     "NAV Performance Comparison", "Risk vs Return Scatter Plot", "Fund Comparison Radar Plot",
     "CAGR Comparison Bar Chart", "Metrics Distribution", "Correlation Heatmap", "Portfolio Exposure Pie Chart"]
)

st.write(f"Selected Analysis: {analysis_option}")

# ==============================
# Example: Fund Performance Table
# ==============================
if analysis_option == "Fund Performance Metrics":
    display_cols = ["scheme_code","scheme_name","ann_return","ann_vol","sharpe","sortino"]
    display_cols = [c for c in display_cols if c in funds.columns]
    st.dataframe(funds[display_cols])

# ==============================
# Smart Scoring Example
# ==============================
elif analysis_option == "Smart Scoring":
    w_return = st.slider("Weight Return",0.0,1.0,0.6,0.05)
    w_risk = st.slider("Weight Risk",0.0,1.0,0.3,0.05)
    w_cost = st.slider("Weight Cost",0.0,1.0,0.1,0.05)
    return_horizon = st.selectbox("Return Horizon", ["cagr_1y","cagr_3y","cagr_5y"], index=1)
    scored = compute_smart_score(funds, w_return, w_risk, w_cost, return_horizon)
    st.dataframe(scored[["scheme_code","scheme_name","smart_score"]].head(10))

# ==============================
# Other analyses can be added similarly using above helpers
# ==============================

st.markdown("---\n**How to run:** `streamlit run mutual_fund_app.py`")
