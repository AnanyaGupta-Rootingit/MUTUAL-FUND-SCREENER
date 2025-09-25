# ==============================================================================
# Mutual Fund Analysis Streamlit App (Light Version - No matplotlib/seaborn/scipy)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="Mutual Fund Analysis")
st.title("📊 Mutual Fund Analysis and Visualization App")

# -----------------------------
# File Upload
# -----------------------------
st.header("📂 Upload Data Files")
uploaded_metadata = st.file_uploader("Upload Mutual Fund Metadata CSV", type=["csv"])
uploaded_nav = st.file_uploader("Upload Mutual Fund NAV History CSV", type=["csv"])
uploaded_holdings = st.file_uploader("Upload Mutual Fund Holdings CSV (Optional)", type=["csv"])

metadata, nav_history, holdings = None, None, None
if uploaded_metadata: metadata = pd.read_csv(uploaded_metadata)
if uploaded_nav: nav_history = pd.read_csv(uploaded_nav)
if uploaded_holdings: holdings = pd.read_csv(uploaded_holdings)

# -----------------------------
# Helper Functions
# -----------------------------
def norm_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def map_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def zscore(arr):
    """Custom z-score implementation (no scipy)"""
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    return (arr - mean) / std if std > 0 else np.zeros_like(arr)

def compute_scheme_metrics(nav_df, ref_date=None):
    rf = 0.06
    results = []
    if ref_date is None: ref_date = nav_df["date"].max()
    for code in nav_df["scheme_code"].unique():
        df = nav_df[nav_df["scheme_code"]==code].sort_values("date").dropna(subset=["nav"])
        if df.empty: continue
        last_date = df["date"].iloc[-1]; last_nav = df["nav"].iloc[-1]

        def nav_on_or_before(years):
            target = last_date - pd.DateOffset(years=years)
            tmp = df[df["date"]<=target]
            if tmp.empty: return None
            return tmp["nav"].iloc[-1]

        def cagr(start, end, years):
            if start is None or start<=0 or end<=0: return np.nan
            return (end/start)**(1.0/years)-1.0

        cagr1 = cagr(nav_on_or_before(1), last_nav,1)
        cagr3 = cagr(nav_on_or_before(3), last_nav,3)
        cagr5 = cagr(nav_on_or_before(5), last_nav,5)

        df["daily_ret"] = df["nav"].pct_change()
        last_1y_cutoff = last_date - pd.DateOffset(days=365)
        rets_1y = df[df["date"]>=last_1y_cutoff]["daily_ret"].dropna()
        if len(rets_1y)<30: rets_1y=df["daily_ret"].dropna()

        if len(rets_1y)>=2:
            ann_vol = rets_1y.std()*np.sqrt(252)
            compounded = (1+rets_1y).prod() if len(rets_1y)>0 else np.nan
            ann_ret = compounded**(252.0/len(rets_1y))-1 if not pd.isna(compounded) else np.nan
            sharpe = (ann_ret-rf)/ann_vol if ann_vol>0 else np.nan
            neg = rets_1y[rets_1y<0]; downside = neg.std()*np.sqrt(252) if len(neg)>0 else np.nan
            sortino = (ann_ret-rf)/downside if downside>0 else np.nan
        else:
            ann_vol, ann_ret, sharpe, sortino = np.nan, np.nan, np.nan, np.nan

        results.append({
            "scheme_code": str(code), "last_date": last_date, "last_nav": last_nav,
            "cagr_1y": cagr1, "cagr_3y": cagr3, "cagr_5y": cagr5,
            "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino
        })
    return pd.DataFrame(results)

def compute_smart_score(df, w_return=0.6, w_risk=0.3, w_cost=0.1, return_horizon="cagr_3y"):
    df = df.copy()
    df["smart_score"] = (w_return*df.get(return_horizon,0).fillna(0)
                         - w_risk*df.get("ann_vol",0).fillna(0)
                         - w_cost*df.get("expense_ratio",0).fillna(0))
    return df.sort_values("smart_score", ascending=False)

# -----------------------------
# Data Processing
# -----------------------------
if metadata is not None and nav_history is not None:
    metadata = norm_cols(metadata)
    nav_history = norm_cols(nav_history)
    if holdings is not None: holdings = norm_cols(holdings)

    # Map columns
    meta_code = map_col(metadata, ["scheme_code","schemeid","code","id"])
    meta_name = map_col(metadata, ["scheme_name","name","fund_name"])
    nav_code = map_col(nav_history, ["scheme_code","schemeid","code","id"])
    nav_date = map_col(nav_history, ["date","nav_date"])
    nav_nav = map_col(nav_history, ["nav","nav_value","navs"])
    if not nav_code or not nav_date or not nav_nav: 
        st.error("NAV must have 'scheme_code','date','nav'"); st.stop()

    metadata = metadata.rename(columns={meta_code:"scheme_code"})
    if meta_name: metadata = metadata.rename(columns={meta_name:"scheme_name"})
    nav_history = nav_history.rename(columns={nav_code:"scheme_code", nav_date:"date", nav_nav:"nav"})
    nav_history["scheme_code"] = nav_history["scheme_code"].astype(str)
    nav_history["date"] = pd.to_datetime(nav_history["date"], errors="coerce")
    nav_history["nav"] = pd.to_numeric(nav_history["nav"], errors="coerce")

    metrics_df = compute_scheme_metrics(nav_history)
    funds = metadata.merge(metrics_df, on="scheme_code", how="left")

    st.success("✅ Data processed successfully.")
else:
    st.info("Upload Metadata and NAV CSVs to proceed.")
    st.stop()

# -----------------------------
# Basic Analysis Example
# -----------------------------
analysis_option = st.selectbox("Choose Analysis", ["Fund Performance Metrics", "Smart Scoring"])
if analysis_option=="Fund Performance Metrics":
    st.dataframe(funds[["scheme_code","scheme_name","ann_return","ann_vol","sharpe","sortino"]])
elif analysis_option=="Smart Scoring":
    scored = compute_smart_score(funds)
    st.dataframe(scored[["scheme_code","scheme_name","smart_score"]].head(10))

# -----------------------------
# Simple Plotly Chart
# -----------------------------
if st.checkbox("Show NAV Trend Example"):
    scheme = st.selectbox("Choose a Scheme", funds["scheme_name"].dropna().unique())
    code = funds[funds["scheme_name"]==scheme]["scheme_code"].iloc[0]
    df_plot = nav_history[nav_history["scheme_code"]==code]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["nav"], mode="lines", name="NAV"))
    fig.update_layout(title=f"NAV Trend for {scheme}", xaxis_title="Date", yaxis_title="NAV")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---\n**Run locally:** `streamlit run app.py`")
