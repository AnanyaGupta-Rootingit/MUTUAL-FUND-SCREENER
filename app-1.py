# mutual_fund_app.py
# Streamlit Mutual Fund Screener - optimized for deployment
# Requirements (put in requirements.txt):
# streamlit
# pandas
# numpy
# plotly
# scikit-learn
# scipy

import importlib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Try to import optional plotting libs; fall back if not present
_has_matplotlib = importlib.util.find_spec("matplotlib") is not None
_has_seaborn = importlib.util.find_spec("seaborn") is not None

if _has_matplotlib:
    import matplotlib.pyplot as plt
if _has_seaborn:
    import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="Mutual Fund Screener (Optimized)")

st.title("Mutual Fund Analysis & Screener — Optimized for Deployment 🚀")
st.markdown(
    "Upload **metadata** and **NAV history** CSVs. Use the buttons to run heavy computations only when ready."
)

# -------------------------
# File upload
# -------------------------
with st.sidebar:
    st.header("Upload Data")
    uploaded_meta = st.file_uploader("Upload Mutual Fund Metadata CSV", type=["csv"])
    uploaded_nav = st.file_uploader("Upload NAV History CSV", type=["csv"])
    uploaded_holdings = st.file_uploader("Upload Holdings CSV (optional)", type=["csv"])
    st.markdown("---")
    st.header("Performance Settings")
    max_years = st.number_input(
        "Limit NAV history to last N years (0 = all data)", min_value=0, max_value=20, value=3, step=1
    )
    run_heavy = st.button("Compute Metrics / Process Data")  # user triggers heavy work

# -------------------------
# Helper utils
# -------------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def map_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# Load CSVs (light)
# -------------------------
metadata = None
nav = None
holdings = None

if uploaded_meta:
    try:
        metadata = pd.read_csv(uploaded_meta)
        st.sidebar.success("Metadata loaded.")
    except Exception as e:
        st.sidebar.error(f"Error reading metadata: {e}")

if uploaded_nav:
    try:
        nav = pd.read_csv(uploaded_nav)
        st.sidebar.success("NAV history loaded.")
    except Exception as e:
        st.sidebar.error(f"Error reading NAV history: {e}")

if uploaded_holdings:
    try:
        holdings = pd.read_csv(uploaded_holdings)
        st.sidebar.success("Holdings loaded.")
    except Exception as e:
        st.sidebar.error(f"Error reading holdings: {e}")

# -------------------------
# Cached heavy compute
# -------------------------
@st.cache_data(show_spinner=False)
def preprocess_nav(nav_df: pd.DataFrame, limit_years: int = 0):
    """
    Normalize columns, convert types, optionally limit history to last N years.
    Returns cleaned nav_df
    """
    df = nav_df.copy()
    df = norm_cols(df)

    # map candidate columns
    code_col = map_col(df, ["scheme_code", "schemeid", "code", "id", "fund_code"])
    date_col = map_col(df, ["date", "nav_date", "as_of_date"])
    nav_col = map_col(df, ["nav", "nav_value", "navs", "close", "nav_price"])

    if not (code_col and date_col and nav_col):
        raise ValueError(
            "NAV file must contain columns like scheme_code (or schemeid), date (or nav_date), and nav (or nav_value)."
        )

    df = df.rename(columns={code_col: "scheme_code", date_col: "date", nav_col: "nav"})
    df["scheme_code"] = df["scheme_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

    # Drop rows without date or nav
    df = df.dropna(subset=["date", "nav"])

    # Optionally limit to last N years
    if limit_years and limit_years > 0:
        cutoff = pd.to_datetime("today") - pd.DateOffset(years=limit_years)
        df = df[df["date"] >= cutoff]

    # sort
    df = df.sort_values(["scheme_code", "date"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def compute_scheme_metrics(nav_df: pd.DataFrame, rf: float = 0.06):
    """
    For each scheme compute last_date, last_nav, cagr 1/3/5 (if available),
    ann_return, ann_vol, sharpe, sortino.
    Returns DataFrame of metrics.
    """
    results = []
    # ensure date sorted outside loop
    codes = nav_df["scheme_code"].unique()
    for code in codes:
        df = nav_df[nav_df["scheme_code"] == code].sort_values("date").reset_index(drop=True)
        if df.empty:
            continue
        last_date = df["date"].iloc[-1]
        last_nav = float(df["nav"].iloc[-1])

        def nav_on_or_before(years):
            target = last_date - pd.DateOffset(years=years)
            tmp = df[df["date"] <= target]
            if tmp.empty:
                return None
            return float(tmp["nav"].iloc[-1])

        def cagr(start, end, years):
            if start is None or start <= 0 or end <= 0:
                return np.nan
            return (end / start) ** (1.0 / years) - 1.0

        nav1 = nav_on_or_before(1)
        nav3 = nav_on_or_before(3)
        nav5 = nav_on_or_before(5)
        cagr1 = cagr(nav1, last_nav, 1) if nav1 is not None else np.nan
        cagr3 = cagr(nav3, last_nav, 3) if nav3 is not None else np.nan
        cagr5 = cagr(nav5, last_nav, 5) if nav5 is not None else np.nan

        # daily returns
        df = df.copy()
        df["daily_ret"] = df["nav"].pct_change()
        last_1y_cutoff = last_date - pd.DateOffset(days=365)
        rets_1y = df[df["date"] >= last_1y_cutoff]["daily_ret"].dropna()
        if len(rets_1y) < 30:
            # fallback to full history
            rets_1y = df["daily_ret"].dropna()

        if len(rets_1y) >= 2:
            ann_vol = rets_1y.std(ddof=1) * np.sqrt(252)
            # geometric annualized return
            compounded = (1 + rets_1y).prod() if len(rets_1y) > 0 else np.nan
            ann_ret = compounded ** (252.0 / len(rets_1y)) - 1.0 if not pd.isna(compounded) and len(rets_1y) > 0 else np.nan
            sharpe = (ann_ret - rf) / ann_vol if (not pd.isna(ann_ret) and not pd.isna(ann_vol) and ann_vol > 0) else np.nan
            neg = rets_1y[rets_1y < 0]
            if len(neg) > 0:
                downside = neg.std(ddof=1) * np.sqrt(252)
                sortino = (ann_ret - rf) / downside if downside > 0 else np.nan
            else:
                sortino = np.nan
        else:
            ann_vol = np.nan
            ann_ret = np.nan
            sharpe = np.nan
            sortino = np.nan

        results.append(
            {
                "scheme_code": str(code),
                "last_date": last_date,
                "last_nav": last_nav,
                "cagr_1y": cagr1,
                "cagr_3y": cagr3,
                "cagr_5y": cagr5,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino,
            }
        )

    return pd.DataFrame(results)


# -------------------------
# UI: show uploaded file info
# -------------------------
st.subheader("Uploaded Files")
col1, col2 = st.columns(2)
with col1:
    st.write("Metadata:", uploaded_meta.name if uploaded_meta else "Not uploaded")
    st.write("Holdings:", uploaded_holdings.name if uploaded_holdings else "Not uploaded")
with col2:
    st.write("NAV:", uploaded_nav.name if uploaded_nav else "Not uploaded")
    st.write("History limit:", f"{max_years} years" if max_years else "All history")

# -------------------------
# Run heavy processing when user clicks
# -------------------------
funds = None
metrics_df = None

if run_heavy:
    if metadata is None or nav is None:
        st.error("Please upload both Metadata and NAV CSVs before running processing.")
        st.stop()

    # Preprocess nav (cached)
    try:
        nav_clean = preprocess_nav(nav, limit_years=max_years)
    except Exception as e:
        st.error(f"Error preprocessing NAV: {e}")
        st.stop()

    # Minimal checks and normalize metadata
    meta_clean = norm_cols(metadata.copy())
    code_col_meta = map_col(meta_clean, ["scheme_code", "schemeid", "code", "id", "fund_code"])
    if code_col_meta:
        meta_clean = meta_clean.rename(columns={code_col_meta: "scheme_code"})
    if "scheme_name" not in meta_clean.columns and map_col(meta_clean, ["scheme_name", "name", "fund_name"]):
        meta_clean = meta_clean.rename(columns={map_col(meta_clean, ["scheme_name", "name", "fund_name"]): "scheme_name"})

    # compute metrics (cached)
    with st.spinner("Computing metrics (this may take a while for large datasets)..."):
        metrics_df = compute_scheme_metrics(nav_clean)

    # merge
    funds = meta_clean.merge(metrics_df, on="scheme_code", how="left")

    # add volatility column if missing (ann_vol maybe present)
    if "Volatility" not in funds.columns and "ann_vol" in funds.columns:
        funds["Volatility"] = funds["ann_vol"]

    st.success("Processing complete ✅")
    st.write(f"Funds processed: {funds.shape[0]}")

    # store cleaned nav and holdings in session_state for interaction
    st.session_state["nav_clean"] = nav_clean
    st.session_state["funds"] = funds
    if holdings is not None:
        st.session_state["holdings"] = norm_cols(holdings.copy())

# If not run yet but previously computed in session
if "funds" in st.session_state and st.session_state["funds"] is not None:
    funds = st.session_state["funds"]
    nav_clean = st.session_state.get("nav_clean", None)
    holdings = st.session_state.get("holdings", None)

# -------------------------
# If funds exist show analyses
# -------------------------
if funds is not None:
    st.header("Analysis & Visualizations")

    option = st.selectbox(
        "Choose analysis",
        [
            "Fund Performance Table",
            "Smart Scoring",
            "Recommendation (simple)",
            "NAV Performance Plot",
            "Risk vs Return (Scatter)",
            "CAGR Bar Chart (selected)",
            "Correlation Heatmap"
        ],
    )

    if option == "Fund Performance Table":
        display_cols = [c for c in ["scheme_code", "scheme_name", "ann_return", "ann_vol", "sharpe", "sortino", "expense_ratio"] if c in funds.columns]
        st.dataframe(funds[display_cols].sort_values("ann_return", ascending=False).reset_index(drop=True))

    elif option == "Smart Scoring":
        st.subheader("Smart Scoring (Return - Risk - Cost)")
        w_return = st.slider("Weight: Return", 0.0, 1.0, 0.6, 0.05)
        w_risk = st.slider("Weight: Risk", 0.0, 1.0, 0.3, 0.05)
        w_cost = st.slider("Weight: Cost", 0.0, 1.0, 0.1, 0.05)
        return_horizon = st.selectbox("Return horizon", [c for c in ["cagr_1y", "cagr_3y", "cagr_5y"] if c in funds.columns], index=0)
        def compute_score(df):
            df = df.copy()
            df["smart_score"] = (w_return * df.get(return_horizon, 0).fillna(0)
                                 - w_risk * df.get("ann_vol", 0).fillna(0)
                                 - w_cost * df.get("expense_ratio", 0).fillna(0))
            return df
        scored = compute_score(funds)
        st.dataframe(scored.sort_values("smart_score", ascending=False)[["scheme_code", "scheme_name", "smart_score"]].head(20))

    elif option == "Recommendation (simple)":
        st.subheader("Simple recommendation by risk appetite")
        risk_app = st.selectbox("Risk appetite", ["low", "medium", "high"])
        nrec = st.slider("Number of recommendations", 1, 20, 5)
        if "ann_vol" not in funds.columns:
            st.warning("ann_vol not available; cannot segment by risk.")
        else:
            if risk_app == "low":
                q = funds["ann_vol"].quantile(0.33)
                pool = funds[funds["ann_vol"] <= q]
            elif risk_app == "medium":
                q1 = funds["ann_vol"].quantile(0.33)
                q2 = funds["ann_vol"].quantile(0.66)
                pool = funds[(funds["ann_vol"] > q1) & (funds["ann_vol"] <= q2)]
            else:
                q = funds["ann_vol"].quantile(0.66)
                pool = funds[funds["ann_vol"] > q]
            pool = pool.copy()
            pool["smart_score_tmp"] = (pool.get("cagr_3y", 0).fillna(0) - pool.get("ann_vol", 0).fillna(0))
            recs = pool.sort_values("smart_score_tmp", ascending=False).head(nrec)
            st.dataframe(recs[["scheme_code", "scheme_name", "cagr_3y", "ann_vol"]])

    elif option == "NAV Performance Plot":
        st.subheader("NAV Performance Comparison (interactive)")
        schemes = st.multiselect("Select schemes (by code)", funds["scheme_code"].dropna().unique().tolist(), max_selections=6)
        if schemes:
            plot_nav = pd.concat([st.session_state["nav_clean"][st.session_state["nav_clean"]["scheme_code"] == s].assign(scheme_code=s) for s in schemes])
            fig = px.line(plot_nav, x="date", y="nav", color="scheme_code", title="NAV over time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choose 1+ schemes to plot NAVs.")

    elif option == "Risk vs Return (Scatter)":
        st.subheader("Risk vs Return (select metrics)")
        x_metric = st.selectbox("X (Risk)", ["ann_vol", "Volatility"] if "ann_vol" in funds.columns else ["ann_vol"], index=0)
        y_metric = st.selectbox("Y (Return)", [c for c in ["cagr_1y", "cagr_3y", "cagr_5y", "ann_return"] if c in funds.columns], index=0)
        fig = px.scatter(funds, x=x_metric, y=y_metric, hover_data=["scheme_name", "scheme_code"], title=f"{y_metric} vs {x_metric}")
        st.plotly_chart(fig, use_container_width=True)

    elif option == "CAGR Bar Chart (selected)":
        st.subheader("CAGR Bar Chart")
        pick = st.multiselect("Choose schemes (names)", funds["scheme_name"].dropna().unique().tolist(), max_selections=10)
        period = st.selectbox("CAGR period", [c for c in ["cagr_1y", "cagr_3y", "cagr_5y"] if c in funds.columns])
        if pick:
            subset = funds[funds["scheme_name"].isin(pick)]
            fig = px.bar(subset, x="scheme_name", y=period, title=f"{period} comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select scheme names to visualize.")

    elif option == "Correlation Heatmap":
        st.subheader("Correlation Heatmap of numeric metrics")
        candidates = [c for c in ["cagr_1y", "cagr_3y", "cagr_5y", "expense_ratio", "ann_vol", "sharpe", "sortino"] if c in funds.columns]
        if len(candidates) < 2:
            st.info("Not enough numeric metrics available for correlation.")
        else:
            corr = funds[candidates].corr()
            # Plotly heatmap
            fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("Tip: Use the sidebar `Limit NAV history` to reduce computation time on large datasets (e.g., 3 years).")

else:
    st.info("No processed funds available. Upload files and click 'Compute Metrics / Process Data' in the sidebar.")

# -------------------------
# Footer: deployment tips
# -------------------------
st.markdown(
    """
**Deployment tips**
- Add a `requirements.txt` with: streamlit, pandas, numpy, plotly, scikit-learn, scipy
- Limit NAV history in production (3 years) to reduce compute.
- For very large datasets, precompute metrics offline and load metrics CSV directly in the app.
"""
)
