# This cell contains the core logic for a Streamlit app, combining elements from previous steps.
# You can copy and paste this code into a Python file (e.g., app.py) and deploy it with Streamlit.

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# --- Helper function to normalize column names ---
def norm_cols(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# --- Helper function to map potential column names ---
def map_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- Helper function to compute metrics ---
def compute_scheme_metrics(nav_df, ref_date=None):
    rf = 0.06  # annual risk free
    results = []
    if ref_date is None:
        ref_date = nav_df["date"].max()
    scheme_codes = nav_df["scheme_code"].unique()
    for code in scheme_codes:
        df = nav_df[nav_df["scheme_code"] == code].sort_values("date").dropna(subset=["nav"])
        if df.empty:
            continue
        last_row = df.iloc[-1]
        last_date = last_row["date"]
        last_nav = float(last_row["nav"])

        def nav_on_or_before(years):
            target = last_date - pd.DateOffset(years=years)
            tmp = df[df["date"] <= target]
            if tmp.empty:
                return None
            return float(tmp.iloc[-1]["nav"])

        def cagr(start_nav, end_nav, years):
            if start_nav is None or start_nav <= 0 or end_nav <= 0:
                return np.nan
            try:
                return (end_nav / start_nav) ** (1.0/years) - 1.0
            except:
                return np.nan

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
            rets_1y = df["daily_ret"].dropna()  # fallback

        if len(rets_1y) >= 2:
            ann_vol = rets_1y.std(ddof=1) * np.sqrt(252)
            # geometric annualized return
            compounded = (1 + rets_1y).prod() if len(rets_1y)>0 else np.nan
            ann_ret = compounded ** (252.0/len(rets_1y)) - 1.0 if not pd.isna(compounded) and len(rets_1y)>0 else np.nan
            # sharpe
            sharpe = (ann_ret - rf) / ann_vol if (not pd.isna(ann_ret) and not pd.isna(ann_vol) and ann_vol>0) else np.nan
            # sortino
            neg = rets_1y[rets_1y < 0]
            if len(neg) > 0:
                downside = neg.std(ddof=1) * np.sqrt(252)
                sortino = (ann_ret - rf) / downside if downside > 0 else np.nan
            else:
                sortino = np.nan
        else:
            ann_vol = np.nan
            sharpe = np.nan
            sortino = np.nan
            ann_ret = np.nan

        results.append({
            "scheme_code": str(code),
            "last_date": last_date,
            "last_nav": last_nav,
            "cagr_1y": cagr1, "cagr_3y": cagr3, "cagr_5y": cagr5,
            "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino
        })

    return pd.DataFrame(results)

# --- Smart Scoring Model ---
def compute_smart_score(df, w_return=0.6, w_risk=0.3, w_cost=0.1, return_horizon="cagr_3y"):
    df = df.copy().reset_index(drop=True)
    if return_horizon not in df.columns:
        raise ValueError("Return horizon not in df: " + return_horizon)

    df["ret_metric"] = pd.to_numeric(df[return_horizon], errors="coerce")
    df["risk_metric"] = pd.to_numeric(df["sharpe"] if "sharpe" in df.columns else df.get("sortino", None), errors="coerce")
    df["cost_metric"] = pd.to_numeric(df.get("expense_ratio", np.nan), errors="coerce")

    for col in ["ret_metric","risk_metric","cost_metric"]:
        if col not in df or df[col].dropna().empty:
            df[col + "_z"] = np.nan
        else:
            arr = df[col].fillna(df[col].median()).values.astype(float)
            if np.nanstd(arr) == 0:
                df[col + "_z"] = 0.0
            else:
                df[col + "_z"] = zscore(arr)

    df["cost_metric_z_inv"] = -1 * df["cost_metric_z"]

    total = float(w_return + w_risk + w_cost)
    if total == 0:
        raise ValueError("Sum of weights must be > 0")
    w_return_n, w_risk_n, w_cost_n = w_return/total, w_risk/total, w_cost/total

    df["smart_score"] = (w_return_n * df["ret_metric_z"].fillna(0) +
                         w_risk_n * df["risk_metric_z"].fillna(0) +
                         w_cost_n * df["cost_metric_z_inv"].fillna(0))

    out_cols = ["scheme_code","scheme_name","amc","category","expense_ratio",
                "cagr_1y","cagr_3y","cagr_5y","sharpe","sortino","smart_score"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df.sort_values("smart_score", ascending=False).reset_index(drop=True)

# --- Fund Recommendation Engine ---
mapping_time_risk = {
    ("short","low"): ["Debt","Liquid","Ultra Short Duration","Low Duration","Gilt"],
    ("short","medium"): ["Debt","Hybrid"],
    ("short","high"): ["Hybrid","Arbitrage","Conservative Hybrid"],
    ("medium","low"): ["Hybrid","Conservative Hybrid","Debt"],
    ("medium","medium"): ["Hybrid","Balanced","Large Cap","Index"],
    ("medium","high"): ["Large Cap","Multi Cap","Flexi Cap"],
    ("long","low"): ["Large Cap","Index","Balanced"],
    ("long","medium"): ["Large Cap","Multi Cap","Flexi Cap"],
    ("long","high"): ["Small Cap","Mid Cap","Sectoral","Thematic"]
}

def recommend_funds(funds_df, time_horizon="medium", risk_appetite="medium", top_n=10,
                    w_return=0.6, w_risk=0.3, w_cost=0.1, return_horizon="cagr_3y"):
    time_horizon = time_horizon.lower()
    risk_appetite = risk_appetite.lower()
    key = (time_horizon, risk_appetite)
    categories = mapping_time_risk.get(key, None)
    if categories is None:
        st.warning("Profile not recognized. Options for time_horizon: short/medium/long; risk_appetite: low/medium/high")
        return None

    scored = compute_smart_score(funds_df, w_return=w_return, w_risk=w_risk, w_cost=w_cost, return_horizon=return_horizon)
    mask = scored["category"].fillna("").apply(lambda x: any(cat.lower() in x.lower() for cat in categories))
    selected = scored[mask].copy()
    if selected.empty:
        selected = scored[scored["category"].fillna("").str.lower().isin([c.lower() for c in categories])]
    final = selected.sort_values("smart_score", ascending=False).head(top_n)
    return final[["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","cagr_5y","sharpe","sortino","smart_score"]]

# --- Fund Overlap Analysis ---
def holdings_overlap(holdings_df, code_a, code_b, by="asset", top_k=20):
    if holdings_df is None:
        st.warning("Holdings dataset not provided.")
        return None
    h = holdings_df.copy()
    if "scheme_code" not in h.columns or by not in h.columns or "weight" not in h.columns:
        st.warning("Holdings file must contain 'scheme_code', '{}', and 'weight' columns.".format(by))
        return None

    a = h[h["scheme_code"].astype(str) == str(code_a)].copy()
    b = h[h["scheme_code"].astype(str) == str(code_b)].copy()
    if a.empty or b.empty:
        st.info("Holdings missing for one/both schemes.")
        return None

    def normalize_weights(df):
        df = df.copy()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        if df["weight"].sum() > 1.5:
            df["weight"] = df["weight"] / 100.0
        return df

    a = normalize_weights(a)
    b = normalize_weights(b)

    a_agg = a.groupby(by)["weight"].sum().reset_index()
    b_agg = b.groupby(by)["weight"].sum().reset_index()

    merged = a_agg.merge(b_agg, on=by, how="outer", suffixes=("_a","_b")).fillna(0.0)
    merged["min_w"] = merged[["weight_a","weight_b"]].min(axis=1)

    overlap_fraction = merged["min_w"].sum()
    overlap_pct = overlap_fraction * 100.0

    if overlap_pct < 20:
        level = "Low"
    elif overlap_pct < 40:
        level = "Medium"
    else:
        level = "High"

    top_common = merged.sort_values("min_w", ascending=False).head(top_k)[[by,"weight_a","weight_b","min_w"]]
    top_common["weight_a_pct"] = top_common["weight_a"]*100
    top_common["weight_b_pct"] = top_common["weight_b"]*100
    top_common["min_pct"] = top_common["min_w"]*100

    return {
        "scheme_a": code_a,
        "scheme_b": code_b,
        "overlap_pct": overlap_pct,
        "level": level,
        "top_common": top_common
    }

# --- Streamlit App Structure ---
st.title("Mutual Fund Analysis and Recommendation")

st.sidebar.header("Upload Data Files")
uploaded_metadata = st.sidebar.file_uploader("Upload Mutual Fund Metadata CSV", type="csv")
uploaded_nav_history = st.sidebar.file_uploader("Upload Mutual Fund NAV History CSV", type="csv")
uploaded_holdings = st.sidebar.file_uploader("Upload Mutual Fund Holdings CSV (Optional)", type="csv")

metadata = None
nav_history = None
holdings = None

if uploaded_metadata is not None:
    metadata = pd.read_csv(uploaded_metadata)
    metadata = norm_cols(metadata)
    meta_code_col = map_col(metadata, ["scheme_code","schemeid","code","id"])
    meta_name_col = map_col(metadata, ["scheme_name","name","fund_name"])
    meta_amc_col = map_col(metadata, ["amc","amc_name","fund_house"])
    meta_cat_col = map_col(metadata, ["category","fund_category","type"])
    meta_exp_col = map_col(metadata, ["expense_ratio","ter","expense"])
    meta_mininv_col = map_col(metadata, ["minimum_investment","min_investment","min_inv"])
    metadata = metadata.rename(columns={c: "scheme_code" for c in [meta_code_col] if c})
    if meta_name_col: metadata = metadata.rename(columns={meta_name_col: "scheme_name"})
    if meta_amc_col: metadata = metadata.rename(columns={meta_amc_col: "amc"})
    if meta_cat_col: metadata = metadata.rename(columns={meta_cat_col: "category"})
    if meta_exp_col: metadata = metadata.rename(columns={meta_exp_col: "expense_ratio"})
    if meta_mininv_col: metadata = metadata.rename(columns={meta_mininv_col: "minimum_investment"})
    metadata["scheme_code"] = metadata["scheme_code"].astype(str)
    if "expense_ratio" in metadata.columns:
        metadata["expense_ratio"] = pd.to_numeric(metadata["expense_ratio"], errors="coerce")


if uploaded_nav_history is not None:
    nav_history = pd.read_csv(uploaded_nav_history)
    nav_history = norm_cols(nav_history)
    nav_code_col = map_col(nav_history, ["scheme_code","schemeid","code","id"])
    nav_date_col = map_col(nav_history, ["date","nav_date"])
    nav_nav_col = map_col(nav_history, ["nav","nav_value","navs"])
    if nav_code_col and nav_date_col and nav_nav_col:
        nav_history = nav_history.rename(columns={nav_code_col: "scheme_code", nav_date_col: "date", nav_nav_col: "nav"})
        nav_history["scheme_code"] = nav_history["scheme_code"].astype(str)
        nav_history["date"] = pd.to_datetime(nav_history["date"], errors="coerce")
        nav_history["nav"] = pd.to_numeric(nav_history["nav"], errors="coerce")
    else:
        st.error("NAV history file must contain columns like 'scheme_code','date','nav' (case-insensitive).")
        nav_history = None

if uploaded_holdings is not None:
    holdings = pd.read_csv(uploaded_holdings)
    holdings = norm_cols(holdings)
    holdings_code = map_col(holdings, ["scheme_code","schemeid","code","fund_code"])
    holdings_asset = map_col(holdings, ["stock","holding","security","name","ticker","sector"])
    holdings_weight = map_col(holdings, ["weight","percentage","pct","weight_pct","holding_percent"])
    if holdings_code:
        holdings = holdings.rename(columns={holdings_code: "scheme_code"})
    if holdings_asset:
        holdings = holdings.rename(columns={holdings_asset: "asset"})
    if holdings_weight:
        holdings = holdings.rename(columns={holdings_weight: "weight"})
    if "scheme_code" in holdings.columns:
        holdings["scheme_code"] = holdings["scheme_code"].astype(str)
    else:
         st.warning("Holdings file should contain a scheme code column.")
         holdings = None


if metadata is not None and nav_history is not None:
    st.header("Data Preview")
    st.write("Metadata Sample:")
    st.dataframe(metadata.head())
    st.write("NAV History Sample:")
    st.dataframe(nav_history.head())

    # Compute metrics
    st.header("Fund Performance Metrics")
    metrics_df = compute_scheme_metrics(nav_history)
    funds = metadata.merge(metrics_df, on="scheme_code", how="left")

    # For display: basic name/amc/category columns
    if "scheme_name" not in funds.columns:
        funds["scheme_name"] = funds["scheme_code"]
    if "amc" not in funds.columns:
        funds["amc"] = np.nan
    if "category" not in funds.columns:
        funds["category"] = np.nan

    st.dataframe(funds[["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","cagr_5y","sharpe","sortino"]].head())

    # Smart Scoring
    st.header("Smart Score Analysis")
    st.sidebar.header("Smart Score Weights")
    w_return = st.sidebar.slider("Weight for Return", 0.0, 1.0, 0.6, 0.05)
    w_risk = st.sidebar.slider("Weight for Risk (Sharpe/Sortino)", 0.0, 1.0, 0.3, 0.05)
    w_cost = st.sidebar.slider("Weight for Cost (Expense Ratio)", 0.0, 1.0, 0.1, 0.05)
    return_horizon_options = ["cagr_1y", "cagr_3y", "cagr_5y"]
    return_horizon = st.sidebar.selectbox("Return Horizon for Scoring", return_horizon_options, index=return_horizon_options.index("cagr_3y"))

    scored_funds = compute_smart_score(funds, w_return, w_risk, w_cost, return_horizon)
    st.subheader("Funds Ranked by Smart Score")
    display_cols = ["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","cagr_5y","sharpe","sortino","smart_score"]
    st.dataframe(scored_funds[display_cols].head(15))

    # Recommendation Engine
    st.header("Fund Recommendation Engine")
    st.sidebar.header("Recommendation Profile")
    time_horizon_rec = st.sidebar.selectbox("Time Horizon", ["short", "medium", "long"])
    risk_appetite_rec = st.sidebar.selectbox("Risk Appetite", ["low", "medium", "high"])

    recommended = recommend_funds(funds, time_horizon_rec, risk_appetite_rec, top_n=10,
                                 w_return=w_return, w_risk=w_risk, w_cost=w_cost, return_horizon=return_horizon)
    st.subheader(f"Recommended Funds for {time_horizon_rec} horizon, {risk_appetite_rec} risk")
    if recommended is not None:
        st.dataframe(recommended)

    # Visualization
    st.header("Fund Performance Visualizations")

    # NAV Comparison Plot
    st.subheader("NAV Performance Comparison")
    if "scheme_code" in funds.columns and not funds["scheme_code"].empty:
        all_scheme_codes = funds["scheme_code"].unique().tolist()
        selected_schemes_nav = st.multiselect("Select schemes to compare NAV", all_scheme_codes, default=all_scheme_codes[:2])

        if selected_schemes_nav:
            plt.figure(figsize=(12,6))
            for scheme in selected_schemes_nav:
                scheme_data = nav_history[nav_history["scheme_code"].astype(str) == str(scheme)].sort_values("date")
                if not scheme_data.empty:
                     # Check if 'date' is datetime and 'nav' is numeric
                    if pd.api.types.is_datetime64_any_dtype(scheme_data['date']) and pd.api.types.is_numeric_dtype(scheme_data['nav']):
                        plt.plot(scheme_data["date"], scheme_data["nav"], label=f"{scheme} - {funds[funds['scheme_code']==scheme]['scheme_name'].iloc[0] if scheme in funds['scheme_code'].values else ''}")
                    else:
                         st.warning(f"Skipping NAV plot for scheme {scheme}: 'date' or 'nav' column has incorrect data type.")
                else:
                    st.warning(f"No NAV data found for {scheme}")
            plt.title("NAV Performance Comparison")
            plt.xlabel("Date")
            plt.ylabel("NAV")
            plt.legend()
            st.pyplot(plt)
            plt.close() # Close the figure to free memory
        else:
            st.info("Select at least one scheme to plot NAV comparison.")

    # Risk vs Return Scatter Plot
    st.subheader("Risk vs Return Scatter Plot")
    if "ann_vol" in funds.columns and return_horizon in funds.columns:
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=funds, x="ann_vol", y=return_horizon, hue="category", alpha=0.7)
        plt.title(f"Risk (Annualized Volatility) vs Return ({return_horizon})")
        plt.xlabel("Annualized Volatility (Risk)")
        plt.ylabel(f"{return_horizon} (Return)")
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
        st.pyplot(plt)
        plt.close()
    else:
        st.info("Risk (Volatility) or selected Return Horizon data not available for scatter plot.")


    # CAGR Bar Chart
    st.subheader("CAGR Comparison Bar Chart")
    if "scheme_code" in funds.columns and return_horizon in funds.columns:
        all_scheme_codes = funds["scheme_code"].unique().tolist()
        selected_schemes_cagr = st.multiselect("Select schemes for CAGR comparison", all_scheme_codes, default=all_scheme_codes[:5])
        if selected_schemes_cagr:
            selected_funds = funds[funds["scheme_code"].isin(selected_schemes_cagr)].copy()
            if not selected_funds.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(x="scheme_name", y=return_horizon, data=selected_funds)
                plt.title(f"CAGR Comparison ({return_horizon})")
                plt.xlabel("Scheme Name")
                plt.ylabel(return_horizon)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            else:
                st.info("No data available for selected schemes for CAGR bar chart.")
        else:
            st.info("Select at least one scheme for CAGR bar chart.")
    else:
        st.info("Scheme code or selected Return Horizon data not available for CAGR bar chart.")


    # Box Plot by Category
    st.subheader("Distribution of Metrics by Category")
    if "category" in funds.columns:
        metric_for_boxplot = st.selectbox("Select metric for Box Plot", ["cagr_1y", "cagr_3y", "cagr_5y", "sharpe", "sortino", "expense_ratio", "ann_vol"], index=2)
        if metric_for_boxplot in funds.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x="category", y=metric_for_boxplot, data=funds)
            plt.title(f"Distribution of {metric_for_boxplot} by Category")
            plt.xlabel("Category")
            plt.ylabel(metric_for_boxplot)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        else:
             st.info(f"Metric '{metric_for_boxplot}' not available in the data for box plot.")
    else:
        st.info("Category data not available for box plot.")


    # Correlation Heatmap
    st.subheader("Correlation Heatmap of Fund Metrics")
    metrics_for_heatmap = ["cagr_1y", "cagr_3y", "cagr_5y", "expense_ratio", "ann_vol", "sharpe", "sortino"]
    available_metrics = [m for m in metrics_for_heatmap if m in funds.columns]
    if len(available_metrics) > 1:
        correlation_matrix = funds[available_metrics].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of Fund Metrics")
        st.pyplot(plt)
        plt.close()
    else:
        st.info("Not enough metrics available to generate a correlation heatmap.")


    # Overlap Analysis
    st.header("Fund Overlap Analysis")
    if holdings is not None and "scheme_code" in holdings.columns and not holdings["scheme_code"].empty:
        all_holdings_schemes = holdings["scheme_code"].unique().tolist()
        scheme_a_overlap = st.selectbox("Select first scheme for overlap analysis", all_holdings_schemes)
        scheme_b_overlap = st.selectbox("Select second scheme for overlap analysis", all_holdings_schemes)
        overlap_by = st.selectbox("Overlap by", ["asset"], index=0) # Assuming 'asset' is the standardized column name

        if scheme_a_overlap and scheme_b_overlap and scheme_a_overlap != scheme_b_overlap:
            overlap_res = holdings_overlap(holdings, scheme_a_overlap, scheme_b_overlap, by=overlap_by, top_k=10)
            if overlap_res:
                st.subheader(f"Overlap between {scheme_a_overlap} and {scheme_b_overlap}")
                st.write(f"Overlap Percentage: **{overlap_res['overlap_pct']:.2f}%** — Level: **{overlap_res['level']}**")
                st.write("Top Common Holdings:")
                st.dataframe(overlap_res["top_common"][["asset","weight_a_pct","weight_b_pct","min_pct"]].rename(columns={"weight_a_pct": f"{scheme_a_overlap} Weight (%)", "weight_b_pct": f"{scheme_b_overlap} Weight (%)", "min_pct": "Min Weight (%)"}))
        else:
            st.info("Select two different schemes for overlap analysis (if holdings data available).")
    else:
        st.info("Holdings data not available for overlap analysis.")


elif uploaded_metadata is None or uploaded_nav_history is None:
    st.info("Please upload the Mutual Fund Metadata and NAV History CSV files to get started.")
