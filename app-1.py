# ==============================================================================
# Streamlit App for Mutual Fund Analysis and Visualization
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import io
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Mutual Fund Analysis")

st.title("Mutual Fund Analysis and Visualization App")

# ==============================
# Data Loading and Preparation
# ==============================

st.header("📂 Upload Data Files")

uploaded_metadata = st.file_uploader("Upload Mutual Fund Metadata CSV", type=["csv"], key="metadata_uploader")
uploaded_nav = st.file_uploader("Upload Mutual Fund NAV History CSV", type=["csv"], key="nav_uploader")
uploaded_holdings = st.file_uploader("Upload Mutual Fund Holdings CSV (Optional)", type=["csv"], key="holdings_uploader")

metadata = None
nav_history = None
holdings = None

if uploaded_metadata is not None:
    try:
        metadata = pd.read_csv(uploaded_metadata)
        st.success("Metadata file uploaded successfully.")
    except Exception as e:
        st.error(f"Error reading metadata file: {e}")

if uploaded_nav is not None:
    try:
        nav_history = pd.read_csv(uploaded_nav)
        st.success("NAV history file uploaded successfully.")
    except Exception as e:
        st.error(f"Error reading NAV history file: {e}")

if uploaded_holdings is not None:
    try:
        holdings = pd.read_csv(uploaded_holdings)
        st.success("Holdings file uploaded successfully.")
    except Exception as e:
        st.error(f"Error reading holdings file: {e}")

# Process data if both required files are uploaded
if metadata is not None and nav_history is not None:
    st.header("⚙️ Data Processing")

    # Normalize column names
    def norm_cols(df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df

    metadata = norm_cols(metadata)
    nav_history = norm_cols(nav_history)
    if holdings is not None:
        holdings = norm_cols(holdings)

    # Expected column mapping helper
    def map_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # map critical columns
    meta_code_col = map_col(metadata, ["scheme_code","schemeid","code","id"])
    meta_name_col = map_col(metadata, ["scheme_name","name","fund_name"])
    meta_amc_col = map_col(metadata, ["amc","amc_name","fund_house"])
    meta_cat_col = map_col(metadata, ["category","fund_category","type"])
    meta_exp_col = map_col(metadata, ["expense_ratio","ter","expense"])
    meta_mininv_col = map_col(metadata, ["minimum_investment","min_investment","min_inv"])

    nav_code_col = map_col(nav_history, ["scheme_code","schemeid","code","id"])
    nav_date_col = map_col(nav_history, ["date","nav_date"])
    nav_nav_col = map_col(nav_history, ["nav","nav_value","navs"])

    if not nav_code_col or not nav_date_col or not nav_nav_col:
        st.error("NAV history must have columns like 'scheme_code','date','nav' (case-insensitive).")
        st.stop()

    # standardize names in dataframes
    metadata = metadata.rename(columns={c: "scheme_code" for c in [meta_code_col] if c})
    if meta_name_col: metadata = metadata.rename(columns={meta_name_col: "scheme_name"})
    if meta_amc_col: metadata = metadata.rename(columns={meta_amc_col: "amc"})
    if meta_cat_col: metadata = metadata.rename(columns={meta_cat_col: "category"})
    if meta_exp_col: metadata = metadata.rename(columns={meta_exp_col: "expense_ratio"})
    if meta_mininv_col: metadata = metadata.rename(columns={meta_mininv_col: "minimum_investment"})

    nav_history = nav_history.rename(columns={nav_code_col: "scheme_code", nav_date_col: "date", nav_nav_col: "nav"})

    if holdings is not None:
        # try to standardize holdings columns: scheme_code, stock/sector, weight
        holdings_code = map_col(holdings, ["scheme_code","schemeid","code","fund_code"])
        holdings_asset = map_col(holdings, ["stock","holding","security","name","ticker","sector"])
        holdings_weight = map_col(holdings, ["weight","percentage","pct","weight_pct","holding_percent"])
        if holdings_code:
            holdings = holdings.rename(columns={holdings_code: "scheme_code"})
        if holdings_asset:
            holdings = holdings.rename(columns={holdings_asset: "asset"})
        if holdings_weight:
            holdings = holdings.rename(columns={holdings_weight: "weight"})

    # Ensure types
    metadata["scheme_code"] = metadata["scheme_code"].astype(str)
    nav_history["scheme_code"] = nav_history["scheme_code"].astype(str)
    nav_history["date"] = pd.to_datetime(nav_history["date"], errors="coerce")
    nav_history["nav"] = pd.to_numeric(nav_history["nav"], errors="coerce")

    if "expense_ratio" in metadata.columns:
        metadata["expense_ratio"] = pd.to_numeric(metadata["expense_ratio"], errors="coerce")

    # ---------------------------
    # Helper metrics functions
    # ---------------------------
    def compute_scheme_metrics(nav_df, ref_date=None):
        """
        For each scheme_code compute:
        - last_date, last_nav
        - CAGR 1/3/5 if available (else NaN)
        - annualized volatility, Sharpe, Sortino from last ~1 year if possible (fallback to all history)
        Returns DataFrame indexed by scheme_code with columns.
        """
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

    # Compute metrics and merge with metadata
    metrics_df = compute_scheme_metrics(nav_history)
    funds = metadata.merge(metrics_df, on="scheme_code", how="left")

    # For display: basic name/amc/category columns
    if "scheme_name" not in funds.columns:
        funds["scheme_name"] = funds["scheme_code"]
    if "amc" not in funds.columns:
        funds["amc"] = np.nan
    if "category" not in funds.columns:
        funds["category"] = np.nan

    # Add Volatility to funds DataFrame for correlation heatmap if not already there
    risk_data_for_merge = []
    for scheme, group in nav_history.groupby("scheme_code"):
        group = group.sort_values("date")
        group["Daily_Return"] = group["nav"].pct_change()
        if group["Daily_Return"].notna().sum() > 0:
            volatility = group["Daily_Return"].std() * np.sqrt(252)  # annualized
            risk_data_for_merge.append([scheme, volatility])
    risk_df_for_merge = pd.DataFrame(risk_data_for_merge, columns=["scheme_code", "Volatility"])
    funds = funds.merge(risk_df_for_merge, on="scheme_code", how="left")

    st.success("Data processed and metrics computed.")

    # ---------------------------
    # Analysis and Visualization
    # ---------------------------
    st.header("📊 Analysis and Visualization")

    analysis_option = st.selectbox(
        "Choose an analysis or visualization:",
        [
            "Fund Performance Metrics (Table)",
            "Smart Scoring",
            "Recommendation Engine",
            "Fund Overlap Analysis",
            "NAV Performance Comparison",
            "Risk vs Return Scatter Plot",
            "Fund Comparison Radar Plot",
            "CAGR Comparison Bar Chart",
            "Metrics Distribution (Box Plot)",
            "Correlation Heatmap",
            "Portfolio Exposure Pie Chart (Requires Holdings)"
        ]
    )

    if analysis_option == "Fund Performance Metrics (Table)":
        st.subheader("📊 Fund Performance Metrics")
        display_cols = ["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","cagr_5y","ann_return","ann_vol","sharpe","sortino"]
        display_cols = [col for col in display_cols if col in funds.columns]
        st.dataframe(funds[display_cols])

    elif analysis_option == "Smart Scoring":
        st.subheader("🧠 Smart Scoring Model")
        st.write("Adjust weights for Return, Risk, and Cost to compute a Smart Score for funds.")

        col1, col2, col3 = st.columns(3)
        w_return = col1.slider("Weight for Return", 0.0, 1.0, 0.6, 0.05)
        w_risk = col2.slider("Weight for Risk", 0.0, 1.0, 0.3, 0.05)
        w_cost = col3.slider("Weight for Cost", 0.0, 1.0, 0.1, 0.05)
        return_horizon = st.selectbox("Return Horizon for Scoring", ["cagr_1y", "cagr_3y", "cagr_5y"], index=1)

        @st.cache_data # Cache the result of computation
        def compute_smart_score_cached(df, w_return, w_risk, w_cost, return_horizon):
             # Need to pass df explicitly to the cached function
             return compute_smart_score(df, w_return, w_risk, w_cost, return_horizon)

        scored_funds = compute_smart_score_cached(funds, w_return, w_risk, w_cost, return_horizon)

        st.write("Top 10 Funds by Smart Score:")
        display_cols = ["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","sharpe","sortino","smart_score"]
        display_cols = [col for col in display_cols if col in scored_funds.columns]
        st.dataframe(scored_funds[display_cols].head(10))


    elif analysis_option == "Recommendation Engine":
        st.subheader("✨ Fund Recommendation Engine")
        st.write("Get fund recommendations based on your investment profile.")

        col1, col2 = st.columns(2)
        time_horizon = col1.selectbox("Time Horizon", ["short", "medium", "long"])
        risk_appetite = col2.selectbox("Risk Appetite", ["low", "medium", "high"])
        num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

        # Use default smart score weights for recommendation
        recommended_funds = recommend_funds(time_horizon, risk_appetite, num_recommendations, df=funds)

        if recommended_funds is not None:
            st.write(f"Recommended Funds for {time_horizon} horizon and {risk_appetite} risk:")
            st.dataframe(recommended_funds)
        else:
            st.info("No recommendations found for the selected profile.")


    elif analysis_option == "Fund Overlap Analysis":
        st.subheader("🤝 Fund Overlap Analysis")
        if holdings is not None:
            scheme_codes_overlap = holdings["scheme_code"].unique().tolist()
            if len(scheme_codes_overlap) >= 2:
                col1, col2 = st.columns(2)
                scheme_a = col1.selectbox("Select Fund A", scheme_codes_overlap)
                scheme_b = col2.selectbox("Select Fund B", [code for code in scheme_codes_overlap if code != scheme_a])

                if scheme_a and scheme_b:
                    overlap_res = holdings_overlap(holdings, scheme_a, scheme_b, by="asset", top_k=10)
                    if overlap_res:
                        st.write(f"Overlap between {overlap_res['scheme_a']} and {overlap_res['scheme_b']}:")
                        st.write(f"**Overlap Percentage:** {overlap_res['overlap_pct']:.2f}%")
                        st.write(f"**Overlap Level:** {overlap_res['level']}")
                        st.write("Top 10 Common Assets:")
                        st.dataframe(overlap_res["top_common"][["asset","weight_a_pct","weight_b_pct","min_pct"]].rename(columns={
                            "asset":"Asset", "weight_a_pct":f"Weight in {scheme_a} (%)", "weight_b_pct":f"Weight in {scheme_b} (%)", "min_pct":"Min Weight (%)"
                        }))
                    else:
                         st.warning("Could not perform overlap analysis for selected schemes.")
            else:
                 st.info("Need at least two funds with holdings data for overlap analysis.")
        else:
            st.info("Please upload a Mutual Fund Holdings CSV file for overlap analysis.")

    elif analysis_option == "NAV Performance Comparison":
        st.subheader("📈 NAV Performance Comparison")
        scheme_codes_nav = st.multiselect("Select Funds to Compare NAV", funds["scheme_code"].unique())
        if scheme_codes_nav:
            fig, ax = plt.subplots(figsize=(12, 6))
            for scheme in scheme_codes_nav:
                scheme_data = nav_history[nav_history["scheme_code"].astype(str) == str(scheme)]
                if not scheme_data.empty:
                    ax.plot(scheme_data["date"], scheme_data["nav"], label=str(scheme))
            ax.set_title("NAV Performance Comparison")
            ax.set_xlabel("Date")
            ax.set_ylabel("NAV")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory

    elif analysis_option == "Risk vs Return Scatter Plot":
        st.subheader("📉 Risk vs Return Scatter Plot")
        risk_return_period = st.selectbox("Select Return Period", ["cagr_1y", "cagr_3y", "cagr_5y"], index=1)
        if risk_return_period in funds.columns:
             fig, ax = plt.subplots(figsize=(10,6))
             sns.scatterplot(data=funds, x="Volatility", y=risk_return_period, hue="category", alpha=0.7, ax=ax)
             ax.set_title(f"Risk vs Return ({risk_return_period})")
             ax.set_xlabel("Volatility (Risk)")
             ax.set_ylabel(f"{risk_return_period} (Return)")
             ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
             ax.grid(True)
             st.pyplot(fig)
             plt.close(fig) # Close the figure to free memory
        else:
             st.warning(f"Return period '{risk_return_period}' not found in data.")


    elif analysis_option == "Fund Comparison Radar Plot":
        st.subheader("🕸️ Fund Comparison Radar Plot")
        scheme_codes_radar = st.multiselect("Select Funds for Radar Plot", funds["scheme_code"].unique())
        if scheme_codes_radar:
            selected = funds[funds["scheme_code"].isin(scheme_codes_radar)].copy()
            metrics = ["cagr_3y", "cagr_5y", "expense_ratio"]

            # Normalize metrics
            scaler = MinMaxScaler()
            selected_for_scaling = selected.dropna(subset=metrics)

            if not selected_for_scaling.empty:
                selected.loc[selected_for_scaling.index, metrics] = scaler.fit_transform(selected_for_scaling[metrics])
                categories = metrics + [metrics[0]]  # loop back
                fig = go.Figure()
                for _, row in selected.iterrows():
                    values = [row[m] if pd.notnull(row[m]) else 0 for m in metrics]
                    values += values[:1]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill="toself",
                        name=row["scheme_name"]
                    ))
                fig.update_layout(
                  polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                  showlegend=True,
                  title="Fund Comparison Radar Plot (Normalized Metrics)"
                )
                st.plotly_chart(fig)
            else:
                st.warning("Cannot generate radar plot: No data available for selected metrics and schemes.")


    elif analysis_option == "CAGR Comparison Bar Chart":
        st.subheader("📊 CAGR Comparison Bar Chart")
        scheme_codes_cagr_bar = st.multiselect("Select Funds for CAGR Bar Chart", funds["scheme_code"].unique())
        cagr_bar_period = st.selectbox("Select CAGR Period", ["cagr_1y", "cagr_3y", "cagr_5y"], index=1, key="cagr_bar_period")
        if scheme_codes_cagr_bar:
            selected = funds[funds["scheme_code"].isin(scheme_codes_cagr_bar)].copy()
            if selected.empty:
                st.warning("No data available for selected schemes.")
            elif cagr_bar_period in selected.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="scheme_name", y=cagr_bar_period, data=selected, ax=ax)
                ax.set_title(f"CAGR Comparison ({cagr_bar_period})")
                ax.set_xlabel("Scheme Name")
                ax.set_ylabel(cagr_bar_period)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Close the figure to free memory
            else:
                st.warning(f"CAGR period '{cagr_bar_period}' not found in data.")


    elif analysis_option == "Metrics Distribution (Box Plot)":
        st.subheader("📦 Metrics Distribution (Box Plot)")
        boxplot_metric = st.selectbox("Select Metric for Box Plot", ["cagr_1y", "cagr_3y", "cagr_5y", "expense_ratio", "Volatility", "sharpe", "sortino"])
        boxplot_category = st.selectbox("Select Category for Box Plot", ["category", "amc"])

        if boxplot_metric in funds.columns and boxplot_category in funds.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=boxplot_category, y=boxplot_metric, data=funds, ax=ax)
            ax.set_title(f"Distribution of {boxplot_metric} by {boxplot_category}")
            ax.set_xlabel(boxplot_category)
            ax.set_ylabel(boxplot_metric)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory
        else:
            st.warning(f"Selected metric ('{boxplot_metric}') or category ('{boxplot_category}') not found in data.")


    elif analysis_option == "Correlation Heatmap":
        st.subheader("🔥 Correlation Heatmap")
        metrics = ["cagr_1y", "cagr_3y", "cagr_5y", "expense_ratio", "Volatility", "sharpe", "sortino"]
        available_metrics = [m for m in metrics if m in funds.columns]

        if len(available_metrics) >= 2:
            correlation_matrix = funds[available_metrics].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap of Fund Metrics")
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory
        else:
            st.info("Need at least two available metrics to plot correlation heatmap.")


    elif analysis_option == "Portfolio Exposure Pie Chart (Requires Holdings)":
        st.subheader("🥧 Portfolio Exposure Pie Chart")
        if holdings is not None:
            scheme_codes_pie = holdings["scheme_code"].unique().tolist()
            if scheme_codes_pie:
                selected_scheme_pie = st.selectbox("Select Fund for Pie Chart", scheme_codes_pie)
                scheme_data = holdings[holdings["scheme_code"] == selected_scheme_pie]
                if not scheme_data.empty and "asset" in scheme_data.columns and "weight" in scheme_data.columns:
                     # Aggregate by asset if needed (e.g., if multiple entries for same asset)
                     scheme_data_agg = scheme_data.groupby("asset")["weight"].sum().reset_index()
                     fig, ax = plt.subplots(figsize=(8,8))
                     ax.pie(scheme_data_agg["weight"], labels=scheme_data_agg["asset"], autopct="%1.1f%%", startangle=140)
                     ax.set_title(f"Portfolio Exposure - {selected_scheme_pie}")
                     st.pyplot(fig)
                     plt.close(fig) # Close the figure to free memory
                else:
                     st.warning("Holdings data for the selected scheme is incomplete (missing 'asset' or 'weight' columns).")

            else:
                st.info("No scheme codes found in holdings data.")
        else:
            st.info("Please upload a Mutual Fund Holdings CSV file for this visualization.")


else:
    st.info("Please upload the required Metadata and NAV History CSV files to proceed.")

# Add instructions on how to run the Streamlit app
st.markdown("""
---
**How to run this Streamlit app:**

1. Save the code above as a Python file (e.g., `mutual_fund_app.py`).
2. Open your terminal or command prompt.
3. Navigate to the directory where you saved the file.
4. Run the command: `streamlit run mutual_fund_app.py`
5. Your web browser will open with the Streamlit application.
""")
