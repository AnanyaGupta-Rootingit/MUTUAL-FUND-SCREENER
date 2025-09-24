# ==============================================================================
# Combined Mutual Fund Analysis and Visualization Code
# ==============================================================================

# ==============================
# Imports and Setup
# ==============================
from google.colab import files
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

# Optional: interactive sliders in Colab
try:
    from ipywidgets import interact, FloatSlider, fixed
    interactive_available = True
except Exception:
    interactive_available = False

# ==============================
# Data Loading and Preparation
# ==============================

# Load required files robustly (normalize names)
def load_uploaded_csv(filename_keys):
    # Check if file exists in /content from previous uploads
    for key in filename_keys:
        if os.path.exists(key):
            print(f"✅ Found existing file: {key}")
            return pd.read_csv(key)

    # If not found locally, try to load from files.upload() dict
    if 'uploaded' in globals() and uploaded:
        for key in filename_keys:
            if key in uploaded:
                 print(f"✅ Loading from uploaded files: {key}")
                 return pd.read_csv(io.BytesIO(uploaded[key]))
        # try case-insensitive from uploaded dict
        for key in uploaded:
            for fk in filename_keys:
                if fk.lower() in key.lower():
                     print(f"✅ Loading from uploaded files (case-insensitive): {key}")
                     return pd.read_csv(io.BytesIO(uploaded[key]))
    return None

# Try loading first, if not found, prompt upload
print("📂 Attempting to load required files...")
metadata = load_uploaded_csv(["mutual_fund_metadata.csv","metadata.csv","funds_metadata.csv"])
nav_history = load_uploaded_csv(["mutual_fund_nav_history.csv","nav_history.csv","nav.csv"])
holdings = load_uploaded_csv(["mutual_fund_holdings.csv","holdings.csv"])  # optional

if metadata is None or nav_history is None:
    print("📂 Required files not found locally. Please upload:")
    print("- mutual_fund_metadata.csv (or similar)")
    print("- mutual_fund_nav_history.csv (or similar)")
    if holdings is None:
        print("- mutual_fund_holdings.csv (optional)")

    uploaded = files.upload() # Prompt upload if files not found

    # Try loading again from the new upload
    metadata = load_uploaded_csv(["mutual_fund_metadata.csv","metadata.csv","funds_metadata.csv"])
    nav_history = load_uploaded_csv(["mutual_fund_nav_history.csv","nav_history.csv","nav.csv"])
    holdings = load_uploaded_csv(["mutual_fund_holdings.csv","holdings.csv"])  # optional

    if metadata is None or nav_history is None:
         raise FileNotFoundError("Required metadata or nav_history CSVs still not found after upload attempt.")
else:
    print("✅ Required files loaded successfully.")


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
    raise ValueError("nav_history must have columns like 'scheme_code','date','nav' (case-insensitive).")

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


# ---------------------------
# Filtering Functions
# ---------------------------
def filter_by_amc(fund_house, df=funds):
    return df[df["amc"].str.contains(fund_house, case=False, na=False)]

def filter_by_category(category, df=funds):
    return df[df["category"].str.contains(category, case=False, na=False)]

def filter_by_expense_ratio(max_ratio, df=funds):
    return df[df["expense_ratio"] <= max_ratio]

# Assuming 'minimum_investment' column exists in metadata and is merged into funds
# if not 'minimum_investment' in funds.columns:
#     print("Warning: 'minimum_investment' column not found. Skipping filter_by_min_investment.")
#     def filter_by_min_investment(max_investment, df=funds):
#         print("'minimum_investment' column not available.")
#         return df # Return original df if column is missing
def filter_by_min_investment(max_investment, df=funds):
     if 'minimum_investment' in df.columns:
        return df[df["minimum_investment"] <= max_investment]
     else:
        print("Warning: 'minimum_investment' column not found. Skipping filter_by_min_investment.")
        return df # Return original df if column is missing


# ---------------------------
# Sorting Functions
# ---------------------------
def top_performing(period="cagr_3y", top_n=10, df=funds):
    if period not in df.columns:
        print(f"Warning: Metric '{period}' not found for sorting.")
        return df.head(top_n)
    return df.sort_values(by=period, ascending=False).head(top_n)

def lowest_expense_ratio(top_n=10, df=funds):
    if "expense_ratio" not in df.columns:
        print("Warning: 'expense_ratio' column not found for sorting.")
        return df.head(top_n)
    return df.sort_values(by="expense_ratio", ascending=True).head(top_n)

def best_risk_adjusted(top_n=10, df=funds):
    if "sharpe" not in df.columns: # Use 'sharpe' as computed in compute_scheme_metrics
        print("Warning: 'sharpe' column not available for sorting.")
        return df.head(top_n)
    return df.sort_values(by="sharpe", ascending=False).head(top_n)

# ---------------------------
# Smart Scoring Model
# ---------------------------
# compute_smart_score function defined above in Data Loading section

# If interactive sliders available, show them (function interactive_scorer defined above)
def interactive_scorer(default_return=0.6, default_risk=0.3, default_cost=0.1, horizon="cagr_3y"):
    if not interactive_available:
        print("Interactive widgets not available. Call compute_smart_score(...) directly.")
        return

    def _compute(wr, wk, wc):
        print(f"Computing Smart Score (weights R={wr:.2f}, Risk={wk:.2f}, Cost={wc:.2f}, horizon={horizon})")
        scored = compute_smart_score(funds, w_return=wr, w_risk=wk, w_cost=wc, return_horizon=horizon)
        display_cols = ["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","sharpe","smart_score"]
        # Filter display columns based on availability
        display_cols = [col for col in display_cols if col in scored.columns]
        display(scored[display_cols].head(15))

    print("\nUse the sliders below to adjust Smart Score weights:")
    interact(_compute,
             wr=FloatSlider(min=0.0, max=1.0, step=0.05, value=default_return, description='w_return'),
             wk=FloatSlider(min=0.0, max=1.0, step=0.05, value=default_risk, description='w_risk'),
             wc=FloatSlider(min=0.0, max=1.0, step=0.05, value=default_cost, description='w_cost'))


# ---------------------------
# Fund Recommendation Engine
# ---------------------------
# Basic mapping from user profile -> categories (mapping_time_risk defined above)
mapping_time_risk = {
    # short horizon: prefer debt/liquid/ultrashort or low-volatility hybrids
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

def recommend_funds(time_horizon="medium", risk_appetite="medium", top_n=10,
                    w_return=0.6, w_risk=0.3, w_cost=0.1, return_horizon="cagr_3y", df=funds):
    # normalize inputs
    time_horizon = time_horizon.lower()
    risk_appetite = risk_appetite.lower()
    key = (time_horizon, risk_appetite)
    categories = mapping_time_risk.get(key, None)
    if categories is None:
        print("Profile not recognized. Options for time_horizon: short/medium/long; risk_appetite: low/medium/high")
        return None

    scored = compute_smart_score(df, w_return=w_return, w_risk=w_risk, w_cost=w_cost, return_horizon=return_horizon)
    # match by category (case-insensitive contain)
    mask = scored["category"].fillna("").apply(lambda x: any(cat.lower() in x.lower() for cat in categories))
    selected = scored[mask].copy()
    if selected.empty and categories: # Fallback if no exact match, try token matching if categories is not empty
        category_tokens = [token.lower() for cat in categories for token in cat.split()]
        mask = scored["category"].fillna("").apply(lambda x: any(token in x.lower() for token in category_tokens))
        selected = scored[mask].copy()

    # final sort by smart_score
    final = selected.sort_values("smart_score", ascending=False).head(top_n)
    display_cols = ["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","cagr_5y","sharpe","sortino","smart_score"]
    display_cols = [col for col in display_cols if col in final.columns]
    return final[display_cols]


# ---------------------------
# Fund Overlap Analysis
# ---------------------------
# holdings_overlap function defined above

# ---------------------------
# Visualization Functions
# ---------------------------
# plot_nav_comparison function defined above
# plot_risk_return function defined above
# plot_radar function defined above
# plot_cagr_bar_chart function defined above
# plot_portfolio_pie function defined above
# plot_boxplot_by_category function defined above
# plot_correlation_heatmap function defined above

# Make sure matplotlib shows plots in Colab
%matplotlib inline

# --- 📊 NAV Performance Comparison ---
def plot_nav_comparison(scheme_codes, nav_history):
    plt.figure(figsize=(12,6))
    for scheme in scheme_codes:
        scheme_data = nav_history[nav_history["scheme_code"].astype(str) == str(scheme)]
        if not scheme_data.empty:
            plt.plot(scheme_data["date"], scheme_data["nav"], label=str(scheme))
        else:
            print(f"⚠️ No NAV data found for {scheme}")
    plt.title("NAV Performance Comparison")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 📈 Risk vs Return Scatter ---
def plot_risk_return(funds, nav_history, period="cagr_3y"):
    risk_data = []
    for scheme, group in nav_history.groupby("scheme_code"):
        group = group.sort_values("date")
        group["Daily_Return"] = group["nav"].pct_change()
        if group["Daily_Return"].notna().sum() > 0:
            volatility = group["Daily_Return"].std() * np.sqrt(252)  # annualized
            risk_data.append([scheme, volatility])
    risk_df = pd.DataFrame(risk_data, columns=["scheme_code", "Volatility"])

    merged = funds.merge(risk_df, on="scheme_code", how="left")

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=merged, x="Volatility", y=period, hue="category", alpha=0.7)
    plt.title(f"Risk vs Return ({period})")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel(f"{period} (Return)")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True)
    plt.show()

# --- 🕸️ Spider/Radar Plot ---
def plot_radar(funds, scheme_codes):
    selected = funds[funds["scheme_code"].isin(scheme_codes)].copy() # Create a copy to avoid SettingWithCopyWarning
    metrics = ["cagr_3y", "cagr_5y", "expense_ratio"]

    # Normalize metrics
    scaler = MinMaxScaler()
    # Exclude rows with NaN in metrics before scaling
    selected_for_scaling = selected.dropna(subset=metrics)

    if not selected_for_scaling.empty:
        selected.loc[selected_for_scaling.index, metrics] = scaler.fit_transform(selected_for_scaling[metrics])
    else:
        print("⚠️ Cannot generate radar plot: No data available for selected metrics and schemes.")
        return

    categories = metrics + [metrics[0]]  # loop back

    fig = go.Figure()

    for _, row in selected.iterrows():
        # Use normalized values, handle potential NaNs after merge
        values = [row[m] if pd.notnull(row[m]) else 0 for m in metrics]
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=row["scheme_name"]
        ))

    fig.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 1])), # Set range for normalized data
      showlegend=True,
      title="Fund Comparison Radar Plot (Normalized Metrics)"
    )
    fig.show()


# --- 📊 Bar Chart for CAGR Comparison ---
def plot_cagr_bar_chart(funds, scheme_codes, period="cagr_3y"):
    selected = funds[funds["scheme_code"].isin(scheme_codes)].copy()
    if selected.empty:
        print("⚠️ No data available for selected schemes.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x="scheme_name", y=period, data=selected)
    plt.title(f"CAGR Comparison ({period})")
    plt.xlabel("Scheme Name")
    plt.ylabel(period)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# --- 🥧 Portfolio Pie Chart (if holdings available) ---
def plot_portfolio_pie(holdings_df, scheme_code):
    scheme_data = holdings_df[holdings_df["scheme_code"] == scheme_code]
    if scheme_data.empty:
        print("⚠️ Holdings data not available for this scheme.")
        return

    plt.figure(figsize=(8,8))
    plt.pie(scheme_data["Weight"], labels=scheme_data["Sector"], autopct="%1.1f%%", startangle=140)
    plt.title(f"Portfolio Exposure - {scheme_code}")
    plt.show()


# --- 📊 Box Plot of Metrics by Category ---
def plot_boxplot_by_category(funds, metric="cagr_3y", category="category"):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=category, y=metric, data=funds)
    plt.title(f"Distribution of {metric} by {category}")
    plt.xlabel(category)
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# --- 📊 Heatmap of Correlations ---
def plot_correlation_heatmap(funds):
    metrics = ["cagr_1y", "cagr_3y", "cagr_5y", "expense_ratio", "Volatility"] # Add Volatility if calculated and merged

    # Ensure 'funds' has 'Volatility' if 'plot_risk_return' was run
    if "Volatility" not in funds.columns:
         risk_data = []
         for scheme, group in nav_history.groupby("scheme_code"):
            group = group.sort_values("date")
            group["Daily_Return"] = group["nav"].pct_change()
            if group["Daily_Return"].notna().sum() > 0:
                volatility = group["Daily_Return"].std() * np.sqrt(252)  # annualized
                risk_data.append([scheme, volatility])
         risk_df = pd.DataFrame(risk_data, columns=["scheme_code", "Volatility"])
         funds = funds.merge(risk_df, on="scheme_code", how="left")


    correlation_matrix = funds[metrics].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Fund Metrics")
    plt.show()


# =====================
# Example Usage
# =====================
print("\n--- Example Usage ---")

# 1) Smart Scoring example (non-interactive)
print("\n▶ Smart Scoring example (non-interactive): top 10 by default weights (0.6,0.3,0.1)")
scored_sample = compute_smart_score(funds, w_return=0.6, w_risk=0.3, w_cost=0.1, return_horizon="cagr_3y")
display_cols = ["scheme_code","scheme_name","amc","category","expense_ratio","cagr_1y","cagr_3y","sharpe","sortino","smart_score"]
display_cols = [col for col in display_cols if col in scored_sample.columns] # Filter based on available columns
print(scored_sample[display_cols].head(10))

# To use interactive scorer (if ipywidgets available):
# interactive_scorer()


# 2) Recommendation engine example
print("\n▶ Recommendation example for (time_horizon='medium', risk='medium'):")
rec = recommend_funds(time_horizon="medium", risk_appetite="medium", top_n=7,
                      w_return=0.6, w_risk=0.3, w_cost=0.1, return_horizon="cagr_3y", df=funds)
if rec is not None:
    display(rec)

# 3) Overlap analysis (only if holdings provided)
if holdings is not None:
    # Example: pick two sample scheme codes
    codes = holdings["scheme_code"].unique()
    if len(codes) >= 2:
        a, b = codes[0], codes[1]
        print(f"\n▶ Overlap analysis between {a} and {b}:")
        overlap_res = holdings_overlap(holdings, a, b, by="asset", top_k=10)
        if overlap_res:
            print(f"Overlap %: {overlap_res['overlap_pct']:.2f}%  — Level: {overlap_res['level']}")
            display(overlap_res["top_common"])
    else:
        print("Holdings file has fewer than 2 schemes; cannot do overlap demo.")
else:
    print("\n(No holdings dataset uploaded — skip overlap analysis.)")

# 4) Visualization examples
print("\n--- Visualization Examples ---")
# Make sure your dataset columns are:
# metadata/funds → ["scheme_code", "scheme_name", "category", "cagr_3y", "cagr_5y", "expense_ratio"]
# nav_history    → ["scheme_code", "date", "nav"]

# Example scheme codes (replace with actual codes from your data if needed)
example_scheme_codes = funds["scheme_code"].unique().tolist()[:3] # Get first 3 codes

if example_scheme_codes:
    print(f"\nPlotting NAV Comparison for schemes: {example_scheme_codes}")
    plot_nav_comparison(example_scheme_codes, nav_history)

    print(f"\nPlotting Risk vs Return ({'cagr_3y'})")
    plot_risk_return(funds, nav_history, "cagr_3y")

    print(f"\nPlotting Radar Chart for schemes: {example_scheme_codes}")
    plot_radar(funds, example_scheme_codes)

    print(f"\nPlotting CAGR Bar Chart ({'cagr_3y'}) for schemes: {example_scheme_codes}")
    plot_cagr_bar_chart(funds, example_scheme_codes, "cagr_3y")

    print(f"\nPlotting Box Plot of CAGR by Category")
    plot_boxplot_by_category(funds, metric="cagr_3y", category="category")

    print(f"\nPlotting Correlation Heatmap")
    plot_correlation_heatmap(funds)

    # If you have holdings data, you can uncomment the pie chart example:
    # if holdings is not None and example_scheme_codes:
    #     print(f"\nPlotting Portfolio Pie Chart for scheme: {example_scheme_codes[0]}")
    #     plot_portfolio_pie(holdings, example_scheme_codes[0])
else:
    print("\nNo scheme codes found in data to generate visualization examples.")


# 5) Save results
output_scored_file = "funds_scored.csv"
scored_sample.to_csv(output_scored_file, index=False)
print(f"\nSaved scored funds to {output_scored_file}. Download from Files panel or re-run with files.download().")

# Optional: Allow download of results
# files.download(output_scored_file)