import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import zscore

# --------------------
# Streamlit Page Setup
# --------------------
st.set_page_config(page_title="Mutual Fund Screener", layout="wide")

st.title("📊 Mutual Fund Screener")
st.markdown("Analyze mutual funds by return, risk, and other metrics.")

# --------------------
# Data Loading (with caching)
# --------------------
@st.cache_data
def load_data():
    # Replace with your actual dataset
    df = pd.DataFrame({
        "Fund": ["Fund A", "Fund B", "Fund C", "Fund D"],
        "Return": [12, 9, 15, 7],
        "Volatility": [8, 6, 10, 5],
        "Expense_Ratio": [1.2, 1.0, 1.5, 0.9]
    })
    return df

df = load_data()

# --------------------
# Sidebar Controls
# --------------------
st.sidebar.header("Filter Options")
min_return = st.sidebar.slider("Minimum Return (%)", 0, 20, 5)
max_volatility = st.sidebar.slider("Maximum Volatility (%)", 0, 20, 12)

filtered_df = df[(df["Return"] >= min_return) & (df["Volatility"] <= max_volatility)]

# --------------------
# Display Data
# --------------------
st.subheader("Filtered Funds")
st.dataframe(filtered_df)

# --------------------
# Plots
# --------------------
st.subheader("Performance Visualization")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Volatility", y="Return", hue="Fund", s=100, ax=ax)
    plt.title("Return vs Volatility")
    st.pyplot(fig)

with col2:
    fig2 = px.bar(filtered_df, x="Fund", y="Expense_Ratio", color="Fund",
                  title="Expense Ratio Comparison", text="Expense_Ratio")
    fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)

# --------------------
# Smart Scoring
# --------------------
st.subheader("Custom Scoring Model")

w_return = st.slider("Weight for Return", 0.0, 1.0, 0.5)
w_risk = st.slider("Weight for Risk (Volatility)", 0.0, 1.0, 0.3)
w_expense = st.slider("Weight for Expense Ratio", 0.0, 1.0, 0.2)

def calculate_score(row, w1, w2, w3):
    return (w1 * row["Return"]) - (w2 * row["Volatility"]) - (w3 * row["Expense_Ratio"])

filtered_df["Score"] = filtered_df.apply(lambda row: calculate_score(row, w_return, w_risk, w_expense), axis=1)

st.dataframe(filtered_df.sort_values("Score", ascending=False))
