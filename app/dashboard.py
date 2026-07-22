import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from predictor import predict_volatility


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Volatility Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Volatility Forecasting Dashboard")
st.write("Predict next-day volatility using a trained LightGBM model.")

st.divider()


# --------------------------------------------------
# File Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRICE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "TCS_NS.csv"
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

if not os.path.exists(PRICE_PATH):
    st.error("Dataset not found.")
    st.stop()

df = pd.read_csv(PRICE_PATH)

df.columns = [c.strip() for c in df.columns]

df["Date"] = pd.to_datetime(df["Date"])


# --------------------------------------------------
# Prediction
# --------------------------------------------------

result = predict_volatility()

if "error" in result:
    st.error(result["error"])
    st.stop()


# --------------------------------------------------
# Metrics
# --------------------------------------------------

st.subheader("Prediction Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Predicted Volatility",
    result["predicted_vol_pct"]
)

c2.metric(
    "Current Volatility",
    f"{result['current_vol']:.2%}"
)

c3.metric(
    "Current Price",
    f"₹{result['current_price']:,.2f}"
)

c4.metric(
    "Risk Level",
    result["risk_level"]
)

st.divider()


# --------------------------------------------------
# Closing Price Chart
# --------------------------------------------------

st.subheader("TCS Closing Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

st.divider()


# --------------------------------------------------
# Volume Chart
# --------------------------------------------------

if "Volume" in df.columns:

    st.subheader("Trading Volume")

    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume"
        )
    )

    fig2.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("Latest Records")

st.dataframe(
    df.tail(10),
    use_container_width=True,
    hide_index=True
)

st.divider()

st.success("Prediction completed successfully.")
