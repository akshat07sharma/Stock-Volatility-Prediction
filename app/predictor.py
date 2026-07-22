import os
import joblib
import pandas as pd


# --------------------------------------------------
# Base Path
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lgb_model_TCS_NS.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "features_TCS_NS.csv"
)

PRICE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "TCS_NS.csv"
)


# --------------------------------------------------
# Risk Level
# --------------------------------------------------
def get_risk_level(volatility):

    if volatility >= 0.30:
        return "High"

    elif volatility >= 0.15:
        return "Medium"

    return "Low"


# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_volatility():

    if not os.path.exists(MODEL_PATH):
        return {
            "error": "Model file not found."
        }

    if not os.path.exists(FEATURE_PATH):
        return {
            "error": "Feature file not found."
        }

    if not os.path.exists(PRICE_PATH):
        return {
            "error": "Price dataset not found."
        }

    # Load Model
    model = joblib.load(MODEL_PATH)

    # Load Engineered Features
    features = pd.read_csv(FEATURE_PATH)

    # Remove target if present
    if "target" in features.columns:
        X = features.drop(columns=["target"])
    else:
        X = features.copy()

    # Latest row only
    latest_features = X.iloc[[-1]]

    prediction = float(model.predict(latest_features)[0])

    # Load original dataset
    price_df = pd.read_csv(PRICE_PATH)

    latest_price = float(price_df["Close"].iloc[-1])

    latest_date = str(price_df.iloc[-1, 0])

    current_volatility = (
        float(features["vol"].iloc[-1])
        if "vol" in features.columns
        else prediction
    )

    current_rsi = (
        float(features["rsi_14"].iloc[-1])
        if "rsi_14" in features.columns
        else 0.0
    )

    return {

        "prediction_date": latest_date,

        "predicted_vol": round(prediction, 4),

        "predicted_vol_pct": f"{prediction:.2%}",

        "current_vol": round(current_volatility, 4),

        "current_price": round(latest_price, 2),

        "rsi": round(current_rsi, 2),

        "risk_level": get_risk_level(prediction)

    }


# --------------------------------------------------
# Test
# --------------------------------------------------
if __name__ == "__main__":
    print(predict_volatility())
