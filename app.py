

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import joblib
import os
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, filename="trade_platform.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fetch stock data
def fetch_stock_data(tickers, years=1):
    logger.info("Fetching stock data for %s", tickers)
    end = datetime.now()
    start = datetime(end.year - years, end.month, end.day)
    try:
        data = yf.download(tickers, start=start, end=end, group_by='ticker')
        if data.empty:
            raise ValueError("No data downloaded. Check tickers or date range.")
        df = data.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        logger.info("Data fetched successfully, shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error downloading data: %s", e)
        return None

# Feature engineering
def feature_engineering(df):
    logger.info("Starting feature engineering")
    df_feat = df.copy()
    df_feat["Date"] = pd.to_datetime(df_feat["Date"])
    df_feat = df_feat.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    df_feat["Daily_Return"] = df_feat.groupby("Ticker")["Close"].pct_change()
    df_feat = df_feat.dropna(subset=["Daily_Return"])
    df_feat["Daily_Range"] = df_feat["High"] - df_feat["Low"]
    df_feat["Open_Close_Diff"] = df_feat["Close"] - df_feat["Open"]
    df_feat["Cum_Return"] = df_feat.groupby("Ticker")["Daily_Return"].transform(
        lambda x: (1 + x).cumprod() - 1
    )
    for lag in [1, 2, 5]:
        df_feat[f"Lag{lag}_Return"] = df_feat.groupby("Ticker")["Daily_Return"].shift(lag)
    df_feat["Lag1_Close"] = df_feat.groupby("Ticker")["Close"].shift(1)
    for window in [5, 10, 20, 50]:
        df_feat[f"SMA_{window}"] = (
            df_feat.groupby("Ticker")["Close"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )
        df_feat[f"Close_vs_SMA_{window}_Diff"] = df_feat["Close"] - df_feat[f"SMA_{window}"]
    for window in [5, 20]:
        df_feat[f"Volatility_{window}d"] = (
            df_feat.groupby("Ticker")["Daily_Return"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())
        )
    df_feat["Volume_Change_Pct"] = df_feat.groupby("Ticker")["Volume"].pct_change()
    df_feat["Tomorrow_Close"] = df_feat.groupby("Ticker")["Close"].shift(-1)
    df_feat["Target"] = (df_feat["Tomorrow_Close"] > df_feat["Close"] * 1.01).astype(int)
    initial_rows = len(df_feat)
    df_feat = df_feat.dropna().reset_index(drop=True)
    logger.info("Dropped %d rows due to NaNs after feature engineering", initial_rows - len(df_feat))
    return df_feat

# Prepare sequences for LSTM
def prepare_sequences(df_feat, features, seq_len=10):
    logger.info("Preparing sequences for LSTM")
    X, y = [], []
    for ticker in df_feat["Ticker"].unique():
        ticker_df = df_feat[df_feat["Ticker"] == ticker]
        ticker_X = ticker_df[features].values
        ticker_y = ticker_df["Target"].values
        for i in range(len(ticker_X) - seq_len):
            X.append(ticker_X[i : i + seq_len])
            y.append(ticker_y[i + seq_len])
    X, y = np.array(X), np.array(y)
    scaler = joblib.load("scaler_lstm.pkl")
    X = X.reshape(X.shape[0], -1)
    X = scaler.transform(X)
    X = X.reshape(-1, seq_len, len(features))
    return X, y

# Trading metrics
def trading_metrics(test_df):
    logger.info("Calculating trading metrics")
    strategy_cum = (1 + test_df["Strategy_Returns"]).cumprod()
    total_return = strategy_cum.iloc[-1] - 1
    rf_annual = 0.04
    trading_days = 252
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    sharpe = (test_df["Strategy_Returns"].mean() - rf_daily) / test_df["Strategy_Returns"].std() * np.sqrt(252)
    dd = (strategy_cum / strategy_cum.cummax() - 1).min()
    cagr = (1 + total_return) ** (252 / len(test_df)) - 1
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": dd, "TotalReturn": total_return}

# Streamlit app
def streamlit_app():
    st.title("Trade Platform")
    correct_pwd = os.getenv("TRADE_PWD", "trade123")
    pwd = st.text_input("Password", type="password")
    if pwd != correct_pwd:
        st.error("Wrong password")
        logger.error("Incorrect password attempt")
        return

    tickers = ['META', 'NVDA', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
    ticker = st.selectbox("Ticker", tickers)
    model_names = ["RandomForest", "XGB", "LogisticRegression", "SVM"]
    if os.path.exists("trade_model.tflite"):
        model_names.append("LSTM")
    model_name = st.selectbox("Model", model_names)
    try:
        best_models = {name: joblib.load(f"best_model_{name}.pkl") for name in model_names if name != "LSTM"}
        features = joblib.load("features.pkl")
    except Exception as e:
        st.error(f"Error loading models or features: {e}")
        logger.error("Error loading models or features: %s", e)
        return

    if st.button("Predict"):
        try:
            logger.info("Predicting for ticker %s with model %s", ticker, model_name)
            if os.path.exists("processed_stock_data.parquet"):
                ticker_df = pd.read_parquet("processed_stock_data.parquet")
                ticker_df = ticker_df[ticker_df["Ticker"] == ticker]
            else:
                ticker_df = fetch_stock_data([ticker], years=1)
                ticker_df = feature_engineering(ticker_df)

            if ticker_df.empty:
                st.error("Insufficient data for prediction")
                logger.error("No data available for ticker %s", ticker)
                return

            if model_name == "LSTM":
                X_single, _ = prepare_sequences(ticker_df, features)
                if X_single.shape[0] == 0:
                    st.error("Insufficient sequence data for LSTM prediction")
                    logger.error("No sequences for LSTM prediction")
                    return
                interpreter = tf.lite.Interpreter(model_path="trade_model.tflite")
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]["index"], X_single[-1:].astype(np.float32))
                interpreter.invoke()
                prob = interpreter.get_tensor(output_details[0]["index"])[0][0]
            else:
                X_single = ticker_df[features]
                if model_name in ["LogisticRegression", "SVM"]:
                    scaler = joblib.load(f"scaler_{model_name.lower()}.pkl")
                    X_single = scaler.transform(X_single)
                if X_single.empty:
                    st.error("Insufficient data for prediction")
                    logger.error("No features available for prediction")
                    return
                model = best_models[model_name]
                prob = model.predict_proba(X_single.iloc[-1:])[:, 1][0]

            pred = "Buy" if prob > 0.5 else "Hold/Sell"
            st.success(f"Prediction: {pred} (Prob: {prob:.2f})")
            logger.info("Prediction: %s (Prob: %.2f)", pred, prob)

            # Plot closing price with buy signals
            fig = plt.figure(figsize=(10, 5))
            sns.lineplot(data=ticker_df, x="Date", y="Close", label="Close")
            if model_name != "LSTM":
                y_pred = model.predict(X_single)
                buy_signals = ticker_df[y_pred == 1]
                plt.scatter(buy_signals["Date"], buy_signals["Close"], color="green", label="Buy Signal", marker="^")
            plt.title(f"Closing Price for {ticker}")
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.legend()
            plt.grid(True)
            st.pyplot(fig)

            # Display trading metrics if available
            if model_name != "LSTM":
                test_df = ticker_df.copy()
                test_df["Predicted"] = y_pred
                test_df["Returns"] = test_df["Tomorrow_Close"] / test_df["Close"] - 1
                test_df["Strategy_Returns"] = test_df["Returns"] * test_df["Predicted"] - 0.001
                metrics = trading_metrics(test_df)
                st.write("### Trading Metrics")
                st.write(f"- **CAGR**: {metrics['CAGR']:.2%}")
                st.write(f"- **Sharpe Ratio**: {metrics['Sharpe']:.2f}")
                st.write(f"- **Max Drawdown**: {metrics['MaxDD']:.2%}")
                st.write(f"- **Total Return**: {metrics['TotalReturn']:.2%}")
                logger.info("Trading metrics for %s: %s", ticker, metrics)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            logger.error("Prediction error: %s", e)

if __name__ == "__main__":
    streamlit_app()
