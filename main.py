import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, filename="trade_platform.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_stock_data(tickers, years=1):
    logger.info("Fetching stock data for %s", tickers)
    end = datetime.now()
    start = datetime(end.year - years, end.month, end.day)
    try:
        data = yf.download(tickers, start=start, end=end, group_by='ticker')
        if data.empty:
            raise ValueError("No data downloaded. Check tickers or date range.")
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                logger.warning("No data for ticker %s", ticker)
        df = data.stack(level=0, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        logger.info("Data fetched successfully, shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error downloading data: %s", e)
        return None
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
    df_feat.to_parquet("processed_stock_data.parquet")
    return df_feat

def handle_correlation(df_feat, exclude_cols):
    logger.info("Handling high correlation features")
    features = [c for c in df_feat.columns if c not in exclude_cols]
    corr_matrix = df_feat[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    features = [f for f in features if f not in to_drop]
    logger.info("Dropped high corr features: %s", to_drop)
    return features
def train_test_split_with_smote(df_feat, X, y):
    logger.info("Performing train/test split")
    df_feat = df_feat.sort_values("Date")
    split_date = df_feat["Date"].quantile(0.8)
    train_mask = df_feat["Date"] <= split_date
    test_mask = df_feat["Date"] > split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    logger.info("Train size: %s, Test size: %s", X_train.shape, X_test.shape)
    logger.info("Tickers in train: %d, test: %d", df_feat[train_mask]["Ticker"].nunique(), df_feat[test_mask]["Ticker"].nunique())
    if y_train.value_counts(normalize=True).min() < 0.3:
        logger.info("Applying SMOTE due to class imbalance")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, test_mask


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

if __name__ == "__main__":
    tickers = ['META', 'NVDA', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
    df = fetch_stock_data(tickers)
    if df is None:
        logger.error("Exiting due to data fetch failure")
        exit(1)

    logger.info("Starting EDA")
    print("Data Head:\n", df.head())
    df = df.sort_values(by=["Ticker", "Date"])
    print("\nStacked Data Head:\n", df.head())
    print(f"\nDataFrame Shape: {df.shape}")
    print("\nDescriptive Stats:\n", df.groupby("Ticker")["Close"].describe())
    high_low = df.groupby("Ticker")["Close"].agg(["min", "max"])
    print("\nHigh/Low Closing Prices:\n", high_low)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Date", y="Close", hue="Ticker", data=df)
    plt.title("Closing Prices of Tech Stocks (1 Year)")
    plt.show()

    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    df.dropna(inplace=True)
    returns = df.pivot(index="Date", columns="Ticker", values="Return")
    corr = returns.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation of Stock Returns")
    plt.show()

    growth = df.groupby("Ticker")["Close"].agg(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    print("\nPrice Growth (%):\n", growth)
    growth.plot(kind="bar", figsize=(8, 5), color="skyblue")
    plt.title("Overall Price Growth of Stocks (%)")
    plt.ylabel("Growth %")
    plt.xlabel("Ticker")
    plt.show()

    df_feat = feature_engineering(df)
    exclude_cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Tomorrow_Close", "Target"]
    features = handle_correlation(df_feat, exclude_cols)
    joblib.dump(features, "features.pkl")
    X = df_feat[features]
    y = df_feat["Target"]

    X_train, X_test, y_train, y_test, test_mask = train_test_split_with_smote(df_feat, X, y)

    pipe_rf = Pipeline([("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))])
    pipe_xgb = Pipeline([("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))])
    pipe_lr = Pipeline([("scaler", RobustScaler()), ("clf", LogisticRegression(solver="liblinear", random_state=42, class_weight="balanced"))])
    pipe_svm = Pipeline([("scaler", RobustScaler()), ("clf", SVC(probability=True, random_state=42, class_weight="balanced"))])
    param_grids = {
        "RandomForest": {
            "clf__n_estimators": [100, 300],#, 500],
            "clf__max_depth": [ 10, None],#, 20, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__max_features": ["sqrt", "log2"]
        },
        "XGB": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5, 8],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__min_child_weight": [1, 5]
        },
        "LogisticRegression": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l1", "l2"]
        },
        "SVM": {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf"],
            "clf__gamma": ["scale", "auto"]
        }
    }
    tscv = TimeSeriesSplit(n_splits=5)
    best_models = {}
    for name, (pipe, grid) in zip(["RandomForest", "XGB", "LogisticRegression", "SVM"], 
                                  [(pipe_rf, param_grids["RandomForest"]), 
                                   (pipe_xgb, param_grids["XGB"]), 
                                   (pipe_lr, param_grids["LogisticRegression"]), 
                                   (pipe_svm, param_grids["SVM"])]):
        logger.info("Tuning %s", name)
        gs = GridSearchCV(pipe, param_grid=grid, cv=tscv, scoring="roc_auc", refit=True, n_jobs=-1, verbose=2)
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_
        joblib.dump(gs.best_estimator_, f"best_model_{name}.pkl")
        logger.info("Best %s params: %s, AUC=%.4f", name, gs.best_params_, gs.best_score_)
    joblib.dump(pipe_lr.named_steps["scaler"], "scaler_lr.pkl")
    joblib.dump(pipe_svm.named_steps["scaler"], "scaler_svm.pkl")


    logger.info("Computing cross-validation results")
    cv_results = {}
    for name, model in best_models.items():
        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_cv_train, X_cv_test = X.iloc[train_idx], X.iloc[test_idx]
            y_cv_train, y_cv_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_cv_train, y_cv_train)
            y_cv_prob = model.predict_proba(X_cv_test)[:, 1]
            cv_scores.append(roc_auc_score(y_cv_test, y_cv_prob))
        cv_results[name] = {"Mean AUC": np.nanmean(cv_scores), "Std AUC": np.nanstd(cv_scores)}
    cv_df = pd.DataFrame(cv_results).T
    print("\n=== Cross-Validation Results ===")
    print(cv_df)

 
    logger.info("Evaluating models")
    results = {}
    roc_curves = {}
    feature_importances_dict = {}
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    for name, model in best_models.items():
        print(f"\nEvaluating {name} on Test Set...")
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.45).astype(int)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['weighted avg']['f1-score']
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {
            "Accuracy": acc,
            "F1": f1,
            "AUC": auc,
            "Precision (Class 1)": report["1"]["precision"],
            "Recall (Class 1)": report["1"]["recall"],
            "Confusion": confusion_matrix(y_test, y_pred),
            "Report": classification_report(y_test, y_pred)
        }
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        if hasattr(model.named_steps["clf"], "feature_importances_"):
            feature_importances_dict[name] = pd.Series(model.named_steps["clf"].feature_importances_, index=features).sort_values(ascending=False)
        elif hasattr(model.named_steps["clf"], "coef_"):
            feature_importances_dict[name] = pd.Series(np.abs(model.named_steps["clf"].coef_[0]), index=features).sort_values(ascending=False)
    plt.title("ROC Curves - Final Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()
    summary_df = pd.DataFrame({
        name: {
            "Accuracy": res["Accuracy"],
            "F1": res["F1"],
            "AUC": res["AUC"],
        }
        for name, res in results.items()
    }).T.sort_values("AUC", ascending=False)
    print("\n=== Model Benchmark Summary (Test Set) ===")
    print(summary_df)
    summary_df.to_csv("model_comparison.csv")

    best_model_name = summary_df.index[0]
    test_df = df_feat[test_mask].copy()
    test_df["Predicted"] = best_models[best_model_name].predict(X_test)
    test_df["Actual"] = y_test
    test_df["Returns"] = test_df["Tomorrow_Close"] / test_df["Close"] - 1
    transaction_cost = 0.001
    test_df["Strategy_Returns"] = test_df["Returns"] * test_df["Predicted"] - transaction_cost * test_df["Predicted"]
    metrics = trading_metrics(test_df)
    print(f"\nTrading Metrics ({best_model_name}):\n{metrics}")
    logger.info("Trading metrics: %s", metrics)
    top_tickers = test_df["Ticker"].value_counts().head(5).index
    fig, axes = plt.subplots(len(top_tickers), 1, figsize=(15, 5*len(top_tickers)))
    for ax, ticker in zip(np.atleast_1d(axes), top_tickers):
        ticker_df = test_df[test_df["Ticker"] == ticker]
        ax.plot(ticker_df["Date"], ticker_df["Close"], label="Close", linestyle="-")
        ax.plot(ticker_df["Date"], ticker_df["Tomorrow_Close"], label="Tomorrow Close", linestyle="--")
        ax.scatter(ticker_df[ticker_df["Predicted"] == 1]["Date"],
                   ticker_df[ticker_df["Predicted"] == 1]["Close"],
                   color="green", label="Buy Signal", marker="^")
        ax.set_title(f"Price Movement with Buy Signals - {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

