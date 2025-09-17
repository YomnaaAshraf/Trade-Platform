Stock Prediction & Trading Strategy Platform
🔍 Overview

This project implements a machine learning pipeline for stock prediction and trading simulation.
It leverages historical stock data, feature engineering, correlation handling, and multiple ML models to predict next-day stock movements and evaluate a trading strategy.

🏗️ System Architecture

The platform is designed as a modular pipeline:

            ┌─────────────────────┐
            │   Data Fetching     │  ← yfinance
            └─────────┬───────────┘
                      │
            ┌─────────▼───────────┐
            │ Feature Engineering │  ← returns, lagged prices, SMA, volatility, etc.
            └─────────┬───────────┘
                      │
            ┌─────────▼───────────┐
            │ Correlation Handling│  ← removes redundant features
            └─────────┬───────────┘
                      │
            ┌─────────▼───────────┐
            │ Train/Test Split    │  ← time-series aware split + SMOTE
            └─────────┬───────────┘
                      │
   ┌──────────────────▼──────────────────┐
   │      Model Training & Tuning        │
   │ RandomForest | XGBoost | LogisticReg│
   │             | SVM                   │
   └──────────────────┬──────────────────┘
                      │
            ┌─────────▼───────────┐
            │    Model Evaluation │  ← AUC, F1, Accuracy, ROC Curves
            └─────────┬───────────┘
                      │
            ┌─────────▼───────────┐
            │ Trading Simulation  │  ← CAGR, Sharpe Ratio, MaxDD
            └─────────────────────┘

            

🔄 Pipeline Steps

Data Fetching

Uses yfinance to download historical OHLCV stock data.

Feature Engineering

Daily returns, lagged returns, moving averages (SMA), volatility, volume changes, etc.

Target = 1 if tomorrow’s close > today’s close * 1.01, else 0.

Correlation Handling

Drops highly correlated features (>0.8) to prevent redundancy and multicollinearity.

Train/Test Split with SMOTE

Time-based split (80% train / 20% test).

Applies SMOTE for balancing if target distribution is skewed.

Model Training & Tuning

Models: RandomForest, XGBoost, Logistic Regression, SVM

Hyperparameter optimization with TimeSeriesSplit cross-validation.

Saves best models via joblib.

Model Evaluation

Classification metrics: Accuracy, F1, AUC, Precision, Recall.

Visualizations: ROC curves, feature importance.

Trading Simulation

Applies predictions to simulate a long-only trading strategy.

Considers transaction cost = 0.1%.

Metrics: CAGR, Sharpe Ratio, Max Drawdown, Total Return.

⚙️ Installation

Clone the repository and install dependencies:

git clone [https://github.com/your-repo/stock-prediction-platform.git](https://github.com/YomnaaAshraf/Trade-Platform/tree/main?tab=readme-ov-file)
cd Trade-platform
pip install -r requirements.txt

▶️ Usage

Run the pipeline end-to-end:

python main.py


This will:

Fetch stock data

Engineer features

Train multiple models

Save best models in .pkl files

Generate evaluation results & trading metrics
